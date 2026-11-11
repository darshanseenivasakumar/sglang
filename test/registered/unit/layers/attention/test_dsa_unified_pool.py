"""DSA attention on the unified memory pool, on CPU.

`req_to_token` holds virtual token ids there, which the backend's page tables
translate once into the physical ids of the full sub-pool: the indexer, the
top-k transforms and attention then all read physical ids. The pool keeps a
token's latent row at `row_index` of its physical id (each page interleaves its
tokens' rows across layers), so every reader of latent rows by token id takes
the mapped rows. A static pool keeps both maps the identity.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _validate_unified_pool_backend,
)
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.layout.dsa import DSAMLASubPoolSpec
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4
REQ_ROWS, WIDTH, VIRTUAL_PAGES = 4, 32, 40


def _virtual_batch():
    """`req_to_token` over shuffled virtual pages, and a v2p that scatters them
    over physical pages (virtual page 0 and the trailing row stay the sink and
    the -1 sentinel)."""
    g = torch.Generator().manual_seed(7)
    pages = torch.randperm(VIRTUAL_PAGES - 1, generator=g)[: REQ_ROWS * WIDTH // PAGE]
    pages = (pages + 1).view(REQ_ROWS, WIDTH // PAGE)
    req_to_token = (
        (pages[:, :, None] * PAGE + torch.arange(PAGE)).view(REQ_ROWS, WIDTH)
    ).to(torch.int32)
    v2p = torch.full((VIRTUAL_PAGES + 1,), -1, dtype=torch.int64)
    v2p[0] = 0
    v2p[1:VIRTUAL_PAGES] = torch.randperm(VIRTUAL_PAGES - 1, generator=g) + 100
    return req_to_token, v2p


def _reference(ids, v2p):
    ids = ids.to(torch.int64)
    return (v2p[ids // PAGE] * PAGE + ids % PAGE).to(torch.int32)


def _runner_ids(v2p):
    """A unified target's RunnerIds, as the DSA backend reads it."""
    space = IdSpace(key=("full", 0), read_v2p=v2p, table=v2p)
    return SimpleNamespace(
        reads_are_translated=True,
        is_translating=True,
        page_size=PAGE,
        space=lambda kind: space if kind == IdSpaceKind.FULL else None,
    )


def _routed(pool):
    pool.start_layer, pool.layer_transfer_counter = 0, None
    pool.layer_router = PlainPoolRouter(pool, IdSpaceKind.FULL)
    return pool


def _row_indexed_pool(get_key_buffer=None):
    """The unified pool's sparse-attention view: K3.1's 656-byte rows of 24
    layers in 64-token pages."""
    spec = DSAMLASubPoolSpec(
        name="full",
        layer_num=24,
        grow_direction="down",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        row_bytes=656,
        page_size=64,
        index_blob_bytes=64 * 132,
        index_layers=tuple(range(24)),
    )
    return _routed(
        SimpleNamespace(
            addresses_rows_by_index=True,
            row_index=spec.row_index,
            get_key_buffer=get_key_buffer,
        )
    ), spec


def _static_pool(get_key_buffer=None):
    return _routed(
        SimpleNamespace(addresses_rows_by_index=False, get_key_buffer=get_key_buffer)
    )


def _metadata_backend(req_to_token, runner_ids):
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.__dict__.update(
        device=torch.device("cpu"),
        runner_ids=runner_ids,
        req_to_token=req_to_token,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=SimpleNamespace(
            size=8, page_size=PAGE, dtype=torch.float8_e4m3fn
        ),
        _kv_shard_pool=None,
        physical_page_size=PAGE,
        real_page_span=PAGE,
        dsa_index_kpool=1,
        dsa_index_topk=16,
        dsa_shards_index_k=False,
        attn_dcp_size=1,
        attn_dcp_rank=0,
        dsa_kv_cache_store_fp8=True,
        dsa_prefill_impl="flashmla_sparse",
        dsa_decode_impl="tilelang",
        enable_auto_select_prefill_impl=False,
        supports_mha_one_shot=False,
        use_mha=False,
        hisparse_coordinator=None,
        speculative_num_draft_tokens=0,
        dsa_topk_backend=DSATopKBackend.TORCH,
        _arange_buf=torch.arange(64, dtype=torch.int32),
        _is_in_breakable_cuda_graph=lambda: False,
        _is_in_tc_piecewise_cuda_graph=lambda: False,
        _get_device_sm=lambda: 100,
        _is_blackwell=lambda: True,
    )
    return backend


def _extend_batch(req_pool_indices, seq_lens, prefix_lens):
    extend = [s - p for s, p in zip(seq_lens, prefix_lens)]
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        req_pool_indices=torch.tensor(req_pool_indices),
        seq_lens=torch.tensor(seq_lens),
        seq_lens_cpu=torch.tensor(seq_lens),
        seq_lens_sum=sum(seq_lens),
        extend_seq_lens=torch.tensor(extend, dtype=torch.int32),
        extend_seq_lens_cpu=extend,
        extend_prefix_lens_cpu=prefix_lens,
        global_num_tokens_cpu=None,
        attn_cp_metadata=None,
    )


class TestPhysicalPageTables(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def _extend(self, runner_ids):
        req_to_token, v2p = _virtual_batch()
        backend = _metadata_backend(req_to_token, runner_ids(v2p))
        rows, seq_lens = [2, 0, 3], [13, 7, 21]
        backend.init_forward_metadata(_extend_batch(rows, seq_lens, [5, 3, 9]))
        metadata = backend.forward_metadata
        virtual = req_to_token[torch.tensor(rows), : max(seq_lens)]
        return metadata, virtual, v2p, rows, seq_lens

    def test_eager_extend_tables_hold_physical_ids(self):
        """page_table_1, real_page_table and page_table_1_flattened (the prefix
        gather) all name physical ids; the capacity check, which bounds the
        static pool's ids, does not reject them though the batch outgrows the
        pool's `size`."""
        metadata, virtual, v2p, rows, seq_lens = self._extend(_runner_ids)
        physical = _reference(virtual, v2p)
        self.assertTrue(torch.equal(metadata.page_table_1, physical))
        self.assertTrue(
            torch.equal(metadata.real_page_table, physical[:, ::PAGE] // PAGE)
        )
        flat = torch.cat([physical[i, :n] for i, n in enumerate(seq_lens)])
        self.assertTrue(torch.equal(metadata.page_table_1_flattened, flat))

    def test_static_pool_tables_hold_req_to_token_ids(self):
        req_to_token = torch.arange(REQ_ROWS * WIDTH, dtype=torch.int32).view(
            REQ_ROWS, WIDTH
        )
        backend = _metadata_backend(req_to_token, None)
        backend.token_to_kv_pool.size = REQ_ROWS * WIDTH
        rows, seq_lens = [2, 0], [13, 7]
        backend.init_forward_metadata(_extend_batch(rows, seq_lens, [5, 3]))
        metadata = backend.forward_metadata
        expected = req_to_token[torch.tensor(rows), :13]
        self.assertTrue(torch.equal(metadata.page_table_1, expected))
        self.assertTrue(
            torch.equal(metadata.real_page_table, expected[:, ::PAGE] // PAGE)
        )

    def test_eager_decode_tables_hold_physical_ids(self):
        req_to_token, v2p = _virtual_batch()
        backend = _metadata_backend(req_to_token, _runner_ids(v2p))
        rows, seq_lens = [3, 1], [17, 6]
        # The paged logits schedule is a GPU kernel's; these tables are not.
        with patch(
            "sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=False
        ):
            backend.init_forward_metadata(
                SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    batch_size=2,
                    req_pool_indices=torch.tensor(rows),
                    seq_lens=torch.tensor(seq_lens),
                    seq_lens_cpu=torch.tensor(seq_lens),
                    seq_lens_sum=sum(seq_lens),
                    global_num_tokens_cpu=None,
                    attn_cp_metadata=None,
                )
            )
        metadata = backend.forward_metadata
        physical = _reference(req_to_token[torch.tensor(rows), :17], v2p)
        self.assertTrue(torch.equal(metadata.page_table_1, physical))
        self.assertTrue(
            torch.equal(metadata.real_page_table, physical[:, ::PAGE] // PAGE)
        )

    def test_unmapped_pages_read_the_sink(self):
        req_to_token, v2p = _virtual_batch()
        backend = _metadata_backend(req_to_token, _runner_ids(v2p))
        dropped = int(req_to_token[1, 0]) // PAGE
        v2p[dropped] = -1
        table = backend._physical_token_table(req_to_token[1:2, :PAGE])
        self.assertEqual(table.tolist(), [[0] * PAGE])


class TestLatentRowsByIndex(CustomTestCase):
    def _backend(self, pool):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.token_to_kv_pool = pool
        return backend

    def test_flashmla_kv_indexes_rows(self):
        """FlashMLA reads row `i` of the layer's rows for index `i`: on the
        unified pool that is `row_index` of the top-k's token ids, -1 kept."""
        topk = torch.tensor([[0, 65, 130, -1], [3, 64, -1, -1]], dtype=torch.int32)
        pool, spec = _row_indexed_pool()
        expected = spec.row_index(topk)
        self.assertEqual(expected[0, 1].item(), 1848 + 24)
        self.assertTrue(torch.equal(self._indices(pool, topk), expected))
        self.assertTrue(torch.equal(self._indices(_static_pool(), topk), topk))

    def _indices(self, pool, topk):
        backend = self._backend(pool)
        seen = {}

        def fake_flash_mla(**kwargs):
            seen.update(kwargs)
            q = kwargs["q"]
            return q.new_zeros(*q.shape[:3], 512), q.new_zeros(q.shape[0], 1)

        backend.__dict__.update(
            flashmla_kv_num_q_heads=2,
            dsa_shards_index_k=False,
            physical_page_size=64,
            kv_cache_dim=656,
            dsa_kv_cache_store_fp8=True,
            dsa_index_topk=topk.shape[1],
            flashmla_kv_format="V32",
            _call_flash_mla_with_kvcache=fake_flash_mla,
        )
        rows = topk.shape[0]
        backend._forward_flashmla_kv(
            q_all=torch.zeros(rows, 2, 576),
            kv_cache=torch.zeros(2 * 64, 1, 656, dtype=torch.uint8),
            v_head_dim=512,
            sm_scale=1.0,
            layer=SimpleNamespace(tp_q_head_num=2, head_dim=576),
            metadata=SimpleNamespace(
                dsa_cache_seqlens_int32=torch.full((rows,), 4, dtype=torch.int32),
                flashmla_metadata=SimpleNamespace(
                    flashmla_metadata=None, num_splits=None
                ),
            ),
            page_table_1=topk,
            forward_batch=SimpleNamespace(),
        )
        return seen["indices"].squeeze(1)

    def test_ragged_prefix_dequant_indexes_rows(self):
        """A prefix-sharing RAGGED prefill dequantizes the batch's latent rows
        through the prefix gather, which takes their rows."""
        flat = torch.tensor([64, 65, 66, 200, 201], dtype=torch.int32)
        pool, spec = _row_indexed_pool(lambda _: torch.zeros(4 * 64, 1, 656))
        self.assertTrue(
            torch.equal(self._dequant_rows(pool, flat), spec.row_index(flat))
        )
        static = _static_pool(lambda _: torch.zeros(4 * 64, 1, 656))
        self.assertTrue(torch.equal(self._dequant_rows(static, flat), flat))

    def _dequant_rows(self, pool, flat):
        backend = self._backend(pool)
        seen = []
        tokens = 3
        backend.__dict__.update(
            dsa_prefill_impl="flashmla_sparse",
            dsa_shards_index_k=False,
            use_mha=False,
            use_fused_topk=True,
            hisparse_coordinator=None,
            forward_metadata=SimpleNamespace(
                page_table_1_flattened=flat,
                dsa_cache_seqlens_int32=torch.full((tokens,), 2, dtype=torch.int32),
            ),
            _resolve_kpool_tail_backend=lambda indices, impl: impl,
            _check_kpool_tail_backend=lambda *args: None,
            get_topk_transform_method=lambda _: TopkTransformMethod.RAGGED,
            _pad_topk_indices=lambda indices, n: indices[:n],
            _get_fused_topk_page_table=lambda indices: indices,
            _forward_flashmla_sparse=lambda **kwargs: kwargs["q_all"],
        )

        def fake_dequant(kv_cache, rows):
            seen.append(rows)
            return torch.zeros(rows.shape[0], 1, 576)

        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            tp_q_head_num=2,
            v_head_dim=512,
            head_dim=576,
            scaling=1.0,
        )
        with (
            patch(
                "sglang.srt.layers.attention.dsa_backend.dequantize_k_cache_paged_dsa",
                side_effect=fake_dequant,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.concat_mla_absorb_q_general",
                side_effect=lambda q, rope: torch.cat((q, rope), dim=-1),
            ),
        ):
            backend.forward_extend(
                q=torch.zeros(tokens, 2 * 512),
                k=torch.zeros(tokens, 1, 512),
                v=torch.zeros(tokens, 1, 512),
                layer=layer,
                forward_batch=SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND, extend_prefix_lens_cpu=[2]
                ),
                save_kv_cache=False,
                q_rope=torch.zeros(tokens, 2, 64),
                k_rope=torch.zeros(tokens, 1, 64),
                topk_indices=torch.zeros(tokens, 2, dtype=torch.int32),
            )
        self.assertEqual(len(seen), 1)
        return seen[0]

    def test_mha_prefix_dequant_indexes_rows(self):
        """MHA one-shot dequantizes the prefix through the same gather."""
        flat = torch.tensor([1, 64, 129], dtype=torch.int32)
        pool, spec = _row_indexed_pool()
        self.assertTrue(torch.equal(self._mha_rows(pool, flat), spec.row_index(flat)))
        self.assertTrue(torch.equal(self._mha_rows(_static_pool(), flat), flat))

    def _mha_rows(self, pool, flat):
        backend = self._backend(pool)
        backend.forward_metadata = SimpleNamespace(page_table_1_flattened=flat)
        seen = []

        def fake_dequant(kv_cache, rows):
            seen.append(rows)
            return torch.zeros(rows.shape[0], 1, 576)

        layers = SimpleNamespace(read=lambda _: SimpleNamespace(key=lambda: None))
        attn = SimpleNamespace(attn_mha=SimpleNamespace(layer_id=0), kv_lora_rank=512)
        with (
            patch.object(forward_mha, "resolve_attn_backend", return_value=backend),
            patch.object(forward_mha, "get_kv_layers", return_value=layers),
            patch.object(forward_mha, "dsa_shards_index_k", return_value=False),
            patch.object(
                forward_mha, "dequantize_k_cache_paged_dsa", side_effect=fake_dequant
            ),
        ):
            forward_mha.DeepseekMHAForwardMixin._get_mla_kv_buffer_from_fp8_for_dsa(
                attn, SimpleNamespace()
            )
        self.assertEqual(len(seen), 1)
        return seen[0]


class TestUnifiedPoolRefusals(CustomTestCase):
    SERVED = dict(
        rows_by_index=True,
        translates_ids=True,
        prefill_impl="flashmla_kv",
        decode_impl="flashmla_kv",
        index_kpool=1,
        dcp_size=1,
        speculative_algorithm=None,
        return_indexer_topk=False,
        hisparse_enabled=False,
    )

    def test_served_configurations_pass(self):
        _validate_unified_pool_backend(**self.SERVED)
        for prefill in ("flashmla_sparse", "flashmla_auto", "triton_sparse_mla"):
            _validate_unified_pool_backend(**{**self.SERVED, "prefill_impl": prefill})

    def test_static_pool_is_not_checked(self):
        _validate_unified_pool_backend(
            **{
                **self.SERVED,
                "rows_by_index": False,
                "translates_ids": False,
                "decode_impl": "trtllm",
                "prefill_impl": "trtllm",
                "dcp_size": 8,
                "index_kpool": 4,
                "speculative_algorithm": "EAGLE",
                "return_indexer_topk": True,
            }
        )

    def test_refusals(self):
        cases = (
            ({"decode_impl": "trtllm", "prefill_impl": "trtllm"}, "trtllm"),
            ({"decode_impl": "fa3"}, "--dsa-decode-backend"),
            ({"prefill_impl": "tilelang"}, "--dsa-prefill-backend"),
            ({"index_kpool": 4}, "pooled indexer keys"),
            ({"dcp_size": 2}, "decode context parallelism"),
            ({"hisparse_enabled": True}, "hisparse"),
            ({"return_indexer_topk": True}, "return-indexer-topk"),
            ({"speculative_algorithm": "EAGLE"}, "speculative decoding"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_unified_pool_backend(**{**self.SERVED, **overrides})


if __name__ == "__main__":
    unittest.main()
