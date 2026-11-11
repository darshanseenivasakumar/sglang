"""The sparse-attention page layout of the unified pool: each page holds its
tokens' latent rows, then one indexer key blob per indexer layer, padded so
that latent rows keep a grid of their own width across pages."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    DSAGeometry,
    MLAGeometry,
    StateGeometry,
    kv_layer_decls,
)
from sglang.srt.mem_cache.kv.layout.dsa import DSAMLASubPoolSpec, DSAMLAView
from sglang.srt.mem_cache.kv.memory.snapshot import _whole_pages
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE = 4
_ROW = 32
_BLOB_PER_TOKEN = 12


def _spec(**overrides):
    fields = dict(
        name="full",
        layer_num=3,
        grow_direction="down",
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        row_bytes=_ROW,
        page_size=_PAGE,
        index_blob_bytes=_PAGE * _BLOB_PER_TOKEN,
        index_layers=(0, 2),
    )
    fields.update(overrides)
    return DSAMLASubPoolSpec(**fields)


class TestPageArithmetic(CustomTestCase):
    def test_kimi_k31_page(self):
        """24 layers of 656-byte FP8 rows and an 8448-byte index blob per layer
        fill 1,210,368 bytes; the page rounds up to 1848 rows of 656 bytes."""
        spec = _spec(
            layer_num=24,
            row_bytes=656,
            page_size=64,
            index_blob_bytes=64 * 132,
            index_layers=tuple(range(24)),
        )
        self.assertEqual(spec.token_bytes(), 24 * 656)
        self.assertEqual(spec.page_bytes(), 1_212_288)
        self.assertEqual(spec.rows_per_page(), 1848)
        self.assertEqual(spec.entry_bytes() * 64, spec.page_bytes())

    def test_row_index_follows_pages(self):
        spec = _spec()
        # 4 tokens x 96 bytes of rows + 2 blobs of 48 bytes = 480 = 15 rows.
        self.assertEqual(spec.page_bytes(), 480)
        loc = torch.tensor([0, 1, 3, 4, 9, -1])
        self.assertEqual(spec.row_index(loc).tolist(), [0, 3, 9, 15, 33, -1])

    def test_runs_reach_every_token_with_few_layers(self):
        """With two latent layers the index blobs alone leave fewer than 63 rows
        after the token rows; the page is padded so that every layer's run, cut
        to whole 64-row blocks, still reaches the last page's last token."""
        spec = _spec(
            layer_num=2,
            row_bytes=656,
            page_size=64,
            index_blob_bytes=64 * 132,
            index_layers=(0, 1),
        )
        rows = spec.rows_per_page()
        self.assertGreaterEqual(rows - 64 * 2, 63)
        for pages in range(1, 40):
            last = int(spec.row_index(torch.tensor([pages * 64 - 1])))
            for layer in range(2):
                run = (pages * rows - layer) // 64 * 64
                self.assertLess(last, run, (pages, layer))

    def test_pages_are_padded_to_the_row_grid(self):
        spec = _spec(index_layers=(0,))
        self.assertEqual(spec.page_bytes() % _ROW, 0)
        self.assertGreaterEqual(
            spec.page_bytes(), spec.index_offset() + spec.index_blob_bytes
        )


class TestViewsOverTheArena(CustomTestCase):
    def _storage(self):
        setup_allocator_context()
        dsa = DSAGeometry(
            row_bytes=_ROW,
            scale_ue8m0=True,
            index_head_dim=8,
            index_bytes_per_token=_BLOB_PER_TOKEN,
            index_key_bytes_per_token=8,
            use_fp4_indexer=False,
            index_layer_ids=(0, 4),
        )
        plan = plan_kv_storage(
            kv_layer_decls(
                full_layer_ids=(0, 2, 4),
                full_geometry=MLAGeometry(
                    kv_lora_rank=8,
                    qk_rope_head_dim=8,
                    kv_cache_dtype=torch.float8_e4m3fn,
                    dsa=dsa,
                ),
                state_layer_ids=(1, 3),
                state_geometry=StateGeometry(
                    conv_state_shapes=((3, 8),),
                    conv_dtype=torch.bfloat16,
                    temporal_state_shape=(1, 4, 8),
                    temporal_dtype=torch.float32,
                ),
            ),
            page_size=_PAGE,
        )
        sized = size_plan(
            plan,
            KVSizingInputs(
                full_tokens=64,
                state_slots=4,
                max_running_requests=2,
                context_len=64,
                alignment=1,
            ),
        )
        storage = build_kv_storage(
            plan,
            sized,
            UnifiedBuildContext(
                device="cpu",
                start_layer=0,
                end_layer=5,
                enable_memory_saver=False,
                need_sort=False,
                max_num_reqs=2,
                model_context_len=64,
                state_cache_params=SimpleNamespace(layers=(1, 3)),
            ),
        )
        return plan, storage

    def test_the_plan_lays_out_the_full_sub_pool_by_page(self):
        plan, storage = self._storage()
        spec = plan.owner().spec
        self.assertIsInstance(spec, DSAMLASubPoolSpec)
        # Model layers 0 and 4 keep index keys: view layers 0 and 2.
        self.assertEqual(spec.index_layers, (0, 2))
        view = storage.view("full")
        self.assertIsInstance(view, DSAMLAView)
        self.assertEqual(view.skip_topk_layers, [False, True, False])
        self.assertEqual(view.page_bytes, spec.page_bytes())

    def test_rows_and_blobs_land_in_their_token_page(self):
        """Every token's rows and every page's blobs occupy disjoint bytes inside
        that page's unit of the arena."""
        plan, storage = self._storage()
        spec = plan.owner().spec
        view = storage.view("full")
        arena = storage.view("full")._arena
        raw = arena.raw
        raw.zero_()
        first = arena.first_entry("full") // _PAGE + 1
        pages = range(first, first + 3)
        stamp = 1
        expected = {}
        for page in pages:
            for slot in range(_PAGE):
                loc = torch.tensor([page * _PAGE + slot])
                row = int(view.row_index(loc))
                for layer in range(spec.layer_num):
                    view.kv_buffer[layer][row] = stamp
                    start = page * spec.page_bytes() + slot * spec.token_bytes()
                    start += layer * _ROW
                    expected[stamp] = range(start, start + _ROW)
                    stamp += 1
            for layer in spec.index_layers:
                blob = view.get_index_k_with_scale_buffer(layer)
                self.assertEqual(blob.stride(0), spec.page_bytes())
                blob[page] = stamp
                start = page * spec.page_bytes() + spec.index_offset()
                start += spec.index_layers.index(layer) * spec.index_blob_bytes
                expected[stamp] = range(start, start + spec.index_blob_bytes)
                stamp += 1
        for value, span in expected.items():
            self.assertTrue(
                bool((raw[span.start : span.stop] == value).all()), f"stamp {value}"
            )
        self.assertEqual(int((raw != 0).sum()), sum(map(len, expected.values())))
        for page in pages:
            unit = raw[page * spec.page_bytes() : (page + 1) * spec.page_bytes()]
            self.assertEqual(
                set(unit.unique().tolist()) - {0},
                {
                    v
                    for v, span in expected.items()
                    if span.start // spec.page_bytes() == page
                },
            )

    def test_layers_without_index_keys_refuse_them(self):
        _, storage = self._storage()
        with self.assertRaisesRegex(AssertionError, "keeps no indexer keys"):
            storage.view("full").get_index_k_with_scale_buffer(1)


class TestRetractionCopiesWholePages(CustomTestCase):
    def test_a_partial_page_is_copied_whole(self):
        ids = torch.tensor([8, 9, 10, 11, 12, 13])
        self.assertEqual(_whole_pages(ids, 4).tolist(), [8, 9, 10, 11, 12, 13, 14, 15])


if __name__ == "__main__":
    unittest.main()
