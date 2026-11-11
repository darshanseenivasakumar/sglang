"""The sparse-attention page layout of the unified pool: each page holds its
tokens' latent rows layer by layer, then one indexer key blob per indexer layer,
padded to whole rows and whole token entries. A layer's rows are a paged buffer
whose pages lie a page entry apart."""

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
        """24 layers of 64 656-byte FP8 rows and an 8448-byte index blob per
        layer fill 1,210,368 bytes; the page rounds up to 1848 rows of 656
        bytes, a whole number of 64-token entries."""
        spec = _spec(
            layer_num=24,
            row_bytes=656,
            page_size=64,
            index_blob_bytes=64 * 132,
            index_layers=tuple(range(24)),
        )
        self.assertEqual(spec.layer_bytes(), 64 * 656)
        self.assertEqual(spec.index_offset(), 24 * 64 * 656)
        self.assertEqual(spec.page_bytes(), 1_212_288)
        self.assertEqual(spec.page_bytes() % 656, 0)
        self.assertEqual(spec.entry_bytes() * 64, spec.page_bytes())

    def test_pages_are_padded_to_whole_rows_and_entries(self):
        spec = _spec(index_layers=(0,))
        self.assertEqual(spec.page_bytes() % _ROW, 0)
        self.assertEqual(spec.page_bytes() % _PAGE, 0)
        self.assertGreaterEqual(
            spec.page_bytes(), spec.index_offset() + spec.index_blob_bytes
        )

    def test_layer_views_are_paged(self):
        """Layer l's buffer is (pages, page_size, 1, row): token t's row is at
        page t // page_size, slot t % page_size, inside layer l's block of the
        page entry."""
        spec = _spec()
        pages = 5
        arena = SimpleNamespace(
            raw=torch.zeros(pages * spec.page_bytes(), dtype=torch.uint8),
            band=lambda name: SimpleNamespace(unit_entries=_PAGE),
            extent_entries=lambda name: pages * _PAGE,
        )
        views = spec.views(arena)
        self.assertEqual(len(views), spec.layer_num)
        for layer, view in enumerate(views):
            self.assertEqual(tuple(view.shape), (pages, _PAGE, 1, _ROW))
            self.assertEqual(view.stride(), (spec.page_bytes(), _ROW, _ROW, 1))
            for token in (0, 1, 5, 4 * _PAGE + 3):
                row = view[token // _PAGE, token % _PAGE, 0]
                offset = row.data_ptr() - arena.raw.data_ptr()
                self.assertEqual(
                    offset,
                    token // _PAGE * spec.page_bytes()
                    + layer * spec.layer_bytes()
                    + token % _PAGE * _ROW,
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
                for layer in range(spec.layer_num):
                    view.kv_buffer[layer][page, slot] = stamp
                    start = page * spec.page_bytes() + layer * spec.layer_bytes()
                    start += slot * _ROW
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
