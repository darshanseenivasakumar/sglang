"""The sparse-attention (DSA) MLA layout: each page holds its tokens' latent rows
and, after them, every indexer layer's key blob for those tokens.

One page of ``page_size`` tokens, ``L`` latent layers and ``I`` indexer layers:

    [ token 0: row_0 .. row_{L-1} | ... | token P-1: row_0 .. row_{L-1} ]
    [ index blob of indexer layer 0 | ... | of indexer layer I-1 ] [ pad ]

A latent row is ``row_bytes`` long (FP8 NoPE with per-block scales, then the
RoPE tail) and an index blob is the page's keys followed by their scales, as
the paged index kernels read it. The page is padded to a multiple of
``row_bytes``, so every latent row of the band sits on a ``row_bytes`` grid
counted from the arena's first byte: a layer's rows form one flat run of rows,
and the row of token ``t`` in layer ``l`` is ``row_index(t)`` in layer ``l``'s
run. Readers and writers that address rows by token take ``row_index`` of the
token ids; the index blobs are pages ``page_bytes`` apart.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton
from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache_separate
from sglang.srt.mem_cache.kv.layout.base import (
    SubPoolSpec,
    check_physical_id_bound,
    dense_rows,
)
from sglang.srt.mem_cache.kv.layout.mla import MLAView
from sglang.srt.mem_cache.layout.token_major import ROW_ALIGN_BYTES
from sglang.srt.mem_cache.memory_pool import unwrap_write_loc


@dataclass(frozen=True, kw_only=True)
class DSAMLASubPoolSpec(SubPoolSpec):
    """Page layout of an MLA sub-pool whose layers attend sparsely.

    ``index_layers`` are the view-local indices (0..layer_num-1) of the layers
    that keep indexer keys, in blob order.
    """

    zero_units_on_handout = True
    copies_whole_units = True

    kv_lora_rank: int
    qk_rope_head_dim: int
    row_bytes: int
    page_size: int
    index_blob_bytes: int
    index_layers: Tuple[int, ...]

    def __post_init__(self):
        super().__post_init__()
        assert self.draft_region is None, (
            "a fused draft region does not ride in sparse-attention pages"
        )
        assert self.row_bytes % ROW_ALIGN_BYTES == 0, (
            f"latent rows of {self.row_bytes} bytes are not "
            f"{ROW_ALIGN_BYTES}-byte aligned"
        )
        assert self.index_blob_bytes % ROW_ALIGN_BYTES == 0
        assert all(0 <= i < self.layer_num for i in self.index_layers)
        assert len(set(self.index_layers)) == len(self.index_layers)

    @property
    def kv_cache_dim(self) -> int:
        return self.row_bytes

    def token_bytes(self) -> int:
        """Latent bytes of one token over every layer."""
        return self.layer_num * self.row_bytes

    def index_offset(self) -> int:
        """Byte offset of the first index blob inside a page."""
        return self.page_size * self.token_bytes()

    def page_bytes(self) -> int:
        """One page, padded so that pages keep the latent rows on their grid
        and split into whole token entries. At least ``page_size - 1`` rows follow
        the token rows, so that a layer's run of rows, cut to whole blocks of
        ``page_size`` rows, still reaches the last page's last token."""
        tail = max(
            len(self.index_layers) * self.index_blob_bytes,
            (self.page_size - 1) * self.row_bytes,
        )
        grid = math.lcm(self.row_bytes, self.page_size)
        return -(-(self.index_offset() + tail) // grid) * grid

    def entry_bytes(self) -> int:
        # The arena counts a band in token entries; a page is `page_size` of them.
        return self.page_bytes() // self.page_size

    def rows_per_page(self) -> int:
        return self.page_bytes() // self.row_bytes

    def row_index(self, loc: torch.Tensor) -> torch.Tensor:
        """The row of each token id in a layer's run of rows; -1 stays -1."""
        page, slot = loc // self.page_size, loc % self.page_size
        rows = page * self.rows_per_page() + slot * self.layer_num
        return torch.where(loc < 0, loc, rows)

    def views(self, arena) -> List[torch.Tensor]:
        """Per-layer runs of latent rows, ``(rows, 1, row_bytes)`` bytes. Each run
        holds a whole number of ``page_size`` rows, as paged readers view it, and
        still reaches every token's row."""
        pages = dense_rows(arena, self.name) // self.page_size
        rows = pages * self.rows_per_page()
        check_physical_id_bound(sub_pool=self.name, n_rows=rows)
        # The last token's row of the last page, in the last layer's run.
        last = (pages - 1) * self.rows_per_page() + (
            self.page_size - 1
        ) * self.layer_num
        runs = []
        for layer in range(self.layer_num):
            run = (rows - layer) // self.page_size * self.page_size
            assert run > last, "a paged run of rows must reach every token"
            runs.append(
                torch.as_strided(
                    arena.raw,
                    (run, 1, self.row_bytes),
                    (self.row_bytes, self.row_bytes, 1),
                    layer * self.row_bytes,
                )
            )
        return runs

    def index_views(self, arena) -> List[torch.Tensor]:
        """Per indexer layer, its blobs as ``(pages, blob)`` bytes, one row per
        physical page."""
        pages = dense_rows(arena, self.name) // self.page_size
        return [
            torch.as_strided(
                arena.raw,
                (pages, self.index_blob_bytes),
                (self.page_bytes(), 1),
                self.index_offset() + i * self.index_blob_bytes,
            )
            for i in range(len(self.index_layers))
        ]

    def get_dtype(self) -> torch.dtype:
        return torch.uint8


class DSAMLAView(MLAView):
    """The latent rows and indexer keys of a sparse-attention MLA sub-pool.

    Latent rows are FP8 with per-block scales, quantized on write. Per-layer
    buffers are runs of rows addressed by ``row_index`` of the physical token
    id, not by the id itself.
    """

    use_dsa = True
    dsa_kv_cache_store_fp8 = True
    quant_block_size = 128
    index_kpool = 1
    index_kpool_compress = False
    tail_extra_slots = 0
    # Readers address this view's rows through `row_index`.
    addresses_rows_by_index = True

    def __init__(
        self,
        *,
        arena,
        spec: DSAMLASubPoolSpec,
        kv_cache_dtype: torch.dtype,
        page_size: int,
        scale_ue8m0: bool,
        index_head_dim: int,
        index_bytes_per_token: int,
        index_key_bytes_per_token: int,
        use_fp4_indexer: bool,
        model_layer_ids: Optional[Tuple[int, ...]] = None,
    ):
        assert kv_cache_dtype == torch.float8_e4m3fn, (
            "sparse-attention latent rows are stored as FP8 with block scales"
        )
        assert spec.page_size == page_size
        self._dsa_spec = spec
        self._init_view(
            arena=arena,
            spec=spec,
            kv_cache_dtype=kv_cache_dtype,
            store_dtype=torch.uint8,
            page_size=page_size,
            model_layer_ids=model_layer_ids,
        )
        self.page_bytes = spec.page_bytes()
        self.kv_scale_ue8m0 = scale_ue8m0
        self.index_head_dim = index_head_dim
        self.index_bytes_per_token = index_bytes_per_token
        self.index_key_bytes_per_token = index_key_bytes_per_token
        self.use_fp4_indexer = use_fp4_indexer
        self.index_page_size = page_size
        self.skip_topk_layers = [
            layer not in spec.index_layers for layer in range(spec.layer_num)
        ]
        index_views = spec.index_views(arena)
        self._index_buffer = [None] * spec.layer_num
        for i, layer in enumerate(spec.index_layers):
            self._index_buffer[layer] = index_views[i]

    def row_index(self, loc: torch.Tensor) -> torch.Tensor:
        return self._dsa_spec.row_index(loc)

    # -- latent rows --

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.kv_buffer[layer_id].view(self.dtype)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError(
            "sparse-attention latent rows are block-quantized; read them through "
            "the sparse attention kernels"
        )

    def get_mla_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError(
            "sparse-attention latent rows are block-quantized; dequantize them "
            "through the sparse attention gather"
        )

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "sparse-attention latent rows are written through set_mla_kv_buffer"
        )

    def set_mla_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        layer_id_override: Optional[int] = None,
    ) -> None:
        """Quantizes the NoPE part per block, then stores the rows."""
        loc, _, _ = unwrap_write_loc(loc_info)
        self._check_write_loc(loc_info, loc, "set_mla_kv_buffer (DSA MLA view)")
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        nope, rope = quantize_k_cache_separate(
            cache_k_nope, cache_k_rope, scale_ue8m0=self.kv_scale_ue8m0
        )
        set_mla_kv_buffer_triton(
            self.kv_buffer[layer_id], self.row_index(loc), nope, rope
        )

    # -- indexer keys --

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        buffer = self._index_buffer[layer_id]
        assert buffer is not None, f"layer {layer_id} keeps no indexer keys"
        return buffer

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        from sglang.kernels.ops.kvcache.dsa_index_k_cache import GetKAndS

        return GetKAndS.execute(
            self,
            self.get_index_k_with_scale_buffer(layer_id),
            seq_len_tensor=seq_len_tensor,
            page_indices=page_indices,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        from sglang.kernels.ops.kvcache.dsa_index_k_cache import SetKAndS

        SetKAndS.execute(
            pool=self,
            buf=self.get_index_k_with_scale_buffer(layer_id),
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
        )

    # -- host transfer and retraction --

    def host_layout(self):
        from sglang.srt.mem_cache.kv.layout.page import page_envelope_host_layout

        return page_envelope_host_layout(
            self._arena,
            self._sub_pool_name,
            self.size
            if self.host_capacity_tokens is None
            else self.host_capacity_tokens,
            self.page_size,
        )

    def get_device_pool_infos(self):
        raise NotImplementedError(
            "a sparse-attention page is moved whole: use its page envelope"
        )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError(
            "retraction backs a unified pool up by whole units in its arena"
        )

    def load_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError(
            "retraction backs a unified pool up by whole units in its arena"
        )
