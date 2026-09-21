"""Ascend causal convolution wrappers for Qwen3.5.

The Ascend implementation adapts RTP-LLM's paged convolution cache to the
layout expected by ``fla_npu``.  All metadata tensors passed to the FLA-NPU
operators are device int32 — no ``.cpu()``, no host-side Python loops — so
that the decode path can participate in stream (graph) capture on Ascend NPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from rtp_llm.models_py.kernels.ascend.state_migration import migrate_state_rows

# 哨兵值 = 保留页 0：引擎 BlockPool 保留块 0（真实块 ID 从 1 开始）、块表零
# 初始化，0 不会与真实页冲突；同时也是 fla_npu npu_causal_conv1d_update 的
# null_block_id 合法值（必须非负或 None），kernel 跳过索引为 0 的行。
PAD_SLOT_ID = 0


@dataclass
class CausalConv1dMetadata:
    """Metadata compatible with the legacy Triton causal-convolution API.

    AscendC does not need the Triton launch metadata, so all three fields are
    empty for an NPU request.  The object is still constructed on every call
    because Qwen3's forward expects it to exist; values are filled by the
    upstream ``prepare_causal_conv1d_metadata`` helper and never inspected on
    Ascend.
    """

    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor
    total: int


def _load_fla_npu_update():
    from fla_npu.ops.ascendc import npu_causal_conv1d_update

    return npu_causal_conv1d_update


def _load_fla_npu_fn():
    from fla_npu.ops.ascendc import npu_causal_conv1d_fn

    return npu_causal_conv1d_fn


def _normalize_activation_for_fla(activation: Union[bool, str, None]) -> Optional[str]:
    """Convert the RTP-LLM activation contract to the FLA-NPU string form.

    FLA-NPU ctypes path understands ``None``, ``"silu"`` and ``"swish"``; RTP-LLM
    additionally accepts the boolean ``True``/``False`` aliases.
    """
    if activation is None or activation is False:
        return None
    if activation is True or activation == "silu" or activation == "swish":
        return "silu" if activation is True or activation == "silu" else "swish"
    raise ValueError(
        f"activation must be None, False, True, 'silu', or 'swish', got {activation!r}"
    )


# ---------------------------------------------------------------------------
# Helpers that have been fully device-ified.  They only read static metadata
# (``.shape``, ``.device``, ``.numel()``) from input tensors — no ``.cpu()``,
# no ``.tolist()``, no Python-level iteration over device content — so they
# are safe inside a graph-capture region.
# ---------------------------------------------------------------------------


def _gather_pages_from_block_map(
    block_map: torch.Tensor,
    block_indices: torch.Tensor,
    pad_slot_id: int,
) -> torch.Tensor:
    """Device-side equivalent of ``_mapped_page`` per sequence.

    ``block_map`` shape: ``(batch, max_blocks)`` — physical page table.
    ``block_indices`` shape: ``(batch,)`` — logical block index per sequence.
    Returns shape ``(batch,)`` int32 device tensor of physical page IDs.
    Entries where ``block_index >= block_map.shape[1]`` are masked to
    ``pad_slot_id`` (sentinel for "no such page") so callers can rely on a
    uniform sentinel.
    """

    # block_map is typically (batch, max_kernel_blocks); if it carries a
    # group prefix we strip that here to keep downstream simple.
    if block_map.dim() == 3:
        # Take group 0; kernel paging is identical across TP groups.
        block_map = block_map[0]
    elif block_map.dim() != 2:
        raise ValueError(
            f"block_map must be 2-D or 3-D, got shape {tuple(block_map.shape)}"
        )

    block_indices = block_indices.to(block_map.device)
    # Clamp indices into the valid column range so gather never goes out of
    # bounds (negative included — the where() below masks those rows to
    # pad_slot_id afterwards).
    max_col = block_map.shape[1]
    in_range = block_indices < max_col
    safe_indices = block_indices.clamp(min=0, max=max_col - 1)
    pages = block_map.gather(1, safe_indices.unsqueeze(1)).squeeze(1)
    # Also clamp negative indices (before any block has been allocated).
    pages = torch.where(in_range & (block_indices >= 0), pages, torch.full_like(pages, pad_slot_id))
    return pages.to(torch.int32)


def _cross_page_copy_device(
    conv_state: torch.Tensor,
    write_page: torch.Tensor,
    read_page: torch.Tensor,
    pad_slot_id: int,
) -> None:
    """Device-side cross-block conv_state migration (graph-capture safe).

    Before writing to a freshly-allocated page (e.g. when crossing a cache
    block boundary), the conv_state at the previous page must be copied in so
    the convolution sees a contiguous history window.  This is the device
    equivalent of the per-sequence ``conv_state[target].copy_(conv_state[source])``
    that the host loop performed — but runs entirely on the NPU.

    ``conv_state``: paged state; only the page axis (dim 0) is indexed, so
    both the RTP ``(pages, dim, state)`` and FLA ``(pages, state, dim)``
    layouts work.
    ``write_page`` / ``read_page`` shape: ``(batch,)`` int32 device tensors.

    Static-shape by design: boolean-mask indexing (``xxx[needs_copy]``) is
    deliberately NOT used — it lowers to ``nonzero()`` whose output shape
    depends on the mask *values*, which either fails ACL graph capture or
    freezes the capture-time shape (the set of rows needing migration changes
    on every replay).  Instead, rows that need no migration become self-copies
    (write -> write, a no-op) and sentinel rows (-1) clamp onto the reserved
    page 0, so the index_copy_/index_select shapes are always ``(batch,)``.
    迁移由 state_migration.migrate_state_rows 的 triton precopy kernel 执行
    （只拷真正跨界的行）；CPU / 无 triton / RTP_LLM_GDN_PRECOPY=0 时回退到
    等价的无条件全量 index_copy_。
    """

    needs_copy = (
        (read_page != pad_slot_id)
        & (write_page != pad_slot_id)
        & (read_page != write_page)
    )
    # 无需迁移的行做成 src == dst：migrate_state_rows 的 triton kernel 只拷
    # 真正跨界的行（回退路径为无条件全量 index_copy_，语义等价）。
    src = torch.where(needs_copy, read_page, write_page)
    migrate_state_rows(conv_state, src, write_page)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def _gather_prefill_states_device(
    x: torch.Tensor,
    conv_states: Optional[torch.Tensor],
    block_map: Optional[torch.Tensor],
    prefix_lengths: torch.Tensor,
    seq_size_per_block: int,
    state_len: int,
    pad_slot_id: int,
) -> torch.Tensor:
    """Device-side equivalent of the old ``_gather_prefill_states`` host loop.

    Reads the last ``state_len = width - 1`` values from the cache page that
    each sequence's prefix ends at, producing the flat ``(batch, state_len,
    dim)`` initial-state buffer the FLA-NPU prefill op requires.  Empty cache,
    missing block_map, or ``state_len == 0`` (width == 1, no state) short-circuit
    to a zero buffer without touching device memory.
    """

    batch = prefix_lengths.shape[0]
    dim = x.shape[0]  # x: (dim, total_tokens)
    initial_states = torch.zeros(
        (batch, state_len, dim), dtype=x.dtype, device=x.device
    )
    if conv_states is None or block_map is None or state_len == 0:
        return initial_states

    prefix_positive = prefix_lengths > 0
    # Prefill runs in eager only (the ACL graph runner is decode-only), so the
    # boolean-mask indexing used below — whose output shape depends on the
    # mask values and is therefore NOT graph-capture safe — is acceptable
    # here.  Do not copy this pattern into the decode path.

    block_indices = (prefix_lengths - 1).clamp(min=0) // seq_size_per_block
    page_indices = _gather_pages_from_block_map(block_map, block_indices, pad_slot_id)

    read_mask = prefix_positive & (page_indices != pad_slot_id)
    # Masked indexing: eager-only, see the note above.
    valid_pages = page_indices[read_mask].long()
    # conv_states: (pages, dim, state) in RTP layout.  FLA wants (batch, state, dim).
    gathered = conv_states.index_select(0, valid_pages).transpose(1, 2)
    initial_states[read_mask] = gathered

    return initial_states


def prepare_causal_conv1d_metadata(
    query_start_loc: torch.Tensor,
    device: torch.device,
) -> CausalConv1dMetadata:
    """Return the no-op launch metadata expected by the shared model."""

    empty = torch.empty(0, dtype=torch.int32, device=device)
    return CausalConv1dMetadata(empty, empty, 0)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    block_map: Optional[torch.Tensor],
    prefix_lengths: torch.Tensor,
    seq_size_per_block: int,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata: Optional[CausalConv1dMetadata] = None,
    validate_data=False,
):
    """Run varlen causal convolution and update the paged cache in place.

    Metadata (``query_start_loc``, ``has_initial_state``) is passed to
    ``fla_npu.npu_causal_conv1d_fn`` as device int32 tensors, and the initial
    states are gathered device-side.

    **Eager-only.**  The multi-block snapshot write-back
    (``_scatter_prefill_states_host``) still uses ``.tolist()`` and Python
    loops by design — prefill is never graph-captured (the ACL graph runner
    is decode-only), so this is acceptable.  Do not call this function from
    a capture region.
    """

    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("NPU prefill expects x=(dim, tokens), weight=(dim, width)")
    if seq_size_per_block <= 0:
        raise ValueError("seq_size_per_block must be positive")

    original_dtype = x.dtype
    x_work = x.to(weight.dtype)
    dim, _ = x_work.shape
    weight_dim, width = weight.shape
    if dim != weight_dim:
        raise ValueError("x and weight feature dimensions must match")

    # Static metadata — no data read, shape only.
    batch = query_start_loc.shape[0] - 1
    if prefix_lengths.shape[0] != batch:
        raise ValueError(
            f"prefix_lengths must contain one value per sequence, got "
            f"{prefix_lengths.shape[0]} vs expected batch={batch}"
        )

    # Device-side initial-state preparation.
    state_len = width - 1
    initial_states = _gather_prefill_states_device(
        x_work,
        conv_states,
        block_map,
        prefix_lengths,
        seq_size_per_block,
        state_len,
        pad_slot_id,
    )

    # has_initial_state: bool per sequence.  FLA-NPU accepts bool int32 or bool tensor.
    has_initial_state = (prefix_lengths > 0).to(torch.int32)

    # FLA-NPU layout: conv_states = (batch, state_len, dim); weight = (width, dim);
    # x = (tokens, dim).  Transpose RTP layout on the fly.
    # 注意：算子会把"序列末状态"原地写回传入的 conv_states 缓冲（与旧 pybind
    # 路径行为一致；旧实现 HEAD causal_conv1d_fn 里正是用
    # `initial_states = temporary_states.clone()` 规避）。_scatter_prefill_states_host
    # 重建早期边界（local_end < state_len，需从 prefill 起点状态补历史）快照时
    # 读的是同一 buffer——必须给算子传副本，原始 buffer 留给 scatter。
    npu_states = initial_states.clone()
    npu_weight = weight.transpose(0, 1).contiguous()
    npu_x = x_work.transpose(0, 1).contiguous()

    npu_causal_conv1d_fn = _load_fla_npu_fn()
    output = npu_causal_conv1d_fn(
        x=npu_x,
        weight=npu_weight,
        bias=bias,
        conv_states=npu_states,
        query_start_loc=query_start_loc.to(torch.int32),
        has_initial_state=has_initial_state,
        activation=_normalize_activation_for_fla(activation),
        pad_slot_id=pad_slot_id,
        validate_data=False,  # Always false to stay off the D2H path.
    )

    # FLA-NPU writes updated state back into npu_states.  Sync to the paged
    # conv_states cache.  For long prefill sequences that cross multiple
    # cache-block edges we must write a snapshot at *each* edge, not just the
    # final position.  This requires reconstructing the last ``state_len``
    # input values at every crossed edge, which is most naturally expressed
    # as a host-side loop over tokens — acceptable because prefill is not
    # graph-captured today.  Device-side vectorization is tracked as a
    # follow-up optimization.
    if conv_states is not None and block_map is not None and state_len > 0:
        _scatter_prefill_states_host(
            x=x_work,
            conv_states=conv_states,
            block_map=block_map,
            query_start_loc=query_start_loc,
            prefix_lengths=prefix_lengths,
            seq_size_per_block=seq_size_per_block,
            initial_states=initial_states,
            state_len=state_len,
            pad_slot_id=pad_slot_id,
        )

    return output.transpose(0, 1).to(original_dtype)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """Decode one or more tokens while preserving RTP-LLM's paged state.

    Fully device-based: index computation, cross-block conv_state migration,
    and operator invocation all happen on NPU tensors.  The only Python-level
    loop iterates over the static ``token_count`` (from the tensor shape) and
    is safe inside Ascend stream-capture.
    """

    if seq_size_per_block <= 0:
        raise ValueError("seq_size_per_block must be positive")
    if block_map is None or sequence_lengths is None:
        raise ValueError("block_map and sequence_lengths are required on NPU")
    if cache_seqlens is not None or query_start_loc is not None:
        raise NotImplementedError(
            "Ascend paged decode does not support cache_seqlens or varlen "
            "query_start_loc"
        )

    original_dtype = x.dtype
    squeeze_token_axis = x.dim() == 2
    if squeeze_token_axis:
        x = x.unsqueeze(-1)
    if x.dim() != 3:
        raise ValueError("NPU decode expects x=(batch, dim, tokens)")

    x_work = x.to(conv_state.dtype)
    batch, dim, token_count = x_work.shape
    if weight.dim() != 2 or weight.shape[0] != dim:
        raise ValueError("weight must have shape (dim, width)")

    # ---- Static metadata validation (no data read, safe inside capture) ----
    if sequence_lengths.shape[0] != batch:
        raise ValueError(
            f"sequence_lengths must contain one value per sequence, got "
            f"{sequence_lengths.shape[0]} vs expected batch={batch}"
        )

    # ---- Device-side index computation ----
    total_len = sequence_lengths.to(torch.int64)  # (batch,)
    width = weight.shape[1]
    state_len = width - 1

    # Logical block index the current sequence length lands in (will be written to).
    write_block_start = (total_len - 1).clamp(min=0) // seq_size_per_block
    # Logical block index holding the previous state (read source before first token).
    read_block = (total_len - 2).clamp(min=0) // seq_size_per_block

    # Physical page IDs for the initial state copy.
    read_page = _gather_pages_from_block_map(block_map, read_block, pad_slot_id)
    write_page_initial = _gather_pages_from_block_map(
        block_map, write_block_start, pad_slot_id
    )

    npu_causal_conv1d_update = _load_fla_npu_update()
    npu_weight = weight.transpose(0, 1).contiguous()  # (width, dim), FLA layout
    # RTP conv_state: (num_pages, dim, state).  FLA expects: (num_pages, state, dim).
    # Use a non-contiguous view so that the kernel's in-place mutations propagate
    # back to the original conv_state.  DO NOT add .contiguous() here — that
    # would create a fresh buffer and leave conv_state stale after replay.
    npu_states = conv_state.transpose(1, 2)

    # Pre-allocate output buffer with static address for graph replay safety.
    # We allocate (batch, token_count, dim) in FLA layout; the op writes into
    # slices of this buffer per-token.  This avoids the op accidentally
    # clobbering x via its in-place fallback path.
    out_buffer = torch.empty(
        (batch, token_count, dim), dtype=x_work.dtype, device=x_work.device
    )

    npu_activation = _normalize_activation_for_fla(activation)

    for token_index in range(token_count):
        # Per-token target block index (spec decode snapshots each token into a
        # consecutive block-map entry, even if it has not crossed a boundary).
        target_block = write_block_start + token_index
        target_page = _gather_pages_from_block_map(
            block_map, target_block, pad_slot_id
        )

        # Cross-page state migration for this token's target page.
        source_page = (
            read_page
            if token_index == 0
            else _gather_pages_from_block_map(
                block_map, target_block - 1, pad_slot_id
            )
        )
        _cross_page_copy_device(
            npu_states, target_page, source_page, pad_slot_id
        )

        # Slice per-token input and output buffers.  x_work is (batch, dim, token_count)
        # in RTP layout → transpose to (batch, 1, dim) for FLA.
        x_fla = x_work[:, :, token_index : token_index + 1].transpose(1, 2).contiguous()
        out_slice = out_buffer[:, token_index : token_index + 1, :]

        # FLA-NPU update API 契约：null_block_id 必须非负或 None（None=禁用
        # 跳过），kernel 跳过 conv_state_indices == null_block_id 的行；该
        # API 的 launch 级 pad_slot_id 被硬编码为不可命中的值，null_block_id
        # 是唯一的跳过哨兵。PAD_SLOT_ID=0（保留页，= fla_npu NULL_BLOCK_ID；
        # 引擎 BlockPool 保留块 0、真实块 ID 从 1 开始，零不会与真实页冲突），
        # 越界位置由 _gather_pages_from_block_map 掩码为 0。clamp 兜底引擎
        # NULL_BLOCK_IDX(-1) 值从表内泄漏，避免负索引裸进 kernel。
        npu_causal_conv1d_update(
            x=x_fla,
            conv_state=npu_states,
            weight=npu_weight,
            bias=bias,
            activation=npu_activation,
            conv_state_indices=target_page.clamp(min=0),
            out=out_slice,
            null_block_id=pad_slot_id,
            validate_data=False,
        )

    # npu_causal_conv1d_update writes each token's result into its ``out``
    # slice (out.copy_(result) inside the op), so out_buffer already holds
    # the full (batch, token_count, dim) result — one transpose+contiguous
    # to RTP (batch, dim, token_count), no cat needed.
    output = out_buffer.transpose(1, 2).contiguous()
    if squeeze_token_axis:
        output = output.squeeze(-1)
    return output.to(original_dtype)


# ---------------------------------------------------------------------------
# Prefill write-back helpers (host-side, kept for correctness of multi-block
# cache snapshots).  These helpers intentionally touch data with .tolist()
# and Python loops; they must *not* be called from graph-capture regions.
# ---------------------------------------------------------------------------


def _history_ending_at(
    initial_state: torch.Tensor,
    sequence_x: torch.Tensor,
    end: int,
) -> torch.Tensor:
    """Return the fixed-width input history ending before ``end``.

    ``initial_state``: (state_len, dim) — cache state before this prefill began.
    ``sequence_x``:    (seq_len, dim) — input tokens of this prefill segment.
    ``end``:            local index (0..seq_len).
    Returns (state_len, dim) — the last ``state_len`` input values before ``end``,
    padding with ``initial_state`` content when ``end < state_len``.
    """

    state_len = initial_state.shape[0]
    if state_len == 0:
        return initial_state
    if end >= state_len:
        return sequence_x[end - state_len : end]
    return torch.cat((initial_state[end:], sequence_x[:end]), dim=0)


def _scatter_prefill_states_host(
    x: torch.Tensor,
    conv_states: Optional[torch.Tensor],
    block_map: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    prefix_lengths: torch.Tensor,
    seq_size_per_block: int,
    initial_states: torch.Tensor,
    state_len: int,
    pad_slot_id: int,
) -> None:
    """Host-side multi-block cache snapshot for prefill.

    Iterates each sequence, finds every crossed ``seq_size_per_block`` boundary,
    reconstructs the last ``state_len`` input tokens at that boundary, and
    writes the snapshot to the corresponding physical page in ``conv_states``.

    **Intentionally not graph-safe.**  Only call from eager-prefill; for
    graph-captured prefill a fully vectorized device path must replace this.
    """

    if conv_states is None or block_map is None or state_len == 0:
        return

    # Single .tolist() to materialize the loop-bound indices — happens once at
    # the start of prefill, before any NPU work.
    query_starts = [int(v) for v in query_start_loc.detach().cpu().tolist()]
    prefix_values = [int(v) for v in prefix_lengths.detach().cpu().tolist()]
    # Normalize block_map to (batch, max_blocks) for indexing.
    block_map_2d = block_map[0] if block_map.dim() == 3 else block_map
    block_rows = [
        [int(p) for p in row.detach().cpu().tolist()]
        for row in block_map_2d
    ]

    for sequence_index, prefix_length in enumerate(prefix_values):
        token_start = query_starts[sequence_index]
        token_end = query_starts[sequence_index + 1]
        sequence_x = x[:, token_start:token_end].transpose(0, 1)
        sequence_len = token_end - token_start

        for local_end in range(1, sequence_len + 1):
            absolute_end = prefix_length + local_end
            if absolute_end % seq_size_per_block != 0 and local_end != sequence_len:
                continue

            block_index = (absolute_end - 1) // seq_size_per_block
            if (
                block_index >= len(block_rows[sequence_index])
                or block_index < 0
            ):
                continue
            page_index = block_rows[sequence_index][block_index]
            if page_index == pad_slot_id:
                continue

            history = _history_ending_at(
                initial_states[sequence_index], sequence_x, local_end
            )
            conv_states[page_index, :, :state_len].copy_(history.transpose(0, 1))
