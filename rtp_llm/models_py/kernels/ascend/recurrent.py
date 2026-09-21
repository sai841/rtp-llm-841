"""Ascend implementation of Qwen3.5 recurrent Gated-DeltaNet decode.

All index computation and state migration happens on device tensors — no
``.cpu()``, no ``.tolist()``, no Python-level loops over device content — so
that the decode path can participate in Ascend stream-capture (graph capture).

The underlying FLA-NPU operator (``npu_recurrent_gated_delta_rule`` on the
``_aclnn_ctypes`` path) already accepts device int32 tensors for
``actual_seq_lengths`` and ``ssm_state_indices``, so the wrapper only needs
to compute those indices device-side and pass them through.
"""

from __future__ import annotations

from typing import Optional

import torch

from rtp_llm.models_py.kernels.ascend.linear_attention import l2norm_fwd
from rtp_llm.models_py.kernels.ascend.state_migration import migrate_state_rows


def _get_ascendc_ops():
    try:
        from fla_npu.ops import ascendc
    except ImportError as exc:  # pragma: no cover - depends on the NPU image
        raise RuntimeError(
            "Qwen3.5 recurrent decode on Ascend requires the SoC-specific "
            "flash-linear-attention-npu wheel."
        ) from exc
    if not hasattr(ascendc, "npu_recurrent_gated_delta_rule"):
        raise RuntimeError(
            "The installed FLA-NPU wheel does not export "
            "npu_recurrent_gated_delta_rule; install a flash-linear-attention-npu "
            "version that exports it."
        )
    return ascendc


# ---------------------------------------------------------------------------
# Device-side page resolution.  Mirrors the host logic of the original
# ``_resolve_state_pages`` but operates entirely on NPU tensors — no D2H, no
# Python loops over block_map entries.
# ---------------------------------------------------------------------------


def _resolve_state_pages_device(
    block_map: Optional[torch.Tensor],
    sequence_lengths: Optional[torch.Tensor],
    batch: int,
    token_count: int,
    seq_size_per_block: int,
    pad_slot_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-side equivalent of ``_resolve_state_pages``.

    Returns a pair of int32 device tensors on the same device as
    ``sequence_lengths`` (or ``block_map`` when ``sequence_lengths`` is None):

    * ``read_pages``  shape ``(batch,)``  — physical page ID each sequence
      reads its state from before the first token of this invocation.
    * ``write_pages`` shape ``(batch * token_count,)`` — flat list of
      physical page IDs each speculative token writes its state snapshot to,
      laid out as ``[b0t0, b0t1, ..., b0tT, b1t0, ..., bBtT]`` which is
      exactly the layout expected by ``ssm_state_indices`` of
      ``npu_recurrent_gated_delta_rule``.

    The ``sequence_lengths`` argument is RTP's ``sequence_lengths_plus_1_d``:
    its value is the total sequence length *after* the first token in this
    invocation.  RTP's continuous-batching contract reads the state before
    that token from ``(length - 2) // block_size`` and stores every
    speculative token in a consecutive block-map entry beginning at
    ``(length - 1) // block_size``.  The latter is intentionally *not*
    ordinary token-to-block placement.
    """

    if seq_size_per_block <= 0:
        raise ValueError("seq_size_per_block must be positive")

    if sequence_lengths is not None:
        device = sequence_lengths.device
    elif block_map is not None:
        device = block_map.device
    else:
        # Degenerate interface-contract branch: no paged metadata at all,
        # synthetic page indices are host-visible CPU tensors.
        device = torch.device("cpu")
    batch_idx = torch.arange(batch, device=device)

    if block_map is None:
        # Degenerate case: one "page" per sequence, speculative writes land
        # on consecutive synthetic slots.  (block_map is always supplied in
        # production decode; this branch exists mainly to match the old
        # interface contract.)
        read_pages = batch_idx.to(torch.int32)
        write_pages = (
            batch_idx[:, None] * token_count
            + torch.arange(token_count, device=device)[None, :]
        ).reshape(-1).to(torch.int32)
        return read_pages, write_pages

    # Normalize block_map shape: accept (group, batch, max_blocks) or (batch, max_blocks).
    # Group 0 is the LINEAR (GDN) cache group whose table is physical-block
    # granular (bpk = 1), matching the seq_size_per_block arithmetic below.
    if block_map.dim() == 3:
        block_map = block_map[0]
    if block_map.dim() != 2 or block_map.shape[0] != batch:
        raise ValueError(
            f"block_map must have shape [batch, max_blocks] or [group, batch, max_blocks], "
            f"got {tuple(block_map.shape)}"
        )

    # Clamp out-of-range block indices so gather doesn't raise; we mask them
    # to pad_slot_id afterwards.
    max_col = block_map.shape[1]

    if sequence_lengths is None:
        first_lengths = torch.ones(batch, dtype=torch.int64, device=device)
    else:
        # Read only static metadata for the shape check — no data is copied.
        if sequence_lengths.shape[0] != batch:
            raise ValueError(
                "sequence_lengths must contain one value per batch, got "
                f"{sequence_lengths.shape[0]} vs batch={batch}"
            )
        first_lengths = sequence_lengths.to(torch.int64)

    # --- Read page: block at (first_length - 2) // seq_size_per_block ---
    read_block_pos = (first_lengths - 2).clamp(min=0) // seq_size_per_block
    read_safe = read_block_pos.clamp(max=max_col - 1)
    read_pages = block_map[batch_idx, read_safe].to(torch.int32)
    # Mask out-of-range or negative block positions.
    read_pages = torch.where(
        (read_block_pos < max_col) & (first_lengths > 1),
        read_pages,
        torch.full_like(read_pages, pad_slot_id),
    )

    # --- Write pages: consecutive block positions starting at
    #     (first_length - 1) // seq_size_per_block, one per speculative token ---
    write_block_start = (first_lengths - 1).clamp(min=0) // seq_size_per_block
    token_offsets = torch.arange(token_count, device=device)  # (T,)
    # (batch, T) block positions.
    write_block_pos = write_block_start[:, None] + token_offsets[None, :]
    write_safe = write_block_pos.clamp(max=max_col - 1)
    # Gather → (batch, T) physical pages, then flatten to (batch * T,).
    write_pages = block_map[batch_idx[:, None], write_safe].to(torch.int32)
    write_pages = write_pages.reshape(-1)
    # Mask out-of-range positions to pad_slot_id.
    in_range = write_block_pos < max_col
    write_pages = torch.where(in_range.reshape(-1), write_pages, torch.full_like(write_pages, pad_slot_id))

    return read_pages, write_pages


def _seed_first_write_pages_device(
    state: torch.Tensor,
    read_pages: torch.Tensor,
    write_pages: torch.Tensor,
    pad_slot_id: int = -1,
) -> None:
    """Device-side cross-page state migration (graph-capture safe).

    Before writing to a freshly-allocated page that differs from the page that
    currently holds the recurrent state, we copy the state over so that the
    subsequent kernel launch sees a contiguous window.  This is the device
    equivalent of the per-batch ``state[destination].copy_(state[source])``
    that the original host loop performed — but runs entirely on the NPU.

    ``state``         shape ``(num_pages, ...)`` in whatever layout the kernel
                      consumes; only the page axis (dim 0) is indexed.
    ``read_pages``    shape ``(batch,)`` int32 device tensor.
    ``write_pages``   shape ``(batch * token_count,)`` int32 device tensor.
                      We only care about the *first* write page of each batch
                      (the one that must be seeded); later pages are populated
                      by the same kernel launch.

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

    batch = read_pages.shape[0]
    # First write page per batch is at indices 0, token_count, 2*token_count, ...
    token_count = write_pages.shape[0] // batch
    first_write_pages = write_pages.reshape(batch, token_count)[:, 0]

    needs_copy = (
        (read_pages != pad_slot_id)
        & (first_write_pages != pad_slot_id)
        & (read_pages != first_write_pages)
    )
    # 无需迁移的行做成 src == dst：migrate_state_rows 的 triton kernel 只拷
    # 真正跨界的行（回退路径为无条件全量 index_copy_，语义等价）。
    src = torch.where(needs_copy, read_pages, first_write_pages)
    migrate_state_rows(state, src, first_write_pages)


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block=1,
    sequence_lengths: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run varlen recurrent Gated-DeltaNet decode with paged state.

    All metadata (block indices, sequence lengths) flows through device int32
    tensors — no Python-side host conversion — so the function is safe inside
    an Ascend stream-capture region.

    The operator used is ``fla_npu.ops.ascendc.npu_recurrent_gated_delta_rule``
    on the ctypes path, which natively accepts device tensors for
    ``actual_seq_lengths`` and ``ssm_state_indices``.
    """

    if cu_seqlens is not None:
        raise NotImplementedError(
            "cu_seqlens is a prefill interface; Ascend recurrent decode uses "
            "block_map and sequence_lengths"
        )
    if initial_state is None:
        raise ValueError("initial_state is required for recurrent decode")
    if not inplace_final_state:
        raise NotImplementedError(
            "Ascend recurrent decode currently requires inplace_final_state=True"
        )
    if initial_state.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("FLA-NPU recurrent state must be bfloat16 or float32")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q/k/v must have shapes [B,T,H,D]")
    batch, token_count = q.shape[:2]
    if v.shape[:2] != (batch, token_count):
        raise ValueError("q/k/v must share batch and token dimensions")
    if beta is not None and beta.shape != v.shape[:-1]:
        raise ValueError("beta must have shape [B,T,HV]")
    if g.shape != v.shape[:-1]:
        raise ValueError("g must have shape [B,T,HV]")
    if initial_state.ndim != 4 or initial_state.shape[1:] != (
        v.shape[2],
        v.shape[3],
        q.shape[3],
    ):
        raise ValueError("initial_state must have shape [pages,HV,DV,DK]")
    # .stride() reads tensor metadata, not data — safe inside capture.
    if initial_state.stride(-1) != 1:
        raise ValueError("initial_state DK dimension must be contiguous")
    if token_count > 8:
        raise ValueError(
            "FLA-NPU recurrent decode supports at most 8 tokens per sequence"
        )
    if beta is None:
        beta = torch.ones_like(v[..., 0])
    if scale is None:
        scale = k.shape[-1] ** -0.5
    elif scale <= 0:
        raise ValueError("scale must be positive")
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    state = initial_state
    if token_count == 0:
        return v.new_empty(v.shape), state

    # ---- Device-side index resolution ----
    read_pages, write_pages = _resolve_state_pages_device(
        block_map,
        sequence_lengths,
        batch,
        token_count,
        int(seq_size_per_block),
    )

    ascendc = _get_ascendc_ops()

    # The operator keeps the recurrence in FP32 across all tokens and uses one
    # state index per token for the BF16 snapshots.  Seed only the first output
    # page from the previously committed page; later pages are written by the
    # same launch, avoiding a BF16 reload between speculative tokens.
    _seed_first_write_pages_device(state, read_pages, write_pages)

    # actual_seq_lengths: flat varlen layout.  First sequence has length 0
    # (placeholder consumed by FLA-NPU's varlen convention), followed by one
    # entry per batch giving the token count for that batch.  All device int32,
    # no host conversion.
    actual_seq_lengths = torch.empty(batch + 1, dtype=torch.int32, device=q.device)
    actual_seq_lengths[0] = 0
    actual_seq_lengths[1:] = token_count
    # npu_recurrent_gated_delta_rule has no null/pad skip parameter (unlike
    # npu_causal_conv1d_update's null_block_id), so -1 sentinel rows would be
    # consumed as raw indices.  Out-of-range write positions cannot occur in
    # production (the cache manager guarantees block-map coverage); clamp
    # defensively so degenerate rows land on the reserved page 0.
    ssm_state_indices = write_pages.clamp(min=0)

    result = ascendc.npu_recurrent_gated_delta_rule(
        q.reshape(-1, *q.shape[2:]).to(torch.bfloat16),
        k.reshape(-1, *k.shape[2:]).to(torch.bfloat16),
        v.reshape(-1, *v.shape[2:]).to(torch.bfloat16),
        state,
        beta=beta.reshape(-1, beta.shape[-1]).to(torch.bfloat16),
        scale=float(scale),
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        g=g.reshape(-1, g.shape[-1]).float(),
    )
    out = result[0] if isinstance(result, (tuple, list)) else result
    return out.reshape(batch, token_count, *out.shape[1:]).to(q.dtype), state


__all__ = ["fused_recurrent_gated_delta_rule"]
