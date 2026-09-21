import torch
import torch_npu

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_attn_params import (
    AscendAttnParams,
    compute_ascend_attn_params,
    infer_blocks_per_phys,
    split_kv_kernel_blocks,
)
from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_kv_cache_write_op import AscendKVCacheWriteOp
from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope_emb import AscendRotaryEmbeddingOp
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.attention import common


class AscendDecodeImpl(FMHAImplBase):
    """Ascend MHA Decode using FIA v2.

    Eager: FIA v2 + split_kv_kernel_blocks (contiguous copy).
    Graph: FIA v2 + graph_task_group + graph_task_update (vllm-ascend pattern).
    """

    def __init__(self, attn_configs, attn_inputs, parallelism_config):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = None

        self.fmha_impl = AscendDecodeAttnOp(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs)
        self.kv_cache_write_op = AscendKVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_inputs.kv_cache.seq_size_per_block if attn_inputs.kv_cache else 128,
        )

        self.params = AscendAttnParams()
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.params)
        self.kv_cache_write_op.set_params(self.params)

        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def _create_rope_impl(self, attn_configs):
        from rtp_llm.ops import RopeStyle
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return AscendRotaryEmbeddingOp(attn_configs)

    def _split_qkv(self, qkv):
        qkv = qkv.reshape(qkv.shape[0], -1)
        num_heads = self.attn_configs.head_num
        num_kv_heads = self.attn_configs.kv_head_num
        head_dim = self.attn_configs.size_per_head
        q, k, v = torch.split(qkv, [
            head_dim * num_heads,
            head_dim * num_kv_heads,
            head_dim * num_kv_heads,
        ], dim=-1)
        query = q.reshape(q.shape[0], num_heads, head_dim)
        key = k.reshape(k.shape[0], num_kv_heads, head_dim).contiguous()
        value = v.reshape(v.shape[0], num_kv_heads, head_dim).contiguous()
        return query, key, value

    def _update_rope_kv_write_params(self, device, kv_cache, layer_idx: int = 0):
        if getattr(self.attn_inputs, "is_cuda_graph", False):
            self._update_rope_kv_write_params_device(device)
            return
        # The kernel block granularity comes from the per-layer cache view; the
        # physical block size it maps onto drives slot_mapping and the writes.
        blocks_per_phys = infer_blocks_per_phys(self.attn_inputs)
        kernel_page = kv_cache.seq_size_per_block if kv_cache is not None else 0
        self.params.blocks_per_phys = blocks_per_phys
        positions, slot_mapping = compute_ascend_attn_params(
            self.attn_inputs, layer_idx, kernel_page * blocks_per_phys
        )
        self.params.positions_d = positions.to(device, non_blocking=True)
        self.params.slot_mapping = slot_mapping.to(device, non_blocking=True)

    def _update_rope_kv_write_params_device(self, device):
        seq_lens_plus_1 = self.attn_inputs.sequence_lengths_plus_1_d
        positions_d = seq_lens_plus_1 - 1

        block_table = self.attn_inputs.kv_cache_kernel_block_id_device
        page_size = (self.attn_inputs.kv_cache.seq_size_per_block
                     if self.attn_inputs.kv_cache is not None else 128)
        if block_table is not None and block_table.numel() > 0 and positions_d.numel() > 0:
            if block_table.ndim != 2:
                block_table = block_table.reshape(-1, block_table.shape[-1])
            max_blocks = block_table.size(1)
            pos_long = positions_d.long()
            block_index = (pos_long // page_size).clamp(max=max_blocks - 1)
            block_offset = pos_long % page_size
            slot_block_numbers = torch.gather(
                block_table, 1,
                block_index.unsqueeze(1).to(block_table.dtype)
            ).squeeze(1).long().clamp(min=0)
            slot_mapping = (slot_block_numbers * page_size + block_offset).to(torch.int64)
        else:
            slot_mapping = torch.empty(0, dtype=torch.int64, device=device)

        self.params.positions_d = positions_d
        self.params.slot_mapping = slot_mapping

    def prepare(self, attn_inputs):
        self.fmha_impl.prepare(attn_inputs)
        self.attn_inputs = attn_inputs

    def prepare_cuda_graph(self, attn_inputs):
        """Called by AscendGraphRunner::prepareInputs() before each replay."""
        self.attn_inputs = attn_inputs
        self.fmha_impl.prepare(attn_inputs)
        batch_size = attn_inputs.sequence_lengths.size(0)
        seq_lens = attn_inputs.sequence_lengths[:batch_size]
        ctx_list = (seq_lens.to(torch.int32) + 1).tolist()
        self.fmha_impl.update_graph_fia(ctx_list, batch_size)

    def forward(self, qkv, kv_cache, layer_idx=0):
        is_graph = getattr(self.attn_inputs, "is_cuda_graph", False)

        if self.need_rope_kv_cache:
            self._update_rope_kv_write_params(qkv.device, kv_cache, layer_idx)
            if self.rope_impl is not None:
                query, key, value = self.rope_impl.forward(qkv)
            else:
                query, key, value = self._split_qkv(qkv)
            self.kv_cache_write_op.forward(key, value, kv_cache)
            q = query
        else:
            q = qkv.chunk(3, dim=-1)[0]

        if is_graph:
            self.fmha_impl.context_lens = self.attn_inputs.sequence_lengths_plus_1_d
        else:
            self.fmha_impl.context_lens = self.attn_inputs.sequence_lengths + 1

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(q, kv_cache, is_graph)

    @staticmethod
    def support(attn_configs, attn_inputs):
        return not attn_inputs.is_prefill and \
               not attn_configs.use_mla and \
               torch.npu.is_available()


class AscendDecodeAttnOp:
    """NPU decode attention: FIA v2 (eager) or FIA v2 + graph_task_group (graph).

    Graph mode uses the vllm-ascend pattern:
    - Capture: graph_task_group_begin/end wraps FIA v2 .out() with pre-computed workspace
    - Replay:  graph_task_update_begin/end updates context_lens dynamically
    """

    _causal_mask = None
    _shared_workspace = None
    _shared_update_stream = None

    @classmethod
    def _get_causal_mask(cls, device):
        if cls._causal_mask is None or cls._causal_mask.device.type != device.type:
            cls._causal_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.int8), diagonal=1
            ).to(device)
        return cls._causal_mask

    def __init__(self, attn_configs, attn_inputs):
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.q_scaling * self.head_dim ** -0.5
        self.page_size = attn_inputs.kv_cache.seq_size_per_block if \
                         attn_inputs.kv_cache else 128
        self.block_table = None
        self.context_lens = None
        self.blocks_per_phys = 1

        self._graph_handles = []
        self._graph_refs = []

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        if getattr(attn_inputs, "is_cuda_graph", False):
            self.block_table = attn_inputs.kv_cache_kernel_block_id_device
            if self.block_table is not None:
                if self.block_table.ndim != 2:
                    self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])
            if attn_inputs.sequence_lengths_plus_1_d.numel() > 0:
                self.context_lens = attn_inputs.sequence_lengths_plus_1_d
            else:
                self.context_lens = None
            return
        # Eager path
        self.block_table = attn_inputs.kv_cache_kernel_block_id_host
        self.blocks_per_phys = infer_blocks_per_phys(attn_inputs)
        if self.block_table is not None:
            self.block_table = self.block_table.clamp(min=0)
            if self.block_table.ndim != 2:
                self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])
        if attn_inputs.sequence_lengths.numel() > 0:
            self.context_lens = attn_inputs.sequence_lengths + 1
        elif attn_inputs.prefix_lengths.numel() > 0 and attn_inputs.input_lengths.numel() > 0:
            self.context_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths
        else:
            self.context_lens = None

    def forward(self, q, kv_cache, use_graph=False):
        block_table = self.block_table
        if block_table is not None and block_table.device.type != q.device.type:
            block_table = block_table.to(q.device)
        context_lens = self.context_lens
        if context_lens is not None and context_lens.device.type != q.device.type:
            context_lens = context_lens.to(q.device)
        if use_graph and torch.npu.is_current_stream_capturing():
            return self._forward_fia_graph(q, kv_cache, block_table, context_lens)
        return self._forward_fia(q, kv_cache, block_table, context_lens)

    def _forward_fia_graph(self, q, kv_cache, block_table, context_lens):
        kv_base = kv_cache.kv_cache_base
        k_cache = kv_base[:, 0].reshape(kv_base.shape[0], self.page_size, -1)
        v_cache = kv_base[:, 1].reshape(kv_base.shape[0], self.page_size, -1)
        batch_size = q.shape[0]
        out = torch.empty(batch_size, self.num_heads, self.head_dim,
                          dtype=q.dtype, device=q.device)
        lse = torch.empty(1, dtype=q.dtype, device=q.device)
        atten_mask = self._get_causal_mask(q.device)

        if AscendDecodeAttnOp._shared_workspace is None:
            AscendDecodeAttnOp._shared_workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=q, key=k_cache, value=v_cache,
                atten_mask=atten_mask, block_table=block_table,
                input_layout="TND", block_size=self.page_size,
                actual_seq_qlen=list(range(1, batch_size + 1)),
                actual_seq_kvlen=[self.page_size] * batch_size,
                num_key_value_heads=self.num_kv_heads,
                num_query_heads=self.num_heads,
                softmax_scale=self.scale, sparse_mode=3)
            ws_bytes = AscendDecodeAttnOp._shared_workspace.numel() * AscendDecodeAttnOp._shared_workspace.element_size()
            AscendDecodeAttnOp._shared_workspace = torch.empty(ws_bytes // q.element_size(),
                                                dtype=q.dtype, device=q.device)

        if not torch.npu.is_current_stream_capturing():
            actual_seq_q = torch.arange(1, batch_size + 1, dtype=torch.int32, device=q.device)
            actual_seq_kv = context_lens.to(torch.int32)
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query=q, key=k_cache, value=v_cache,
                atten_mask=atten_mask, block_table=block_table,
                input_layout="TND", block_size=self.page_size,
                actual_seq_qlen=actual_seq_q, actual_seq_kvlen=actual_seq_kv,
                num_key_value_heads=self.num_kv_heads,
                num_query_heads=self.num_heads,
                softmax_scale=self.scale, sparse_mode=3)
            return attn_output

        actual_seq_q = list(range(1, batch_size + 1))
        actual_seq_kv = [self.page_size] * batch_size
        stream = torch.npu.current_stream()
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=q, key=k_cache, value=v_cache,
            atten_mask=atten_mask, block_table=block_table,
            input_layout="TND", block_size=self.page_size,
            actual_seq_qlen=actual_seq_q, actual_seq_kvlen=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            sparse_mode=3,
            softmax_scale=self.scale,
            workspace=AscendDecodeAttnOp._shared_workspace,
            out=[out, lse])
        handle = torch.npu.graph_task_group_end(stream)

        self._graph_handles.append(handle)
        self._graph_refs.append((
            q, k_cache, v_cache,
            block_table, atten_mask,
            out, lse))
        return out

    def update_graph_fia(self, ctx_list, batch_size):
        if not self._graph_handles:
            return
        if AscendDecodeAttnOp._shared_update_stream is None:
            AscendDecodeAttnOp._shared_update_stream = torch.npu.Stream()
        us = AscendDecodeAttnOp._shared_update_stream
        actual_seq_q = list(range(1, batch_size + 1))
        with torch.npu.stream(us):
            for i, handle in enumerate(self._graph_handles):
                q, k_cache, v_cache, block_table, atten_mask, out, lse = self._graph_refs[i]
                if q is None or k_cache is None or out is None:
                    continue
                torch.npu.graph_task_update_begin(us, handle)
                torch_npu.npu_fused_infer_attention_score_v2.out(
                    query=q, key=k_cache, value=v_cache,
                    atten_mask=atten_mask, block_table=block_table,
                    input_layout="TND", block_size=self.page_size,
                    actual_seq_qlen=actual_seq_q, actual_seq_kvlen=ctx_list,
                    num_key_value_heads=self.num_kv_heads,
                    num_query_heads=self.num_heads,
                    sparse_mode=3,
                    softmax_scale=self.scale,
                    workspace=AscendDecodeAttnOp._shared_workspace,
                    out=[out, lse])
                torch.npu.graph_task_update_end(us)
        us.synchronize()

    def _forward_fia(self, q, kv_cache, block_table, context_lens):
        # Eager path: split K/V at kernel-block granularity
        k_cache, v_cache, page_size = split_kv_kernel_blocks(
            kv_cache, self.blocks_per_phys)
        batch_size = q.shape[0]
        actual_seq_q = torch.arange(
            1, batch_size + 1, dtype=torch.int32, device=q.device
        )
        actual_seq_kv = context_lens.to(torch.int32)
        if actual_seq_kv.device.type != q.device.type:
            actual_seq_kv = actual_seq_kv.to(q.device)
        atten_mask = self._get_causal_mask(q.device)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=q, key=k_cache, value=v_cache,
            atten_mask=atten_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=page_size,
            actual_seq_qlen=actual_seq_q,
            actual_seq_kvlen=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            softmax_scale=self.scale,
            sparse_mode=3,
        )
        return attn_output
