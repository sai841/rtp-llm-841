// AscendGraphRunner implementation.
//
// All Ascend-specific device calls go through rtp_llm::ascend_graph::*
// helpers so this file stays free of #if USING_ASCEND where possible.

#include "rtp_llm/cpp/ascend_graph/ascend_graph_runner.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#if USING_ASCEND
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#endif

using namespace torch_ext;

namespace rtp_llm {

// ============================================================
// Construction / destruction
// ============================================================

AscendGraphRunner::AscendGraphRunner(const GraphParams& graph_params, py::object py_instance):
    GraphBase(std::move(py_instance)),
    enable_graph_(graph_params.enable_cuda_graph),
    enable_graph_debug_mode_(graph_params.enable_cuda_graph_debug_mode),
    is_target_verify_(graph_params.is_target_verify),
    num_tokens_per_bs_(graph_params.num_tokens_per_bs),
    max_seq_len_(graph_params.max_seq_len),
    seq_size_per_block_(graph_params.tokens_per_block),
    kernel_seq_size_per_block_(graph_params.kernel_tokens_per_block),
    hidden_size_(graph_params.hidden_size),
    sp_steps_(graph_params.sp_steps),
    model_data_type_(graph_params.model_data_type),
    decode_capture_batch_sizes_(graph_params.decode_capture_batch_sizes),
#if USING_ASCEND
    capture_stream_(ascend_graph::graphGetStreamFromPool(true)),
#endif
    forward_event_(ascend_graph::makeGraphEvent()),
    kv_cache_layer_to_group_(graph_params.kv_cache_layer_to_group),
    kv_cache_group_num_(graph_params.kv_cache_group_num) {
    if (kernel_seq_size_per_block_ <= 0) {
        throw std::runtime_error("AscendGraphRunner constructor: kernel_tokens_per_block must be > 0.");
    }
    // Prefill graph mode is not supported on Ascend ACL Graph today.
    if (graph_params.is_prefill_cuda_graph_mode) {
        throw std::runtime_error(
            "AscendGraphRunner: prefill cuda graph mode is not supported on Ascend ACL Graph.");
    }
    max_bs_ = graph_params.max_context_batch_size;

    py::gil_scoped_acquire gil;
    if (!py_instance_ || py_instance_.is_none()) {
        throw std::runtime_error("AscendGraphRunner constructor: Python instance is null or none.");
    }
    py_attn_pyobj_method_ = py_instance_.attr("prepare_fmha_impl");
    py_forward_method_    = py_instance_.attr("forward");

#if USING_ASCEND
    options_npu_int32_ =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kPrivateUse1).requires_grad(false);
    options_npu_float_ =
        torch::TensorOptions().dtype(model_data_type_).device(torch::kPrivateUse1).requires_grad(false);
#else
    // Non-Ascend platforms only compile this TU; they never instantiate these options.
    options_npu_int32_ = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
    options_npu_float_ = torch::TensorOptions().dtype(model_data_type_).device(torch::kCPU).requires_grad(false);
#endif
    options_cpu_int32_ = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);

    RTP_LLM_LOG_INFO(
        "Initialize AscendGraphRunner with parameters: enable_graph=%d, max_bs=%d, debug=%d, "
        "max_seq_len=%d, kernel_seq_size_per_block=%d, hidden_size=%d, num_tokens_per_bs=%d, "
        "is_target_verify=%d",
        enable_graph_,
        max_bs_,
        enable_graph_debug_mode_,
        max_seq_len_,
        kernel_seq_size_per_block_,
        hidden_size_,
        num_tokens_per_bs_,
        is_target_verify_);
}

AscendGraphRunner::~AscendGraphRunner() {
    RTP_LLM_LOG_INFO("Release AscendGraphRunner .....");
    py::gil_scoped_acquire gil;
    py_instance_.release();
    RTP_LLM_LOG_INFO("Release AscendGraphRunner Successfully");
}

// ============================================================
// Setters (no-op on Ascend if not configured)
// ============================================================

void AscendGraphRunner::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = position_encoding;
}

void AscendGraphRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = token_type_embedding;
}

void AscendGraphRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
}

// ============================================================
// Bucket selection (decode only)
// ============================================================

std::vector<int> AscendGraphRunner::getDecodeBatchSizesToCapture() {
    if (!decode_capture_batch_sizes_.empty()) {
        RTP_LLM_LOG_INFO("Using decode capture batch sizes from Python: %zu sizes",
                         decode_capture_batch_sizes_.size());
        std::sort(decode_capture_batch_sizes_.begin(), decode_capture_batch_sizes_.end());
        return decode_capture_batch_sizes_;
    }

    std::vector<int> capture_bs;
    int              max_generate_batch_size = max_bs_;
    RTP_LLM_LOG_INFO("max_generate_batch_size for ascend graph: %d", max_generate_batch_size);
    for (int i : {1, 8, 16, 24, 32}) {
        if (i <= max_generate_batch_size) {
            capture_bs.push_back(i);
        }
    }
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (capture_bs.back() != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

bool AscendGraphRunner::tryGetRealGraphDecodeBatchSize(const PyModelInputs& inputs, CudaGraphState& state) {
    int  cuda_graph_bs = inputs.attention_inputs.input_lengths.size(0);
    state.current_batch_size = cuda_graph_bs;
    RTP_LLM_LOG_DEBUG("AscendGraphRunner canRun judge for batch size: %d", cuda_graph_bs);
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("ascend graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state.current_batch_size);
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("ascend graph decode batch size %d exceeds max captured %d, fallback to normal run",
                            state.current_batch_size,
                            capture_range_.back());
        return false;
    }
    state.current_real_graph_bs = *it;

    if (inputs.attention_inputs.is_prefill) {
        state.seq_len_sum = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    } else {
        state.seq_len_sum = cuda_graph_bs;
    }
    RTP_LLM_LOG_DEBUG("ascend graph can run for decode, batch=%d graph_bs=%d",
                      state.current_batch_size,
                      state.current_real_graph_bs);
    return true;
}

bool AscendGraphRunner::canRun(const PyModelInputs& inputs, CudaGraphState& state) {
    RTP_LLM_PROFILE_SCOPE("ascend_graph.canRun");

    // Speculative decoding: target verify path
    if (is_target_verify_) {
        if (inputs.attention_inputs.is_target_verify) {
    return tryGetRealGraphDecodeBatchSize(inputs, state);
        }
        return false;
    }

    if (!enable_graph_) {
        return false;
    }
    // Decode only: any prefill goes to eager.
    if (inputs.attention_inputs.is_prefill) {
        return false;
    }

    // Hybrid KV cache group count check
    if (!inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()) {
        const size_t group = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        if (kv_cache_group_num_ <= 0) {
            RTP_LLM_LOG_WARNING("Hybrid kv cache detected but kv_cache_group_num_ is not set, fallback to normal run.");
            return false;
        }
        if (group != static_cast<size_t>(kv_cache_group_num_)) {
            RTP_LLM_LOG_WARNING("Hybrid kv cache group size mismatch: inputs=%zu, captured=%d, fallback to normal run.",
                                group,
                                kv_cache_group_num_);
            return false;
        }
    }

    return tryGetRealGraphDecodeBatchSize(inputs, state);
}

// ============================================================
// Capture-time tensor allocation (mirror CudaGraphRunner)
// ============================================================

void AscendGraphRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    inputs.attention_inputs.is_target_verify = is_target_verify_;
    // ACL Graph capture path is decode-only; is_prefill is forced false.
    inputs.attention_inputs.is_prefill = num_tokens_per_bs > 1;

    inputs.input_ids = torch::zeros({max_num_token_}, options_npu_int32_);
    // input_lengths [batch_size, int32] pinned host
    inputs.attention_inputs.input_lengths =
        torch::full({int(max_bs_)}, num_tokens_per_bs, options_cpu_int32_).pin_memory();
    inputs.attention_inputs.input_lengths_d = inputs.attention_inputs.input_lengths.to(options_npu_int32_.device());
    // sequence_lengths [batch_size, int32] pinned host
    // Use page_size as capture context_len: covers typical short prompts,
    // keeps ATB workspace small, avoids copy_stream sync with large ctx.
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.sequence_lengths.fill_(seq_size_per_block_ - 1);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();

    const int64_t max_kv_blocks =
        static_cast<int64_t>(((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_) + sp_steps_);
    const int64_t max_blocks = max_kv_blocks * seq_size_per_block_ / kernel_seq_size_per_block_;

    inputs.attention_inputs.kv_cache_kernel_block_id_device =
        torch::zeros({int(max_bs_), max_blocks}, options_npu_int32_);
    inputs.attention_inputs.kv_cache_kernel_block_id_host =
        torch::zeros({int(max_bs_), max_blocks}, options_cpu_int32_).pin_memory();
    // compute_ascend_attn_params() reads kv_cache_block_id_host (not the kernel
    // variant) to derive slot_mapping. Allocate it alongside the kernel version
    // so the capture warmup forward produces a non-empty slot_mapping instead
    // of crashing npu_scatter_pa_kv_cache with a size mismatch.
    inputs.attention_inputs.kv_cache_block_id_device =
        torch::zeros({int(max_bs_), max_blocks}, options_npu_int32_);
    inputs.attention_inputs.kv_cache_block_id_host =
        torch::zeros({int(max_bs_), max_blocks}, options_cpu_int32_).pin_memory();

    const auto layer_num = kv_cache_layer_to_group_.size();
    if (layer_num > 0) {
        auto kv_cache_layer_to_group_capture_ =
            torch::empty({static_cast<int64_t>(layer_num)}, options_cpu_int32_).pin_memory();
        auto* dst = kv_cache_layer_to_group_capture_.data_ptr<int32_t>();
        for (size_t i = 0; i < layer_num; ++i) {
            dst[i] = static_cast<int32_t>(kv_cache_layer_to_group_[i]);
        }
        inputs.attention_inputs.kv_cache_layer_to_group = kv_cache_layer_to_group_capture_;
    }

    // Hybrid cache: per-group block tables
    inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.clear();
    inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.clear();
    if (kv_cache_group_num_ > 1) {
        inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.reserve(kv_cache_group_num_);
        inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.reserve(kv_cache_group_num_);
        for (int g = 0; g < kv_cache_group_num_; ++g) {
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.push_back(
                torch::zeros({int(max_bs_), max_blocks}, options_npu_int32_));
            inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.push_back(
                torch::zeros({int(max_bs_), max_blocks}, options_cpu_int32_).pin_memory());
        }
    }

    if (num_tokens_per_bs_ > 1) {
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, max_seq_len_ - num_tokens_per_bs_, options_cpu_int32_).pin_memory();
        inputs.attention_inputs.prefix_lengths_d = inputs.attention_inputs.prefix_lengths.to(options_npu_int32_.device());
    } else {
        inputs.attention_inputs.prefix_lengths   = torch::empty({0}, options_cpu_int32_).pin_memory();
    }
    inputs.attention_inputs.padding_offset            = torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.padding_offset            = inputs.attention_inputs.padding_offset.pin_memory();
    inputs.attention_inputs.dtype                     = model_data_type_;
    inputs.attention_inputs.is_s_padded               = true;
    inputs.attention_inputs.sequence_lengths_plus_1_d = torch::full({int(max_bs_)}, seq_size_per_block_, options_npu_int32_);
    inputs.attention_inputs.decode_cu_seqlens_d       = torch::arange(0, max_bs_ + 1, 1, options_npu_int32_);
}

void AscendGraphRunner::initCaptureBertEmbeddingInputs(PyModelInputs& inputs,
                                                       int /*max_bs*/,
                                                       int /*max_num_token*/) {
    inputs.bert_embedding_inputs.combo_position_ids =
        torch::zeros({max_seq_len_ * static_cast<int>(max_bs_)}, options_npu_int32_);
    inputs.bert_embedding_inputs.position_encoding       = position_encoding_;
    inputs.bert_embedding_inputs.combo_tokens_type_ids   =
        torch::zeros({max_seq_len_ * static_cast<int>(max_bs_)}, options_npu_int32_);
    inputs.bert_embedding_inputs.token_type_embedding    = token_type_embedding_;
    inputs.bert_embedding_inputs.input_embedding_scalar  = input_embedding_scalar_;
}

void AscendGraphRunner::initKernelInternalMemory() {
    torch::Tensor cu_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU)).pin_memory();
    torch::Tensor cu_kv_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU)).pin_memory();
    auto input_lengths  = capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths;
    auto prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;

    cu_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.cumsum(0);
    if (prefix_lengths.defined() && prefix_lengths.size(0) > 0) {
        cu_kv_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.add(prefix_lengths).cumsum(0);
    }
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens_host = cu_seqlens;
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens      = cu_seqlens.to(options_npu_int32_.device());
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens   = cu_kv_seqlens.to(options_npu_int32_.device());
}

void AscendGraphRunner::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int num_tokens) {
    const auto& cap = capture_mem_hold_.py_model_inputs_;
    inputs.attention_inputs.is_target_verify = is_target_verify_;
    inputs.attention_inputs.is_prefill       = num_tokens_per_bs_ > 1;
    inputs.input_ids     = cap.input_ids.slice(0, 0, num_tokens);
    inputs.input_hiddens = cap.input_hiddens.slice(0, 0, num_tokens);
    inputs.attention_inputs.input_lengths    = cap.attention_inputs.input_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.input_lengths_d  = cap.attention_inputs.input_lengths_d.slice(0, 0, batch_size);
    inputs.attention_inputs.padding_offset   = cap.attention_inputs.padding_offset.slice(0, 0, num_tokens);

    if (cap.attention_inputs.prefix_lengths.defined() && cap.attention_inputs.prefix_lengths.size(0) > 0) {
        inputs.attention_inputs.prefix_lengths   = cap.attention_inputs.prefix_lengths.slice(0, 0, batch_size);
        inputs.attention_inputs.prefix_lengths_d = cap.attention_inputs.prefix_lengths_d.slice(0, 0, batch_size);
    } else {
        inputs.attention_inputs.prefix_lengths = cap.attention_inputs.prefix_lengths;
    }
    inputs.attention_inputs.sequence_lengths =
        cap.attention_inputs.sequence_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_kernel_block_id_device =
        cap.attention_inputs.kv_cache_kernel_block_id_device.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_kernel_block_id_host =
        cap.attention_inputs.kv_cache_kernel_block_id_host.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_device =
        cap.attention_inputs.kv_cache_block_id_device.defined() ?
            cap.attention_inputs.kv_cache_block_id_device.slice(0, 0, batch_size) :
            torch::Tensor();
    inputs.attention_inputs.kv_cache_block_id_host =
        cap.attention_inputs.kv_cache_block_id_host.defined() ?
            cap.attention_inputs.kv_cache_block_id_host.slice(0, 0, batch_size) :
            torch::Tensor();
    inputs.attention_inputs.cu_seqlens_host = cap.attention_inputs.cu_seqlens_host.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_seqlens      = cap.attention_inputs.cu_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_kv_seqlens   = cap.attention_inputs.cu_kv_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens_d =
        cap.attention_inputs.decode_cu_seqlens_d.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.sequence_lengths_plus_1_d =
        cap.attention_inputs.sequence_lengths_plus_1_d.slice(0, 0, batch_size);

    inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.clear();
    inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.clear();
    if (!cap.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()
        && !cap.attention_inputs.kv_cache_kernel_block_id_host_by_group.empty()) {
        const size_t group = cap.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.reserve(group);
        inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.reserve(group);
        for (size_t g = 0; g < group; ++g) {
            inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.push_back(
                cap.attention_inputs.kv_cache_kernel_block_id_device_by_group[g].slice(0, 0, batch_size));
            inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.push_back(
                cap.attention_inputs.kv_cache_kernel_block_id_host_by_group[g].slice(0, 0, batch_size));
        }
    }

    inputs.attention_inputs.dtype                 = cap.attention_inputs.dtype;
    inputs.attention_inputs.kv_cache_layer_to_group = cap.attention_inputs.kv_cache_layer_to_group;
    inputs.bert_embedding_inputs                  = cap.bert_embedding_inputs;
    inputs.attention_inputs.is_s_padded           = true;
}

ascend_graph::AscendGraphMemHold AscendGraphRunner::createMemHold(PyModelInputs& inputs, int tokens_count) {
    return ascend_graph::AscendGraphMemHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, tokens_count),
                                            inputs,
                                            /*is_embedding=*/false);
}

// ============================================================
// initCapture
// ============================================================

void AscendGraphRunner::initCapture() {
    if (!enable_graph_) {
        RTP_LLM_LOG_INFO("Ascend graph capture is not enabled, skipping initialization");
        return;
    }

    RTP_LLM_LOG_INFO("Ascend graph capture is enabled (decode-only)");
    shared_graph_pool_ = ascend_graph::GraphPoolHandle{};

    max_num_token_ = max_bs_ * num_tokens_per_bs_;
    capture_range_ = getDecodeBatchSizesToCapture();

    PyModelInputs inputs;
    inputs.input_ids     = torch::zeros({max_num_token_}, options_npu_int32_);
    inputs.input_hiddens = torch::zeros({max_num_token_, hidden_size_}, options_npu_float_);
    initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);
    initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

    torch::Tensor output;
    capture_mem_hold_ = ascend_graph::AscendGraphMemHold(output, inputs, /*is_embedding=*/false);
    initKernelInternalMemory();

    // Warm-up forward to settle output dtype / kernel lazy init.
    auto attn_pyobj = py_attn_pyobj_method_(capture_mem_hold_.py_model_inputs_, true);
    RTP_LLM_LOG_INFO("AscendGraphRunner initCapture warmup forward start");

    py_forward_method_(capture_mem_hold_.py_model_inputs_, attn_pyobj);
    RTP_LLM_LOG_INFO("AscendGraphRunner initCapture warmup forward end");
    output = torch::zeros({max_num_token_, hidden_size_}, options_npu_float_);
    capture_mem_hold_.setHiddenStates(output);

    captureDecode();
    RTP_LLM_LOG_INFO("Ascend graph initCapture done, captured %zu graph instances",
                     graph_instances_.size());
}

// ============================================================
// Capture
// ============================================================

void AscendGraphRunner::captureDecodeOneBatchSize(int bs) {
    captureOneGraphInstance(bs, "batch size");
}

void AscendGraphRunner::captureDecode() {
    std::string range_str;
    for (size_t i = 0; i < capture_range_.size(); ++i) {
        range_str += std::to_string(capture_range_[i]);
        if (i + 1 < capture_range_.size()) range_str += ", ";
    }
    RTP_LLM_LOG_INFO("Ascend graph Capture Decode Start, %zu buckets (capture order large->small): [%s]",
                     capture_range_.size(),
                     range_str.c_str());
    for (int bs : capture_range_) {
        graph_instances_.try_emplace(bs);
    }
    // Capture from large to small so mempool high-water mark is set first.
    int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        prepareCaptureInputs(inputs, bs, bs * num_tokens_per_bs_);

        int max_input_len  = inputs.attention_inputs.input_lengths.max().item<int>();
        int max_prefix_len = 0;
        if (inputs.attention_inputs.prefix_lengths.defined()
            && inputs.attention_inputs.prefix_lengths.size(0) > 0) {
            max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
        }
        inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);

        graph_instances_[bs].mem_hold_ = createMemHold(inputs, bs * num_tokens_per_bs_);
        graph_instances_[bs].mem_hold_.attn_pyobj_ =
            py_attn_pyobj_method_(graph_instances_[bs].mem_hold_.py_model_inputs_, true);
        captureDecodeOneBatchSize(bs);
        replayAndSyncCheck(bs, "batch size");
        RTP_LLM_LOG_INFO("ascend graph capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Ascend graph Capture Decode End");
}

void AscendGraphRunner::captureOneGraphInstance(int key, const char* key_type) {
#if USING_ASCEND
    auto inputs = graph_instances_[key].mem_hold_.py_model_inputs_;

    RTP_LLM_LOG_INFO("Ascend graph WarmUp for %s %d start.", key_type, key);
    auto attn_pyobj = graph_instances_[key].mem_hold_.attn_pyobj_;
    try {
        py_forward_method_(inputs, attn_pyobj);
        py_forward_method_(inputs, attn_pyobj);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Ascend graph WarmUp forward failed for %s %d: %s", key_type, key, e.what());
        throw;
    }
    RTP_LLM_LOG_INFO("Ascend graph WarmUp for %s %d successfully.", key_type, key);

    {
        // Sync before capture so the warm-up ops finish first.
        ascend_graph::graphDeviceSynchronize();

        ascend_graph::AscendGraphStreamLife stream_life(capture_stream_);
        auto& graph = graph_instances_[key].graph_;

        RTP_LLM_LOG_INFO("Ascend graph Capture for %s %d begin.", key_type, key);
        PyModelOutputs outputs;
        {
            ascend_graph::graphCaptureBegin(graph, shared_graph_pool_, ascend_graph::GraphCaptureMode::Relaxed);
            ascend_graph::AscendGraphCaptureGuard capture_guard;
            try {
                auto py_outputs_obj = py_forward_method_(inputs, attn_pyobj);
                outputs             = py_outputs_obj.cast<PyModelOutputs>();
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_ERROR("Ascend graph capture forward failed for %s %d: %s", key_type, key, e.what());
                throw;
            }
            // Copy forward output into persistent buffer so replay-time readers have a stable address.
            graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
            ascend_graph::graphCaptureEnd(graph);
        }
        RTP_LLM_LOG_INFO("Ascend graph Capture for %s %d success, captured output shape: [%lld x %lld]",
                         key_type,
                         key,
                         (long long)graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.size(0),
                         (long long)graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.size(1));
    }
#else
    (void)key;
    (void)key_type;
#endif
}

void AscendGraphRunner::replayGraph(int key) {
#if USING_ASCEND
    ascend_graph::graphReplay(graph_instances_[key].graph_);
#else
    (void)key;
#endif
}

void AscendGraphRunner::replayDecode(int bs) {
    RTP_LLM_LOG_DEBUG("Ascend graph replayDecode for bs=%d", bs);
    replayGraph(bs);
}

void AscendGraphRunner::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("ascend graph replay start check for %s %d", key_type, key);
    replayGraph(key);
    ascend_graph::graphDeviceSynchronize();
    RTP_LLM_LOG_INFO("ascend graph replay end check for %s %d", key_type, key);
}

// ============================================================
// prepareInputs / forward (replay path)
// ============================================================

// Helper: copy `src` data into `dst` slice. Supports host->host, host->device,
// device->device, contiguous or 2-D strided. Mirrors the per-tensor semantics
// of CudaGraphRunner::prepareInputs without the fused kernel launch (which is
// CUDA-specific). On Ascend, tensor.copy_ dispatches to aclrtMemcpyAsync.
static void copyTensorSlice(const torch::Tensor& src, torch::Tensor& dst) {
    if (!src.defined() || !dst.defined() || src.numel() <= 0) return;
    RTP_LLM_PROFILE_SCOPE("ascend_graph.copyTensorSlice");
    auto s = src;
    auto d = dst;
    // Squeeze leading singleton dims until dims match
    while (s.dim() > d.dim() && s.size(0) == 1) {
        s = s.squeeze(0);
    }
    while (d.dim() > s.dim() && d.size(0) == 1) {
        d = d.squeeze(0);
    }
    if (s.sizes() == d.sizes()) {
        d.copy_(s, /*non_blocking=*/true);
        return;
    }
    if (s.dim() < 2) {
        d.slice(0, 0, s.size(0)).copy_(s, /*non_blocking=*/true);
        return;
    }
    // When src has more dims than dst (e.g. 3D [group, batch, cols] -> 2D [batch, cols]),
    // select group 0 slice instead of flattening. This mirrors the Python-side
    // _squeeze_block_table / compute_ascend_attn_params logic and the legacy
    // field handling in setupKVCacheForAttentionInputs (which also uses group 0).
    if (s.dim() != d.dim()) {
        while (s.dim() > d.dim()) {
            s = s.size(0) == 1 ? s.squeeze(0) : s[0];
        }
        if (s.sizes() == d.sizes()) {
            d.copy_(s, /*non_blocking=*/true);
            return;
        }
    }
    int64_t rows = std::min(s.size(0), d.size(0));
    int64_t cols = std::min(s.size(1), d.size(1));
    d.slice(0, 0, rows).slice(1, 0, cols).copy_(
        s.slice(0, 0, rows).slice(1, 0, cols), /*non_blocking=*/true);
}

void AscendGraphRunner::prepareInputs(const PyModelInputs& inputs, CudaGraphState& state) {
    RTP_LLM_PROFILE_SCOPE("ascend_graph.prepareInputs");
    RTP_LLM_LOG_DEBUG("Ascend graph prepareInputs: batch_size=%d -> graph_bs=%d, token_num=%d",
                      state.current_batch_size,
                      state.current_real_graph_bs,
                      inputs.input_ids.size(0));
    // Wait for the previous forward to finish before overwriting persistent buffers.
    forward_event_.synchronize();

    const size_t graph_idx       = state.current_real_graph_bs;
    auto&        py_model_inputs = graph_instances_[graph_idx].mem_hold_.py_model_inputs_;
    auto         attn_pyobj      = graph_instances_[graph_idx].mem_hold_.attn_pyobj_;

    // Block tables must be cleared before each replay to prevent stale KV cache
    // block IDs from polluting subsequent batches (same rationale as CudaGraphRunner).
    py_model_inputs.attention_inputs.kv_cache_kernel_block_id_device.fill_(0);
    py_model_inputs.attention_inputs.kv_cache_kernel_block_id_host.fill_(0);
    if (py_model_inputs.attention_inputs.kv_cache_block_id_device.defined())
        py_model_inputs.attention_inputs.kv_cache_block_id_device.fill_(0);
    if (py_model_inputs.attention_inputs.kv_cache_block_id_host.defined())
        py_model_inputs.attention_inputs.kv_cache_block_id_host.fill_(0);
    for (auto& tbl_d : py_model_inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group) {
        tbl_d.fill_(0);
    }
    for (auto& tbl_h : py_model_inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group) {
        tbl_h.fill_(0);
    }

    const int token_num = inputs.input_ids.size(0);

    // -------- Device tensors (D2D) --------
    copyTensorSlice(inputs.input_ids, py_model_inputs.input_ids);
    if (inputs.input_hiddens.defined() && inputs.input_hiddens.numel() > 0) {
        py_model_inputs.input_hiddens.slice(0, 0, token_num).copy_(inputs.input_hiddens, /*non_blocking=*/true);
    }
    copyTensorSlice(inputs.attention_inputs.cu_seqlens, py_model_inputs.attention_inputs.cu_seqlens);   
    copyTensorSlice(inputs.attention_inputs.cu_kv_seqlens, py_model_inputs.attention_inputs.cu_kv_seqlens);
    copyTensorSlice(inputs.attention_inputs.input_lengths_d, py_model_inputs.attention_inputs.input_lengths_d);         
    copyTensorSlice(inputs.attention_inputs.kv_cache_kernel_block_id_device,
                    py_model_inputs.attention_inputs.kv_cache_kernel_block_id_device);
    copyTensorSlice(inputs.attention_inputs.kv_cache_block_id_device,
                    py_model_inputs.attention_inputs.kv_cache_block_id_device);

    // Decode-only fields
    copyTensorSlice(inputs.attention_inputs.prefix_lengths_d, py_model_inputs.attention_inputs.prefix_lengths_d);
    copyTensorSlice(inputs.attention_inputs.sequence_lengths_plus_1_d,
                    py_model_inputs.attention_inputs.sequence_lengths_plus_1_d);
    copyTensorSlice(inputs.attention_inputs.decode_cu_seqlens_d,
                    py_model_inputs.attention_inputs.decode_cu_seqlens_d);

    // Hybrid cache per-group device block tables
    const bool has_hybrid_cache = !inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()
                                  && !py_model_inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty();
    if (has_hybrid_cache) {
        const size_t group = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        RTP_LLM_CHECK_WITH_INFO(
            group == py_model_inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size(),
            "ascend graph: kv_cache_kernel_block_id_device_by_group size mismatch");
        for (size_t g = 0; g < group; ++g) {
            copyTensorSlice(inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group[g],
                            py_model_inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group[g]);
        }
    }

    // -------- Host tensors (H2H, pinned memory) --------
    if (inputs.attention_inputs.cu_seqlens_host.defined()) {
        auto src = inputs.attention_inputs.cu_seqlens_host.slice(0, 0, state.current_batch_size + 1);
        py_model_inputs.attention_inputs.cu_seqlens_host.slice(0, 0, state.current_batch_size + 1).copy_(src);
    }
    if (inputs.attention_inputs.input_lengths.defined()) {
        auto src = inputs.attention_inputs.input_lengths.slice(0, 0, state.current_batch_size);
        py_model_inputs.attention_inputs.input_lengths.slice(0, 0, state.current_batch_size).copy_(src);
    }
    if (inputs.attention_inputs.prefix_lengths.defined() && inputs.attention_inputs.prefix_lengths.numel() > 0) {
        auto src = inputs.attention_inputs.prefix_lengths.slice(0, 0, state.current_batch_size);
        py_model_inputs.attention_inputs.prefix_lengths.slice(0, 0, state.current_batch_size).copy_(src);
    }
    // Host block tables (Ascend attention reads the host version)
    copyTensorSlice(inputs.attention_inputs.kv_cache_kernel_block_id_host,
                    py_model_inputs.attention_inputs.kv_cache_kernel_block_id_host);
    copyTensorSlice(inputs.attention_inputs.kv_cache_block_id_host,
                    py_model_inputs.attention_inputs.kv_cache_block_id_host);
    if (inputs.attention_inputs.kv_cache_layer_to_group.defined()
        && inputs.attention_inputs.kv_cache_layer_to_group.numel() > 0
        && py_model_inputs.attention_inputs.kv_cache_layer_to_group.defined()) {
        auto src = inputs.attention_inputs.kv_cache_layer_to_group;
        auto dst = py_model_inputs.attention_inputs.kv_cache_layer_to_group;
        // Reshape src to match dst shape (handle 1D vs 2D mismatch)
        if (src.sizes() != dst.sizes()) {
            src = src.reshape(dst.sizes());
        }
        dst.copy_(src, /*non_blocking=*/true);
    }
    if (inputs.attention_inputs.sequence_lengths.defined()) {
        auto src = inputs.attention_inputs.sequence_lengths.slice(0, 0, state.current_batch_size);
        py_model_inputs.attention_inputs.sequence_lengths.slice(0, 0, state.current_batch_size).copy_(src);
    }
    py_model_inputs.attention_inputs.context_total_kv_length =
        inputs.attention_inputs.context_total_kv_length;
    // Hybrid cache per-group host block tables
    if (has_hybrid_cache) {
        const size_t group = inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group.size();
        for (size_t g = 0; g < group; ++g) {
            copyTensorSlice(inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group[g],
                            py_model_inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group[g]);
        }
    }

    // -------- Padding region clear (batch < max) --------
    if (state.current_batch_size < static_cast<int>(max_bs_)) {
        py_model_inputs.attention_inputs.input_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
        py_model_inputs.attention_inputs.input_lengths_d.slice(0, state.current_batch_size, max_bs_).fill_(0);
        py_model_inputs.attention_inputs.sequence_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
        py_model_inputs.attention_inputs.sequence_lengths_plus_1_d.slice(0, state.current_batch_size, max_bs_).fill_(0);
        py_model_inputs.attention_inputs.decode_cu_seqlens_d.slice(0, state.current_batch_size + 1, max_bs_ + 1).fill_(0);
        if (py_model_inputs.attention_inputs.prefix_lengths.defined()
            && py_model_inputs.attention_inputs.prefix_lengths.numel() > 0) {
            py_model_inputs.attention_inputs.prefix_lengths.slice(0, state.current_batch_size, max_bs_).fill_(0);
        }
        if (py_model_inputs.attention_inputs.prefix_lengths_d.defined()
            && py_model_inputs.attention_inputs.prefix_lengths_d.numel() > 0) {
            py_model_inputs.attention_inputs.prefix_lengths_d.slice(0, state.current_batch_size, max_bs_).fill_(0);
        }
        py_model_inputs.attention_inputs.kv_cache_kernel_block_id_device.slice(
            0, state.current_batch_size, max_bs_).fill_(0);
        py_model_inputs.attention_inputs.kv_cache_kernel_block_id_host.slice(
            0, state.current_batch_size, max_bs_).fill_(0);
        if (py_model_inputs.attention_inputs.kv_cache_block_id_device.defined())
            py_model_inputs.attention_inputs.kv_cache_block_id_device.slice(
                0, state.current_batch_size, max_bs_).fill_(0);
        if (py_model_inputs.attention_inputs.kv_cache_block_id_host.defined())
            py_model_inputs.attention_inputs.kv_cache_block_id_host.slice(
                0, state.current_batch_size, max_bs_).fill_(0);
        for (auto& tbl_d : py_model_inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group) {
            tbl_d.slice(0, state.current_batch_size, max_bs_).fill_(0);
        }
        for (auto& tbl_h : py_model_inputs.attention_inputs.kv_cache_kernel_block_id_host_by_group) {
            tbl_h.slice(0, state.current_batch_size, max_bs_).fill_(0);
        }
    }

    // Compute sequence_lengths_plus_1_d from sequence_lengths (engine doesn't populate it)
    py_model_inputs.attention_inputs.sequence_lengths_plus_1_d.slice(0, 0, state.current_batch_size) =
        py_model_inputs.attention_inputs.sequence_lengths.slice(0, 0, state.current_batch_size)
            .to(options_npu_int32_.device()) + 1;

    // -------- Update attention impl with the freshly-copied inputs --------
    {
        RTP_LLM_PROFILE_SCOPE("ascend_graph.prepareInputs(prepare_cuda_graph)");
        attn_pyobj.attr("prepare_cuda_graph")(py_model_inputs.attention_inputs);
    }
}

PyModelOutputs AscendGraphRunner::forward(const PyModelInputs& inputs, CudaGraphState& state) {
    PyModelOutputs outputs;
    RTP_LLM_LOG_DEBUG("Ascend graph Replay Start, batch_size=%d -> graph_bs=%d, seq_len_sum=%d",
                      state.current_batch_size,
                      state.current_real_graph_bs,
                      state.seq_len_sum);
    prepareInputs(inputs, state);
    {
        RTP_LLM_PROFILE_SCOPE("ascend_graph.forward(replayDecode)");
        replayDecode(state.current_real_graph_bs);
    }
    // Read output from the persistent buffer slice (stable address across replays).
    outputs.hidden_states =
        graph_instances_[state.current_real_graph_bs].mem_hold_.decoder_layer_hidden_states_.slice(
            0, 0, state.seq_len_sum).clone();
    forward_event_.record(ascend_graph::graphGetCurrentStream());
    RTP_LLM_LOG_DEBUG("Ascend graph Replay End, graph_bs=%d, output shape: [%lld x %lld]",
                      state.current_real_graph_bs,
                      (long long)outputs.hidden_states.size(0),
                      (long long)outputs.hidden_states.size(1));
    return outputs;
}

}  // namespace rtp_llm
