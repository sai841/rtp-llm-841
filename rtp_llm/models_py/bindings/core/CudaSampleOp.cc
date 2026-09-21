#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include <cstddef>
#include <random>
#include <memory>
#elif USING_ASCEND
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#endif

using namespace std;

namespace rtp_llm {

#if USING_CUDA

using SamplerT = float;

namespace {

void processLogits(const GreedyParams&  params,
                   const torch::Tensor& device_tokens,
                   const torch::Tensor& transposed_tokens) {
    const auto vocab_size_padded  = params.logits.size(1);
    const auto decoder_batch_size = params.sequence_lengths.size(0);
    const auto batch_size         = params.logits.size(0);
    const auto step               = params.step;
    auto       cur_stream         = at::cuda::getCurrentCUDAStream().stream();

    if (std::any_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [&](auto t) { return t != 1.0f; })) {
        auto temperature_gpu = params.temperature.to(torch::kCUDA, true);
        invokeBatchApplyTemperaturePenalty(params.logits.data_ptr<float>(),
                                           (float*)nullptr,  // embedding_bias
                                           temperature_gpu.data_ptr<float>(),
                                           batch_size,
                                           vocab_size_padded,
                                           vocab_size_padded,
                                           cur_stream);
    }

    if (params.repetition_penalty.has_value()) {
        RTP_LLM_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value());
        const auto& repetition_penalty = params.repetition_penalty.value();
        const auto& presence_penalty   = params.presence_penalty.value();
        const auto& frequency_penalty  = params.frequency_penalty.value();
        if (std::any_of(repetition_penalty.data_ptr<float>(),
                        repetition_penalty.data_ptr<float>() + batch_size,
                        [&](auto t) { return t != 1.0f; })
            || std::any_of(presence_penalty.data_ptr<float>(),
                           presence_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })
            || std::any_of(frequency_penalty.data_ptr<float>(),
                           frequency_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })) {
            // Build sequence_lengths: clone input_lengths, then overwrite first decoder_batch_size entries
            auto sequence_lengths_gpu = params.input_lengths.to(torch::kCUDA, true);
            if (decoder_batch_size > 0) {
                auto dst_slice = sequence_lengths_gpu.slice(0, 0, decoder_batch_size);
                dst_slice.copy_(params.sequence_lengths.to(torch::kCUDA, true), true);
            }
            auto penalty_ws             = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size_padded},
                                           torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            auto repetition_penalty_gpu = repetition_penalty.to(torch::kCUDA, true);
            auto presence_penalty_gpu   = presence_penalty.to(torch::kCUDA, true);
            auto frequency_penalty_gpu  = frequency_penalty.to(torch::kCUDA, true);
            invokeBatchApplyRepetitionPenalty(params.logits.data_ptr<float>(),
                                              penalty_ws.data_ptr<int32_t>(),
                                              repetition_penalty_gpu.data_ptr<float>(),
                                              presence_penalty_gpu.data_ptr<float>(),
                                              frequency_penalty_gpu.data_ptr<float>(),
                                              transposed_tokens.data_ptr<int32_t>(),
                                              batch_size,
                                              batch_size,  // local_batch_size
                                              vocab_size_padded,
                                              sequence_lengths_gpu.data_ptr<int32_t>(),
                                              step + 1,  // max_input_length
                                              step + 1,  // step
                                              cur_stream);
            // NOTE: here step is max_len - 1
        }
    }

    if (decoder_batch_size && params.no_repeat_ngram_size.has_value()) {
        const auto& no_repeat_ngram_size = params.no_repeat_ngram_size.value();
        if (any_of(no_repeat_ngram_size.data_ptr<int32_t>(),
                   no_repeat_ngram_size.data_ptr<int32_t>() + decoder_batch_size,
                   [](auto s) { return s != 0; })) {
            auto no_repeat_ngram_size_gpu = no_repeat_ngram_size.to(torch::kCUDA, true);
            // Build array of pointers to each batch's token ids
            auto output_ids_ptrs =
                torch::empty({(int64_t)decoder_batch_size}, torch::TensorOptions().dtype(torch::kInt64)).pin_memory();
            for (int64_t i = 0; i < (int64_t)decoder_batch_size; i++) {
                output_ids_ptrs.data_ptr<int64_t>()[i] = (int64_t)(device_tokens.data_ptr<int32_t>() + i * (step + 1));
            }
            auto output_ids_ptrs_gpu  = output_ids_ptrs.to(torch::kCUDA, true);
            auto sequence_lengths_gpu = params.sequence_lengths.to(torch::kCUDA, true);

            tensorrt_llm::kernels::invokeBanRepeatNgram(params.logits.data_ptr<float>(),
                                                        (int32_t const**)(output_ids_ptrs_gpu.data_ptr()),
                                                        nullptr,  // finished_buf
                                                        nullptr,  // parent_ids_buf
                                                        nullptr,  // batch_slot
                                                        sequence_lengths_gpu.data_ptr<int32_t>(),
                                                        decoder_batch_size,
                                                        1,  // beam_width
                                                        step + 1,
                                                        no_repeat_ngram_size_gpu.data_ptr<int32_t>(),
                                                        vocab_size_padded,
                                                        step + 1,
                                                        cur_stream);
        }
    }
}

}  // anonymous namespace

static GreedyOutput flashinferSampleGreedy(const GreedyParams& params, const torch::Tensor& transposed_tokens) {
    const auto batch_size = params.logits.size(0);
    auto       cur_stream = at::cuda::getCurrentCUDAStream().stream();

    // [batch_size, vocab_size] — compute softmax probabilities.
    // Copy result back to logits to preserve the in-place behavior of the original kernel,
    // since callers may reuse the logits tensor across iterations.
    auto probs_t = torch::softmax(params.logits, -1);
    params.logits.copy_(probs_t, true);
    auto success = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    // [1, batch_size] — last row of transposed_tokens
    auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0));

    torch::TensorOptions options          = torch::TensorOptions(probs_t.scalar_type()).device(torch::kCUDA);
    constexpr bool       deterministic    = true;
    constexpr int        max_top_k_rounds = 32;
    auto                 uniform_samples  = torch::rand({max_top_k_rounds, (int)batch_size}, options);
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
        if (params.generator[i].defined()) {
            uniform_samples.index({torch::indexing::Slice(), i}) =
                torch::rand({max_top_k_rounds}, params.generator[i], nullopt, options);
        }
    }

    torch::Tensor success_t = success;
    torch::Tensor top_k_t   = params.top_k;
    torch::Tensor top_p_t   = params.top_p;
    torch::Tensor output_all_probs_t;
    if (params.output_all_probs.has_value()) {
        output_all_probs_t = params.output_all_probs.value();
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }

    // top_k/top_p are CPU tensors with int32/float32 dtype
    auto top_k_ptr = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
    auto top_p_ptr = params.top_p.data_ptr<float>();

    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [&](auto t) { return std::abs(t) < 1e-7 ? 1.0 : t; });

    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens, true);
        success = torch::Tensor();  // mark as undefined — all succeeded
        if (output_all_probs_t.defined()) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)cur_stream);
        }
    } else if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t <= 0; })) {
        top_p_sampling_from_probs(
            probs_t, uniform_samples, samples_t, success_t, top_p_t, 1.0, deterministic, (int64_t)cur_stream);
        if (output_all_probs_t.defined()) {
            top_p_renorm_probs(probs_t, output_all_probs_t, top_p_t, 1.0, (int64_t)cur_stream);
        }
    } else if (std::all_of(top_p_ptr, top_p_ptr + batch_size, [&](auto t) { return std::abs(t - 1.0f) < 1e-7; })) {
        std::transform(top_k_ptr, top_k_ptr + batch_size, top_k_ptr, [&](auto t) { return t <= 0 ? 1 << 30 : t; });
        top_k_sampling_from_probs(
            probs_t, uniform_samples, samples_t, success_t, top_k_t, 0, deterministic, (int64_t)cur_stream);
        if (output_all_probs_t.defined()) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)cur_stream);
        }
    } else {
        std::transform(top_k_ptr, top_k_ptr + batch_size, top_k_ptr, [&](auto t) { return t <= 0 ? 1 << 30 : t; });
        top_k_top_p_sampling_from_probs(probs_t,
                                        uniform_samples,
                                        samples_t,
                                        success_t,
                                        top_k_t,
                                        1.0,
                                        top_p_t,
                                        1.0,
                                        deterministic,
                                        (int64_t)cur_stream);
        if (output_all_probs_t.defined()) {
            torch::Tensor temp_t = torch::zeros_like(output_all_probs_t);
            top_k_renorm_probs(probs_t, temp_t, top_k_t, 1.0, (int64_t)cur_stream);
            top_p_renorm_probs(temp_t, output_all_probs_t, top_p_t, 1.0, (int64_t)cur_stream);
        }
    }

    if (params.cum_log_probs.has_value()) {

        // [batch_size]
        auto cum_log_probs_t = params.cum_log_probs.value();
        // [batch_size]
        auto token_probs_t     = output_all_probs_t.gather(1, samples_t.transpose(1, 0).to(torch::kLong)).squeeze(1);
        auto token_probs_t_log = token_probs_t.log();
        cum_log_probs_t.add_(token_probs_t_log.to(cum_log_probs_t.device()));
    }

    // Copy results back: transpose and copy to token_ids
    auto transposed_t = transposed_tokens.transpose(0, 1).contiguous();
    params.token_ids.copy_(transposed_t, true);
    check_cuda_error();
    return {success};
}

GreedyOutput sampleGreedy(const GreedyParams& params) {
    // [batch_size, step + 1] — clone to GPU
    auto device_tokens = params.token_ids.to(torch::kCUDA, true);
    // [step + 1, batch_size]
    auto transposed_tokens = device_tokens.transpose(0, 1).contiguous();

    const auto batch_size        = params.logits.size(0);
    bool       has_not_do_sample = params.do_sample.has_value()
                             && std::any_of(params.do_sample.value().data_ptr<bool>(),
                                            params.do_sample.value().data_ptr<bool>() + batch_size,
                                            [&](auto t) { return !t; });
    bool need_do_sample = (!params.do_sample.has_value())
                          || std::any_of(params.do_sample.value().data_ptr<bool>(),
                                         params.do_sample.value().data_ptr<bool>() + batch_size,
                                         [&](auto t) { return t; });
    if (need_do_sample) {
        torch::Tensor selected_logits;
        torch::Tensor mask_tensor;
        if (has_not_do_sample) {
            auto do_sample_gpu = params.do_sample.value().to(torch::kCUDA, true);
            mask_tensor        = do_sample_gpu.reshape({(int64_t)batch_size, 1}).logical_not();
            selected_logits    = params.logits.masked_select(mask_tensor);
        }
        processLogits(params, device_tokens, transposed_tokens);
        if (has_not_do_sample) {
            params.logits.masked_scatter_(mask_tensor, selected_logits);
        }
    }

    // fast path for topk = 1
    auto top_k_ptr = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })
        && !params.output_all_probs.has_value()) {
        torch::Tensor samples_t =
            transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).squeeze(0);
        torch::Tensor probs_t         = params.logits;
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens, true);

        auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
        params.token_ids.copy_(output_tokens, true);

        return GreedyOutput{};
    }

    return flashinferSampleGreedy(params, transposed_tokens);
}

void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    chain_speculative_sampling(params.draft_probs_d,
                               params.draft_token_ids_d,
                               params.uniform_samples_d,
                               params.target_probs_d,
                               params.output_token_ids_d,
                               params.output_accepted_token_num_d,
                               params.output_emitted_token_num_d,
                               true,
                               int64_t(stream));
}

#elif USING_ASCEND

}  // namespace rtp_llm — close temporarily for torch_npu includes

#include "rtp_llm/models_py/bindings/ascend/ops/ascend_apply_top_k_top_p_custom.h"

namespace rtp_llm {  // reopen

// ============================================================
// Sample ops (Ascend) — NPU-accelerated implementation
//
// Strategy:
//   - Temperature:        PyTorch div_() → torch_npu maps to NPU
//   - Top-k/Top-p:        AscendC custom op applyTopKTopP
//   - Repetition penalty: not yet available (AscendC kernel TBD)
//   - Sampling:           torch::multinomial (same fallback as ROCm)
// ============================================================

GreedyOutput sampleGreedy(const GreedyParams& params) {
    static const auto device_type = torch::Device(torch::kPrivateUse1);
    const auto        batch_size  = params.logits.size(0);

    // [batch_size, step + 1] — clone to NPU
    auto device_tokens     = params.token_ids.to(device_type);
    // [step + 1, batch_size]
    auto transposed_tokens = device_tokens.transpose(0, 1).contiguous();

    // ---- Handle do_sample: save logits for non-sampling (greedy) batches ----
    // temperature <= 0 also means greedy (OpenAI convention: temperature=0 = greedy).
    bool need_do_sample    = (!params.do_sample.has_value()) ||
                              std::any_of(params.do_sample.value().data_ptr<bool>(),
                                         params.do_sample.value().data_ptr<bool>() + batch_size,
                                         [](auto t) { return t; });
    bool has_temp_greedy   = need_do_sample &&
                             std::any_of(params.temperature.data_ptr<float>(),
                                         params.temperature.data_ptr<float>() + batch_size,
                                         [](auto t) { return t <= 0.0f; });
    bool has_not_do_sample = (params.do_sample.has_value() &&
                              std::any_of(params.do_sample.value().data_ptr<bool>(),
                                          params.do_sample.value().data_ptr<bool>() + batch_size,
                                          [](auto t) { return !t; })) || has_temp_greedy;

    torch::Tensor selected_logits;
    torch::Tensor mask_tensor;  // true = greedy rows (should NOT be sampled/penalized)
    if (has_not_do_sample && need_do_sample) {
        if (params.do_sample.has_value()) {
            auto do_sample_npu = params.do_sample.value().to(device_type);
            mask_tensor = do_sample_npu.reshape({(int64_t)batch_size, 1}).logical_not();
        }
        if (has_temp_greedy) {
            auto temp_npu = params.temperature.to(device_type);
            auto temp_mask = (temp_npu <= 0.0f).reshape({(int64_t)batch_size, 1});
            mask_tensor = mask_tensor.defined() ? mask_tensor.logical_or(temp_mask) : temp_mask;
        }
        selected_logits = params.logits.masked_select(mask_tensor);
    }

    // ---- 1. Apply temperature penalty (PyTorch op, torch_npu handles NPU) ----
    if (need_do_sample) {
        if (std::any_of(params.temperature.data_ptr<float>(),
                        params.temperature.data_ptr<float>() + batch_size,
                        [](auto t) { return t != 1.0f; })) {
            // temperature == 0 requests greedy decoding but do_sample stays true:
            // dividing by zero yields inf/NaN and breaks argmax. Treat
            // non-positive temperatures as 1.0 (no scaling, pure greedy).
            auto temperature_npu = params.temperature.to(device_type);
            temperature_npu.masked_fill_(temperature_npu <= 0.0f, 1.0f);
            params.logits.div_(temperature_npu.reshape({(int64_t)batch_size, 1}));
        }
    }

    // Restore non-sampling batch logits (penalties should not affect greedy batches)
    if (has_not_do_sample && need_do_sample) {
        params.logits.masked_scatter_(mask_tensor, selected_logits);
    }

    // ---- 2. Fast path: top_k = 1 for all batches → argmax only ----
    auto top_k_ptr = params.top_k.data_ptr<int32_t>();  // signed read (top_k=-1 is "no top-k" sentinel)
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t == 1; }) &&
        !params.output_all_probs.has_value()) {
        auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1,
                                                  transposed_tokens.size(0)).squeeze(0);
        auto selected_tokens = torch::argmax(params.logits, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);

        auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
        params.token_ids.copy_(output_tokens);
        return GreedyOutput{};
    }

    // ---- 3. Apply top-k/top-p filtering via AscendC custom operator ----
    auto top_p_ptr = params.top_p.data_ptr<float>();
    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr,
                   [](auto t) { return std::abs(t) < 1e-7f ? 1.0f : t; });

    bool has_top_k = !std::all_of(top_k_ptr, top_k_ptr + batch_size,
                                   [](auto t) { return t <= 0; });
    bool has_top_p = !std::all_of(top_p_ptr, top_p_ptr + batch_size,
                                   [](auto t) { return std::abs(t - 1.0f) < 1e-7f; });

    if (has_top_k || has_top_p) {
        // Move top_k/top_p from CPU to NPU.
        // applyTopKTopP expects: k → INT32 1D [batch_size], p → FLOAT32 1D [batch_size].
        c10::optional<at::Tensor> k_npu;
        c10::optional<at::Tensor> p_npu;
        if (has_top_k) {
            k_npu = params.top_k.to(torch::TensorOptions().dtype(torch::kInt32).device(device_type));
        }
        if (has_top_p) {
            p_npu = params.top_p.to(torch::TensorOptions().dtype(torch::kFloat32).device(device_type));
        }

        at::Tensor output = at::empty_like(params.logits);
        rtp_llm::ascend::applyTopKTopP(params.logits, p_npu, k_npu, output);
        params.logits.copy_(output);
    }

    // ---- 4. Softmax → probabilities ----
    auto probs_t = torch::softmax(params.logits, -1);
    params.logits.copy_(probs_t);

    // ---- 5. Output_all_probs / cum_log_probs setup ----
    torch::Tensor output_all_probs_t;
    bool          need_output_all_probs = params.output_all_probs.has_value();
    if (need_output_all_probs) {
        output_all_probs_t = params.output_all_probs.value();
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }

    // ---- 6. Sample ----
    auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1,
                                              transposed_tokens.size(0)).squeeze(0);

    bool all_topk_1 = std::all_of(top_k_ptr, top_k_ptr + batch_size,
                                   [](auto t) { return t == 1; });
    if (all_topk_1) {
        // Greedy: argmax
        auto selected = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected);
    } else {
        // Random sample: multinomial (same fallback as ROCm).
        // Only use a generator whose device matches the probs tensor;
        // a mismatched generator (e.g. CPU gen on NPU data) causes a
        // RuntimeError. If no matching generator is found, pass nullopt
        // to use the default per-device generator (random in production).
        c10::optional<at::Generator> gen;
        for (auto& g : params.generator) {
            if (g.defined() && g.device().type() == probs_t.device().type()) {
                gen = g;
                break;
            }
        }
        auto selected = torch::multinomial(probs_t, 1, /*replacement=*/false, gen).squeeze(-1);
        // A mixed batch must honour per-request do_sample and temperature=0:
        // greedy rows (do_sample=false OR temperature<=0) are overwritten with
        // argmax instead of keeping their random multinomial draw.
        if (has_not_do_sample) {
            auto greedy_sel  = torch::argmax(probs_t, -1, /*keepdim=*/false);
            if (mask_tensor.defined()) {
                // mask_tensor is true for greedy rows (do_sample=false OR temp<=0)
                auto sample_mask = mask_tensor.squeeze(-1).logical_not();  // true = sample
                selected         = torch::where(sample_mask, selected, greedy_sel);
            } else {
                // need_do_sample=false: all rows are greedy
                selected = greedy_sel;
            }
        }
        samples_t.copy_(selected);
    }

    if (need_output_all_probs) {
        output_all_probs_t.copy_(probs_t);
    }

    // ---- 7. Update cum_log_probs ----
    if (params.cum_log_probs.has_value()) {
        auto cum_log_probs_t = params.cum_log_probs.value();
        // params.logits was overwritten with softmax probs in step 4, so
        // log_softmax(logits) double-softmaxes and yields wrong values.
        // Take the probability of the selected token directly from
        // probs_t and take its log instead.
        auto token_probs     = probs_t.gather(-1, samples_t.reshape({(int64_t)batch_size, 1})).squeeze(-1);
        auto token_log_probs = token_probs.clamp_min(1e-10f).log();
        cum_log_probs_t.add_(token_log_probs.to(cum_log_probs_t.device()));
    }

    // ---- 8. Copy results back to token_ids ----
    auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
    params.token_ids.copy_(output_tokens);

    return GreedyOutput{};
}

void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

#else  // !USING_CUDA — ROCm platform

}  // namespace rtp_llm — temporarily close for includes

// Forward-declare penalty kernels to avoid transitive amd_bfloat16.h breakage
// (sampling_penalty_kernels.h → cuda_shims.h → amd_bfloat16.h is broken in .cc on ROCm 6.4.3 + clang 19)
namespace rtp_llm {
template<typename T>
void invokeBatchApplyTemperaturePenalty(T*           logits,
                                        const T*     bias,
                                        const float* temperatures,
                                        int          batch_size,
                                        int          vocab_size,
                                        int          vocab_size_padd,
                                        hipStream_t  stream);
template<typename T>
void invokeBatchApplyRepetitionPenalty(T*           logits,
                                       int*         penalty_ws,
                                       const float* repetition_penalty,
                                       const float* presence_penalty,
                                       const float* frequency_penalty,
                                       const int*   output_ids,
                                       int          batch_size,
                                       int          local_batch_size,
                                       int          vocab_size,
                                       const int*   input_lengths,
                                       int          max_input_length,
                                       int          step,
                                       hipStream_t  stream);
}  // namespace rtp_llm

#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/kernels/sampling/sampling.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"

namespace rtp_llm {  // reopen

GreedyOutput sampleGreedy(const GreedyParams& params) {
    const auto batch_size         = params.logits.size(0);
    const auto vocab_size_padded  = params.logits.size(1);
    const auto step               = params.step;
    const auto decoder_batch_size = params.sequence_lengths.size(0);
    auto       cur_stream         = at::hip::getCurrentHIPStream().stream();

    // [batch_size, step + 1] — clone to GPU
    // On ROCm, hipMemcpyAsync from pageable memory is truly async (unlike CUDA where it
    // falls back to sync). Use blocking transfer to avoid memory access faults.
    auto device_tokens = params.token_ids.to(torch::kCUDA);
    // [step + 1, batch_size]
    auto transposed_tokens = device_tokens.transpose(0, 1).contiguous();

    // 1. Apply temperature penalty
    if (std::any_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [&](auto t) { return t != 1.0f; })) {
        auto temperature_gpu = params.temperature.to(torch::kCUDA);
        invokeBatchApplyTemperaturePenalty(params.logits.data_ptr<float>(),
                                           (float*)nullptr,  // embedding_bias
                                           temperature_gpu.data_ptr<float>(),
                                           batch_size,
                                           vocab_size_padded,
                                           vocab_size_padded,
                                           cur_stream);
    }

    // 2. Apply repetition/presence/frequency penalty
    if (params.repetition_penalty.has_value()) {
        RTP_LLM_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value());
        const auto& repetition_penalty = params.repetition_penalty.value();
        const auto& presence_penalty   = params.presence_penalty.value();
        const auto& frequency_penalty  = params.frequency_penalty.value();
        if (std::any_of(repetition_penalty.data_ptr<float>(),
                        repetition_penalty.data_ptr<float>() + batch_size,
                        [&](auto t) { return t != 1.0f; })
            || std::any_of(presence_penalty.data_ptr<float>(),
                           presence_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })
            || std::any_of(frequency_penalty.data_ptr<float>(),
                           frequency_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })) {
            auto sequence_lengths_gpu = params.input_lengths.to(torch::kCUDA);
            if (decoder_batch_size > 0) {
                auto dst_slice = sequence_lengths_gpu.slice(0, 0, decoder_batch_size);
                dst_slice.copy_(params.sequence_lengths.to(torch::kCUDA));
            }
            auto penalty_ws             = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size_padded},
                                           torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            auto repetition_penalty_gpu = repetition_penalty.to(torch::kCUDA);
            auto presence_penalty_gpu   = presence_penalty.to(torch::kCUDA);
            auto frequency_penalty_gpu  = frequency_penalty.to(torch::kCUDA);
            invokeBatchApplyRepetitionPenalty(params.logits.data_ptr<float>(),
                                              penalty_ws.data_ptr<int32_t>(),
                                              repetition_penalty_gpu.data_ptr<float>(),
                                              presence_penalty_gpu.data_ptr<float>(),
                                              frequency_penalty_gpu.data_ptr<float>(),
                                              transposed_tokens.data_ptr<int32_t>(),
                                              batch_size,
                                              batch_size,  // local_batch_size
                                              vocab_size_padded,
                                              sequence_lengths_gpu.data_ptr<int32_t>(),
                                              step + 1,  // max_input_length
                                              step + 1,  // step
                                              cur_stream);
        }
    }

    // 3. Fast path for topk = 1
    auto top_k_ptr = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })
        && !params.output_all_probs.has_value()) {
        torch::Tensor samples_t =
            transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).squeeze(0);
        torch::Tensor probs_t         = params.logits;
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);

        auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
        params.token_ids.copy_(output_tokens);

        return GreedyOutput{};
    }

    // 4. Compute softmax probabilities
    auto probs_t = torch::softmax(params.logits, -1);
    params.logits.copy_(probs_t);

    // 5. Prepare sampling parameters
    constexpr bool deterministic = true;
    auto           seed_h        = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    auto           offset_h      = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
        auto [sd, ofst] = get_seed_and_offset(
            batch_size * 32, params.generator[i].defined() ? std::make_optional(params.generator[i]) : std::nullopt);
        seed_h.data_ptr<int64_t>()[i]   = static_cast<int64_t>(sd);
        offset_h.data_ptr<int64_t>()[i] = static_cast<int64_t>(ofst);
    }

    auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).flatten();
    auto top_k_t   = params.top_k;
    auto top_p_t   = params.top_p;
    auto top_p_ptr = params.top_p.data_ptr<float>();

    bool          need_output_all_probs = params.output_all_probs.has_value();
    torch::Tensor output_all_probs_t;
    if (need_output_all_probs) {
        output_all_probs_t = params.output_all_probs.value();
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }

    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [&](auto t) { return std::abs(t) < 1e-7 ? 1.0 : t; });

    // 6. Sample
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);
        if (need_output_all_probs) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(cur_stream));
        }
    } else {
        // Use pure PyTorch sampling instead of FlashInfer ROCm kernels.
        // FlashInfer's ROCm sampling kernels (top_p_sampling_from_probs,
        // top_k_sampling_from_probs, top_k_top_p_sampling_from_probs) crash
        // with GPU memory access faults in multi-GPU (TP>1) configurations.
        // torch::multinomial is well-tested and handles all cases correctly.
        //
        // Apply top_k filtering if needed
        auto filtered_probs = probs_t;
        bool has_top_k      = !std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t <= 0; });
        if (has_top_k) {
            for (int64_t b = 0; b < (int64_t)batch_size; b++) {
                int k = top_k_ptr[b] <= 0 ? vocab_size_padded : top_k_ptr[b];
                if ((int64_t)k < vocab_size_padded) {
                    auto row                    = filtered_probs[b];
                    auto [topk_vals, topk_inds] = row.topk(k);
                    auto min_val                = topk_vals[-1];
                    row.masked_fill_(row < min_val, 0.0f);
                }
            }
        }
        // Apply top_p (nucleus) filtering if needed
        bool has_top_p =
            !std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto t) { return std::abs(t - 1.0f) < 1e-7; });
        if (has_top_p) {
            for (int64_t b = 0; b < (int64_t)batch_size; b++) {
                float p = top_p_ptr[b];
                if (std::abs(p - 1.0f) >= 1e-7) {
                    auto row                            = filtered_probs[b];
                    auto [sorted_probs, sorted_indices] = row.sort(/*dim=*/0, /*descending=*/true);
                    auto cumsum                         = sorted_probs.cumsum(0);
                    auto mask                           = cumsum - sorted_probs > p;
                    sorted_probs.masked_fill_(mask, 0.0f);
                    row.scatter_(0, sorted_indices, sorted_probs);
                }
            }
        }
        // Re-normalize and sample
        auto row_sums  = filtered_probs.sum(-1, /*keepdim=*/true);
        filtered_probs = filtered_probs / row_sums.clamp_min(1e-10);
        auto selected  = torch::multinomial(filtered_probs, 1, /*replacement=*/false).squeeze(-1);
        samples_t.copy_(selected);
        if (need_output_all_probs) {
            output_all_probs_t.copy_(filtered_probs);
        }
    }

    // 7. Update cum_log_probs
    if (params.cum_log_probs.has_value()) {
        auto cum_log_probs_t = params.cum_log_probs.value();
        cum_log_probs_t.add_(probs_t.log());
    }

    // 8. Copy results back
    auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
    params.token_ids.copy_(output_tokens);
    return GreedyOutput{};
}

}  // namespace rtp_llm

// Forward-declare in global namespace (matches rtp_llm/models_py/bindings/rocm/speculative_sampling/sampling.cu)
void chain_speculative_sampling(at::Tensor draft_probs,
                                at::Tensor draft_token_ids,
                                at::Tensor uniform_samples,
                                at::Tensor target_probs,
                                at::Tensor output_token_ids,
                                at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_draft_token_num,
                                bool       deterministic,
                                int64_t    hip_stream);

namespace rtp_llm {

void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    ::chain_speculative_sampling(params.draft_probs_d,
                                 params.draft_token_ids_d,
                                 params.uniform_samples_d,
                                 params.target_probs_d,
                                 params.output_token_ids_d,
                                 params.output_accepted_token_num_d,
                                 params.output_emitted_token_num_d,
                                 true,
                                 int64_t(stream));
}
#endif  // USING_CUDA

}  // namespace rtp_llm
