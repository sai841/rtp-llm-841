#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"

namespace rtp_llm {
void registerPyModuleOps(pybind11::module& m) {
    // Qwen3.5 / linear-attention models persist SSM & KV states through the
    // cache store after each prefill; the op itself is platform-agnostic
    // (common/WriteCacheStoreOp.cc), only the binding was missing on Ascend.
    m.def("write_cache_store",
          &WriteCacheStoreOp,
          "WriteCacheStoreOp kernel",
          pybind11::arg("input_lengths"),
          pybind11::arg("prefix_lengths"),
          pybind11::arg("kv_cache_block_id_host"),
          pybind11::arg("cache_store_member"),
          pybind11::arg("kv_cache"));
}
}
