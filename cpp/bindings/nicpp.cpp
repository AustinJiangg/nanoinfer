// pybind11 bindings (C++ stage F6) — expose the nanoinfer C++ core to Python.
//
// This is the pivot from "pure C++ engine" to the vLLM shape: Python orchestration
// on top of our own C++ kernels. The binding stays thin — it hands Python the same
// objects the C++ drivers use (Model, KVCache, SamplingParams) plus a forward() that
// returns logits as a numpy array, so the Python serving layer (F7: scheduler +
// continuous batching) can be built on these primitives without touching the kernels.
//
// Tokenization stays in Python (HF AutoTokenizer, allowed by the golden rule); the
// core works purely on token ids, exactly like the C++ drivers.
//
// The heavy calls (forward/generate) drop the GIL while in C++ so a future
// multi-threaded Python driver isn't serialized on the kernels.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "cache.hpp"
#include "config.hpp"
#include "generate.hpp"
#include "model.hpp"
#include "paged.hpp"
#include "quant.hpp"
#include "sampling.hpp"
#include "tensor.hpp"

#ifdef NI_CUDA
#include "cuda/cuda_backend.hpp"
#endif

namespace py = pybind11;
using namespace ni;

namespace {

// ni::Tensor -> numpy array (copy). The Tensor is row-major contiguous float32, so
// the shape carries straight over and we memcpy the block into a fresh py::array.
// Must be called with the GIL held (it allocates a Python object).
py::array_t<float> tensor_to_numpy(const Tensor& t) {
    std::vector<py::ssize_t> shape(t.shape().begin(), t.shape().end());
    py::array_t<float> out(shape);
    std::memcpy(out.mutable_data(), t.data(),
                static_cast<size_t>(t.numel()) * sizeof(float));
    return out;
}

// "fp32"|"q8"|"q4"|"q4g" -> QuantMode, for the demo CLI / ergonomic callers.
QuantMode quant_mode_from_string(const std::string& s) {
    if (s == "fp32" || s == "none") return QuantMode::None;
    if (s == "q8") return QuantMode::Q8;
    if (s == "q4") return QuantMode::Q4;
    if (s == "q4g") return QuantMode::Q4G;
    if (s == "w8a8") return QuantMode::W8A8;
    throw std::invalid_argument("unknown quant mode '" + s + "' (fp32|q8|q4|q4g|w8a8)");
}

}  // namespace

PYBIND11_MODULE(nicpp, m) {
    m.doc() =
        "nanoinfer C++ core (F6 pybind11 binding): Model, KVCache, sampling — the "
        "Python-orchestration / C++-kernel pivot.";

    py::enum_<QuantMode>(m, "QuantMode", "Weight-only quantization mode.")
        .value("NONE", QuantMode::None, "fp32 (no quantization)")
        .value("Q8", QuantMode::Q8, "per-channel int8")
        .value("Q4", QuantMode::Q4, "per-channel int4")
        .value("Q4G", QuantMode::Q4G, "group-wise int4 (Q4_0-style, 32-blocks)")
        .value("W8A8", QuantMode::W8A8, "int8 weight × int8 activation (int8×int8→int32 — compute, not just memory)");

    m.def("quant_mode", &quant_mode_from_string, py::arg("name"),
          "Parse 'fp32'|'q8'|'q4'|'q4g' into a QuantMode.");

    // Model dimensions, read from config.txt. Returned by value (a tiny POD) so
    // there's no lifetime tie to the Model.
    py::class_<Config>(m, "Config")
        .def_readonly("vocab_size", &Config::vocab_size)
        .def_readonly("hidden_size", &Config::hidden_size)
        .def_readonly("intermediate_size", &Config::intermediate_size)
        .def_readonly("num_layers", &Config::num_layers)
        .def_readonly("num_attention_heads", &Config::num_attention_heads)
        .def_readonly("num_kv_heads", &Config::num_kv_heads)
        .def_readonly("head_dim", &Config::head_dim)
        .def_readonly("max_position_embeddings", &Config::max_position_embeddings)
        .def_readonly("rms_norm_eps", &Config::rms_norm_eps)
        .def_readonly("rope_theta", &Config::rope_theta)
        .def_readonly("tie_word_embeddings", &Config::tie_word_embeddings)
        .def_readonly("eos_token_id", &Config::eos_token_id)
        .def("n_rep", &Config::n_rep, "Query heads per KV head (GQA).");

    py::class_<SamplingParams>(m, "SamplingParams")
        .def(py::init([](float temperature, int64_t top_k, float top_p,
                         float repetition_penalty) {
                 SamplingParams p;
                 p.temperature = temperature;
                 p.top_k = top_k;
                 p.top_p = top_p;
                 p.repetition_penalty = repetition_penalty;
                 return p;
             }),
             py::arg("temperature") = 0.0f, py::arg("top_k") = 0,
             py::arg("top_p") = 1.0f, py::arg("repetition_penalty") = 1.0f,
             "Defaults are greedy (temperature 0). Warpers compose in HF order.")
        .def_readwrite("temperature", &SamplingParams::temperature)
        .def_readwrite("top_k", &SamplingParams::top_k)
        .def_readwrite("top_p", &SamplingParams::top_p)
        .def_readwrite("repetition_penalty", &SamplingParams::repetition_penalty)
        .def("greedy", &SamplingParams::greedy)
        .def("__repr__", [](const SamplingParams& p) {
            return "SamplingParams(temperature=" + std::to_string(p.temperature) +
                   ", top_k=" + std::to_string(p.top_k) +
                   ", top_p=" + std::to_string(p.top_p) +
                   ", repetition_penalty=" + std::to_string(p.repetition_penalty) + ")";
        });

    // KV cache interface: forward()/forward_batch() accept any subclass, so the same
    // pass runs over the contiguous (C3) or paged (F8b) cache. Abstract — no init.
    py::class_<KVCacheBase>(m, "KVCacheBase")
        .def_property_readonly("length", &KVCacheBase::length, "Filled positions so far.")
        .def("truncate", &KVCacheBase::truncate, py::arg("length"),
             "Roll back to `length` filled positions (speculative-decode rollback): drop the "
             "K/V for positions >= length. Implemented by every cache (S1): contiguous moves the "
             "length pointer, paged frees the rejected tail blocks.");

    // Contiguous per-sequence KV cache (C3). Created by Model.make_cache, fed back
    // into forward(); move-only in C++, so no copy/constructor is exposed.
    py::class_<KVCache, KVCacheBase>(m, "KVCache")
        .def_property_readonly("max_seq", &KVCache::max_seq);

    // Paged KV cache (F8b): a shared pool of fixed-size blocks + a per-sequence
    // block table. PagedKVCache draws/returns blocks from the pool (it frees them on
    // destruction), so there is no per-sequence max_seq preallocation.
    py::class_<BlockPool>(m, "BlockPool")
        .def_property_readonly("num_blocks", &BlockPool::num_blocks)
        .def_property_readonly("block_size", &BlockPool::block_size)
        .def_property_readonly("free_blocks", &BlockPool::free_blocks)
        .def_property_readonly("used_blocks", &BlockPool::used_blocks)
        // Refcount primitives for the Python prefix cache (RadixAttention): hold a
        // shared block with incref, release it with free.
        .def("incref", &BlockPool::incref, py::arg("block"))
        .def("free", &BlockPool::free, py::arg("block"))
        .def("refcount", &BlockPool::refcount, py::arg("block"))
        .def("make_cache", [](BlockPool& self) { return new PagedKVCache(&self); },
             py::return_value_policy::take_ownership, py::keep_alive<0, 1>(),
             "A fresh PagedKVCache drawing blocks from this pool (keeps the pool alive).");

    py::class_<PagedKVCache, KVCacheBase>(m, "PagedKVCache")
        .def(py::init<BlockPool*>(), py::arg("pool"), py::keep_alive<1, 2>(),
             "A sequence's view onto a BlockPool (keeps the pool alive).")
        .def_property_readonly("num_blocks", &PagedKVCache::num_blocks,
                               "Physical blocks this sequence currently holds.")
        .def_property_readonly("block_table", &PagedKVCache::block_table,
                               "Logical->physical block ids (for the prefix cache).")
        .def("share_prefix", &PagedKVCache::share_prefix, py::arg("blocks"),
             py::arg("length"),
             "Seed this (fresh) sequence with shared prefix blocks; length must be "
             "blocks * block_size. The sequence then prefills only its suffix.");

#ifdef NI_CUDA
    // GPU paged attention (G4b/G4d): device siblings of BlockPool / PagedKVCache, exposed
    // with the SAME Python surface so the scheduler drives either device transparently —
    // it calls pool.make_cache() and uses the KVCacheBase / share_prefix interface.
    py::class_<CudaBlockPool>(m, "CudaBlockPool")
        .def_property_readonly("num_blocks", &CudaBlockPool::num_blocks)
        .def_property_readonly("block_size", &CudaBlockPool::block_size)
        .def_property_readonly("free_blocks", &CudaBlockPool::free_blocks)
        .def_property_readonly("used_blocks", &CudaBlockPool::used_blocks)
        .def("incref", &CudaBlockPool::incref, py::arg("block"))
        .def("free", &CudaBlockPool::free, py::arg("block"))
        .def("refcount", &CudaBlockPool::refcount, py::arg("block"))
        .def("make_cache", [](CudaBlockPool& self) { return new CudaPagedKVCache(&self); },
             py::return_value_policy::take_ownership, py::keep_alive<0, 1>(),
             "A fresh CudaPagedKVCache drawing blocks from this pool.");

    py::class_<CudaPagedKVCache, KVCacheBase>(m, "CudaPagedKVCache")
        .def_property_readonly("num_blocks", &CudaPagedKVCache::num_blocks)
        .def_property_readonly("block_table", &CudaPagedKVCache::block_table)
        .def("share_prefix", &CudaPagedKVCache::share_prefix, py::arg("blocks"), py::arg("length"));
#endif

    py::class_<Model>(m, "Model")
        .def(py::init([](const std::string& weights_dir, QuantMode mode, const std::string& device) {
                 Device dev = Device::CPU;
                 if (device == "cuda") dev = Device::CUDA;
                 else if (device != "cpu")
                     throw std::invalid_argument("device must be 'cpu' or 'cuda'");
                 return std::make_unique<Model>(weights_dir, mode, dev);
             }),
             py::arg("weights_dir"), py::arg("mode") = QuantMode::None, py::arg("device") = "cpu",
             "Load config.txt + every <name>.bin (NIT0) from weights_dir. device='cuda' runs "
             "every op on the GPU (requires a CUDA build); 'cpu' is the default.")
        .def_property_readonly(
            "config", [](const Model& self) { return self.config(); },
            "Model dimensions (a copy of the Config).")
        .def(
            "forward",
            [](const Model& self, const std::vector<int64_t>& ids, KVCacheBase* cache) {
                Tensor logits;
                {
                    py::gil_scoped_release release;  // numpy build below needs the GIL
                    logits = self.forward(ids, cache);
                }
                return tensor_to_numpy(logits);
            },
            py::arg("ids"), py::arg("cache") = static_cast<KVCacheBase*>(nullptr),
            "Token ids -> logits [seq, vocab] (numpy). With a KVCache, `ids` is the "
            "new token(s), placed at positions cache.length..; the cache advances after.")
        .def(
            "forward_batch",
            [](const Model& self, const std::vector<int64_t>& tokens, py::list caches_list) {
                // One new token per sequence, each with its own cache. Cast the
                // KVCache handles to pointers here (GIL held) before releasing.
                std::vector<KVCacheBase*> caches;
                caches.reserve(caches_list.size());
                for (const py::handle h : caches_list) caches.push_back(h.cast<KVCacheBase*>());
                Tensor logits;
                {
                    py::gil_scoped_release release;
                    logits = self.forward_batch(tokens, caches);
                }
                return tensor_to_numpy(logits);
            },
            py::arg("tokens"), py::arg("caches"),
            "Batched single-token decode (F8a): N new tokens + their N KVCaches -> "
            "logits [N, vocab]. Row s equals forward([tokens[s]], caches[s]); each "
            "cache advances by 1. The projection GEMMs are fused over the N rows.")
        .def("make_cache",
             [](const Model& self, int64_t max_seq) { return self.make_kv_cache(max_seq); },
             py::arg("max_seq"),
             "Allocate a KV cache for this model on its device — the contiguous CPU cache, "
             "or the device-resident cache for a CUDA model. Returned via KVCacheBase.")
        .def(
            "make_block_pool",
            [](const Model& self, int64_t block_size, int64_t num_blocks) -> py::object {
                const Config& c = self.config();
#ifdef NI_CUDA
                if (self.device() == Device::CUDA)
                    return py::cast(CudaBlockPool(c.num_layers, c.num_kv_heads, c.head_dim,
                                                  block_size, num_blocks));
#endif
                return py::cast(BlockPool(c.num_layers, c.num_kv_heads, c.head_dim, block_size,
                                          num_blocks));
            },
            py::arg("block_size"), py::arg("num_blocks"),
            "Allocate a paged-attention block pool on the model's device (BlockPool on CPU, "
            "CudaBlockPool on a CUDA model). Sequences draw caches via pool.make_cache().")
        .def(
            "generate",
            [](const Model& self, const std::vector<int64_t>& prompt, int max_tokens,
               const SamplingParams& params, uint64_t seed, int64_t eos_id) {
                GenerateConfig gc;
                gc.max_tokens = max_tokens;
                gc.params = params;
                gc.seed = seed;
                gc.eos_id = eos_id;
                std::vector<int64_t> out;
                {
                    py::gil_scoped_release release;
                    out = ni::generate(self, prompt, gc);
                }
                return out;
            },
            py::arg("prompt"), py::arg("max_tokens") = 20,
            py::arg("params") = SamplingParams(), py::arg("seed") = 0,
            py::arg("eos_id") = -1,
            "Continue prompt ids; returns generated ids (excluding the prompt). "
            "Uses a KV cache internally. Default params are greedy.")
        .def("weight_bytes", &Model::weight_bytes,
             "(actual_bytes, fp32_bytes) — the quantization-savings report.");
}
