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
#include "quant.hpp"
#include "sampling.hpp"
#include "tensor.hpp"

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
    throw std::invalid_argument("unknown quant mode '" + s + "' (fp32|q8|q4|q4g)");
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
        .value("Q4G", QuantMode::Q4G, "group-wise int4 (Q4_0-style, 32-blocks)");

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

    // Opaque per-sequence KV cache. Created by Model.make_cache and fed back into
    // forward(); move-only in C++, so no copy/constructor is exposed.
    py::class_<KVCache>(m, "KVCache")
        .def_property_readonly("length", &KVCache::length, "Filled positions so far.")
        .def_property_readonly("max_seq", &KVCache::max_seq);

    py::class_<Model>(m, "Model")
        .def(py::init<const std::string&, QuantMode>(), py::arg("weights_dir"),
             py::arg("mode") = QuantMode::None,
             "Load config.txt + every <name>.bin (NIT0) from weights_dir.")
        .def_property_readonly(
            "config", [](const Model& self) { return self.config(); },
            "Model dimensions (a copy of the Config).")
        .def(
            "forward",
            [](const Model& self, const std::vector<int64_t>& ids, KVCache* cache) {
                Tensor logits;
                {
                    py::gil_scoped_release release;  // numpy build below needs the GIL
                    logits = self.forward(ids, cache);
                }
                return tensor_to_numpy(logits);
            },
            py::arg("ids"), py::arg("cache") = static_cast<KVCache*>(nullptr),
            "Token ids -> logits [seq, vocab] (numpy). With a KVCache, `ids` is the "
            "new token(s), placed at positions cache.length..; the cache advances after.")
        .def(
            "forward_batch",
            [](const Model& self, const std::vector<int64_t>& tokens, py::list caches_list) {
                // One new token per sequence, each with its own cache. Cast the
                // KVCache handles to pointers here (GIL held) before releasing.
                std::vector<KVCache*> caches;
                caches.reserve(caches_list.size());
                for (const py::handle h : caches_list) caches.push_back(h.cast<KVCache*>());
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
        .def("make_cache", &Model::make_cache, py::arg("max_seq"),
             "Allocate a KV cache sized for this model.")
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
