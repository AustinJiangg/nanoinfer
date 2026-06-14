#include "config.hpp"

#include <fstream>
#include <stdexcept>

namespace ni {

Config load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("load_config: cannot open " + path);

    Config c;
    std::string key;
    while (f >> key) {
        if (key == "vocab_size") f >> c.vocab_size;
        else if (key == "hidden_size") f >> c.hidden_size;
        else if (key == "intermediate_size") f >> c.intermediate_size;
        else if (key == "num_layers") f >> c.num_layers;
        else if (key == "num_attention_heads") f >> c.num_attention_heads;
        else if (key == "num_kv_heads") f >> c.num_kv_heads;
        else if (key == "head_dim") f >> c.head_dim;
        else if (key == "max_position_embeddings") f >> c.max_position_embeddings;
        else if (key == "rms_norm_eps") f >> c.rms_norm_eps;
        else if (key == "rope_theta") f >> c.rope_theta;
        else { std::string ignored; f >> ignored; }  // unknown key: skip its value
    }
    return c;
}

}  // namespace ni
