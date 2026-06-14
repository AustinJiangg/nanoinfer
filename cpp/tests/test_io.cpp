// Error-path tests for the config parser and tensor IO guards (added after a
// review flagged silent failure on malformed config / corrupt .bin files).
#include <cstdio>
#include <fstream>
#include <string>

#include "config.hpp"
#include "serialize.hpp"
#include "tensor.hpp"
#include "test_util.hpp"

static void write_file(const std::string& path, const std::string& contents) {
    std::ofstream f(path);
    f << contents;
}

template <typename Fn>
static bool throws(Fn fn) {
    try {
        fn();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

int main() {
    const std::string ok = "io_ok.txt", bad = "io_bad.txt", nd = "io_nd.txt", bin = "io_t.bin";

    // A complete config parses, and n_rep() is correct.
    write_file(ok,
               "vocab_size 100\nhidden_size 8\nintermediate_size 16\nnum_layers 2\n"
               "num_attention_heads 4\nnum_kv_heads 2\nhead_dim 2\n"
               "max_position_embeddings 64\nrms_norm_eps 1e-6\nrope_theta 10000\n"
               "tie_word_embeddings 1\n");
    ni::Config c = ni::load_config(ok);
    CHECK(c.num_layers == 2);
    CHECK(c.n_rep() == 2);
    CHECK(c.tie_word_embeddings == 1);

    // A config missing required keys is rejected, not silently zero-filled.
    write_file(bad, "vocab_size 100\nhidden_size 8\n");
    CHECK(throws([&] { ni::load_config(bad); }));

    // num_attention_heads not divisible by num_kv_heads is rejected.
    write_file(nd,
               "vocab_size 100\nhidden_size 8\nintermediate_size 16\nnum_layers 1\n"
               "num_attention_heads 5\nnum_kv_heads 2\nhead_dim 2\n");
    CHECK(throws([&] { ni::load_config(nd); }));

    // load_bin round-trips a tensor...
    ni::Tensor t({2, 3});
    for (int64_t i = 0; i < t.numel(); ++i) t[i] = float(i);
    ni::save_bin(bin, t);
    ni::Tensor r = ni::load_bin(bin);
    CHECK(r.numel() == 6);
    CHECK_CLOSE(r[5], 5.0, 1e-9);

    // ...and rejects a file with trailing bytes (corrupt header / wrong shape).
    {
        std::ofstream f(bin, std::ios::binary | std::ios::app);
        char z = 0;
        f.write(&z, 1);
    }
    CHECK(throws([&] { ni::load_bin(bin); }));

    for (const auto& p : {ok, bad, nd, bin}) std::remove(p.c_str());
    std::printf(g_failures ? "test_io: %d failures\n" : "test_io: ok\n", g_failures);
    return g_failures ? 1 : 0;
}
