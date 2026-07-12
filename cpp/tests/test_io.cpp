// Error-path tests for the config parser and tensor IO guards (added after a
// review flagged silent failure on malformed config / corrupt .bin files).
#include <cstdint>
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

    // NIT1 (B1): a bf16 payload inflates to fp32 exactly (bits << 16). Hand-write the file —
    // C++ save_bin deliberately only writes NIT0; the bf16 writer is Python-side (ni/nit0.py).
    const std::string bin1 = "io_t1.bin";
    {
        std::ofstream f(bin1, std::ios::binary);
        f.write("NIT1", 4);
        const int32_t dtype = 1, ndim = 1, dim = 3;
        f.write(reinterpret_cast<const char*>(&dtype), 4);
        f.write(reinterpret_cast<const char*>(&ndim), 4);
        f.write(reinterpret_cast<const char*>(&dim), 4);
        // bf16 bit patterns: 1.0 = 0x3F80, -2.5 = 0xC020, 0.0 = 0x0000 (fp32's top halves).
        const uint16_t payload[3] = {0x3F80, 0xC020, 0x0000};
        f.write(reinterpret_cast<const char*>(payload), sizeof(payload));
    }
    ni::Tensor rb = ni::load_bin(bin1);
    CHECK(rb.numel() == 3);
    CHECK(rb[0] == 1.0f && rb[1] == -2.5f && rb[2] == 0.0f);  // exact, not approximate

    // A NIT1 header with an unknown dtype is rejected, not misread as shape.
    const std::string bin2 = "io_t2.bin";
    {
        std::ofstream f(bin2, std::ios::binary);
        f.write("NIT1", 4);
        const int32_t dtype = 7, ndim = 1, dim = 1;
        f.write(reinterpret_cast<const char*>(&dtype), 4);
        f.write(reinterpret_cast<const char*>(&ndim), 4);
        f.write(reinterpret_cast<const char*>(&dim), 4);
        const float v = 1.0f;
        f.write(reinterpret_cast<const char*>(&v), 4);
    }
    CHECK(throws([&] { ni::load_bin(bin2); }));

    // A truncated bf16 payload is a short read, not silent garbage.
    const std::string bin3 = "io_t3.bin";
    {
        std::ofstream f(bin3, std::ios::binary);
        f.write("NIT1", 4);
        const int32_t dtype = 1, ndim = 1, dim = 4;
        f.write(reinterpret_cast<const char*>(&dtype), 4);
        f.write(reinterpret_cast<const char*>(&ndim), 4);
        f.write(reinterpret_cast<const char*>(&dim), 4);
        const uint16_t payload[2] = {0x3F80, 0x3F80};  // 2 of the declared 4 elements
        f.write(reinterpret_cast<const char*>(payload), sizeof(payload));
    }
    CHECK(throws([&] { ni::load_bin(bin3); }));

    for (const auto& p : {ok, bad, nd, bin, bin1, bin2, bin3}) std::remove(p.c_str());
    std::printf(g_failures ? "test_io: %d failures\n" : "test_io: ok\n", g_failures);
    return g_failures ? 1 : 0;
}
