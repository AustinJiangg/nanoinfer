#include "serialize.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ni {

Tensor load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_bin: cannot open " + path);

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "NIT0", 4) != 0)
        throw std::runtime_error("load_bin: bad magic in " + path);

    int32_t ndim = 0;
    f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    if (ndim < 0) throw std::runtime_error("load_bin: negative ndim in " + path);

    std::vector<int64_t> shape(static_cast<size_t>(ndim));
    for (int32_t i = 0; i < ndim; ++i) {
        int32_t d = 0;
        f.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (d < 0) throw std::runtime_error("load_bin: negative shape dim in " + path);
        shape[i] = d;
    }

    Tensor t(shape);
    const std::streamsize need = t.numel() * static_cast<std::streamsize>(sizeof(float));
    f.read(reinterpret_cast<char*>(t.data()), need);
    if (!f || f.gcount() != need) throw std::runtime_error("load_bin: short read in " + path);
    // Reject a file longer than its declared shape — a truncated/corrupt header
    // (or a wrong, smaller shape) would otherwise read silently as wrong data.
    if (f.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("load_bin: trailing bytes (shape/data mismatch) in " + path);
    return t;
}

void save_bin(const std::string& path, const Tensor& t) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_bin: cannot open " + path);

    f.write("NIT0", 4);
    const int32_t ndim = static_cast<int32_t>(t.ndim());
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int64_t d : t.shape()) {
        const int32_t d32 = static_cast<int32_t>(d);
        f.write(reinterpret_cast<const char*>(&d32), sizeof(d32));
    }
    f.write(reinterpret_cast<const char*>(t.data()), t.numel() * static_cast<int64_t>(sizeof(float)));
}

}  // namespace ni
