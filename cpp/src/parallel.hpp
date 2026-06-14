// Multithreading knobs for stage C5.
//
// The kernels parallelize over OUTPUT CHANNELS (each output feature, or each
// attention head, is computed in full by exactly one thread). That split is
// special: a thread owns a whole reduction, so the per-output result is
// bit-identical to the serial code — threading buys wall-clock with zero effect
// on the logits, and the existing parity tests keep guarding correctness.
//
// OpenMP is optional: with _OPENMP the kernels add `#pragma omp parallel for`;
// without it they run serially and still pass. This header just centralizes the
// "is it on / how many threads / is a loop big enough to be worth a team" bits.
#pragma once

#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ni {

// Threads the kernels may use (respects OMP_NUM_THREADS); 1 without OpenMP.
inline int max_threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline const char* threading_backend() {
#if defined(_OPENMP)
    return "openmp";
#else
    return "serial";
#endif
}

// Spinning up a team costs a few microseconds, so only fan out when there are
// enough output channels to amortize it; below this a projection runs serially.
// (k/v projections on Qwen2.5-0.5B have 128 out-features, comfortably above.)
constexpr int64_t kParallelMinRows = 64;

}  // namespace ni
