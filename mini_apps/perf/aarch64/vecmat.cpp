//===----------------------------------------------------------------------===//
//
// (c) 2026 Qualcomm Technologies, Inc. All rights reserved.
//
// See https://spdx.org/licenses/BSD-3-Clause-Clear.html for license information.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstddef>
#include <ripple.h>

#include <ripple-test-suite/ripple-test-suite.h>

#define TILE_SIZE 16
#define BLOCK_DIM TILE_SIZE*2
#define VEC 0

namespace {

static void pack_rhs(float *packed, float *b, float *bias, unsigned n_dim, unsigned k_dim, unsigned block_dim) {
    const unsigned n_tiles = (n_dim + block_dim - 1) / block_dim;
    for (unsigned t = 0; t < n_tiles; ++t) {
        unsigned n_start = t * block_dim;
        for (unsigned ni = 0; ni < block_dim; ++ni) {
            unsigned n_idx = n_start + ni;
            if (n_idx < n_dim)
                packed[t * (k_dim + 1) * block_dim + ni] = bias[n_idx];
        }
        for (unsigned ki = 0; ki < k_dim; ++ki) {
            for (unsigned ni = 0; ni < block_dim; ++ni) {
                unsigned n_idx = n_start + ni;
                if (n_idx < n_dim)
                    packed[(t * (k_dim + 1) + 1 + ki) * block_dim + ni] = b[ki * n_dim + n_idx];
            }
        }
    }
}

} // namespace

namespace {

// Weights b are K x N non-transposed (k outer, n inner).
static void vecmat_ref(float *a, float *b, float *c, unsigned n, unsigned k, float *bias,
                       float clamp_min, float clamp_max) {
    for (unsigned n_idx = 0; n_idx < n; ++n_idx) {
        c[n_idx] = bias[n_idx];
        for (unsigned k_idx = 0; k_idx < k; ++k_idx)
            c[n_idx] += a[k_idx] * b[k_idx * n + n_idx];
        c[n_idx] = std::max(clamp_min, std::min(clamp_max, c[n_idx]));
    }
}

} // namespace

namespace {

#ifdef __ARM_FEATURE_SME
__arm_locally_streaming
#endif
void vecmat_ripple(float *a, float *b, float *c, unsigned n, unsigned k,
                   float clamp_min, float clamp_max) {
    const size_t block_dim = TILE_SIZE * 2;
    ripple_block_t B = ripple_set_block_shape(VEC, block_dim);
    float *tile = b;
    size_t stride = block_dim * k;
    // We tile across n, chunking it by block_dim.
    __builtin_prefetch(a, 0, /*locality*/ 2);
    for (size_t n_idx = 0; n_idx < n; n_idx += block_dim) {
        size_t n_to_process = std::min(block_dim, n - n_idx); // could have an incomplete block.
        ripple_parallel(B, 0);
        for (size_t ti = 0; ti < n_to_process; ++ti) {
            // load bias
            float tmp = __builtin_nontemporal_load(tile + ti);
            tile += block_dim;
            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                tmp += a[k_idx] * tile[k_idx * block_dim + ti];
            }
            // clamp and store
            tmp = std::max(clamp_min, std::min(clamp_max, tmp));
            __builtin_nontemporal_store(tmp, c + n_idx + ti);
        }
        tile += stride;
        __builtin_prefetch(a, 0, /*locality*/ 2);
    }
}

} // namespace



namespace ripple_test_suite {

enum KernelT {
  Reference,
  RippleOpt,
};

template <KernelT KT> class VecmatTest : public Test {
    static constexpr unsigned N = 1000, K = 1000;
    static constexpr unsigned N_tiles = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    static constexpr float CLAMP_MIN = -65504.0;
    static constexpr float CLAMP_MAX = 65504.0;
    float A[K], B[K * N], C[N], Ref[N], Bias[K];
    float B_packed[N_tiles * (K - 1) * BLOCK_DIM]{};

public:
    VecmatTest(TestFramework &TestFramework) : Test(TestFramework) {
        for (unsigned i = 0; i < K; ++i) {
            A[i] = -10 + randn() * 20;
        }
        for (unsigned i = 0; i < N * K; ++i) {
            B[i] = -10 + randn() * 20;
        }
        for (unsigned i = 0; i < K; ++i) {
            Bias[i] = -1 + randn() * 2;
        }
        pack_rhs(B_packed, B, Bias, N, K, BLOCK_DIM);
        vecmat_ref(A, B, Ref, N, K, Bias, CLAMP_MIN, CLAMP_MAX);
    }

    void run(unsigned) override {
        if (KT == KernelT::Reference)
            vecmat_ref(A, B, C, N, K, Bias, CLAMP_MIN, CLAMP_MAX);
        if (KT == KernelT::RippleOpt)
            vecmat_ripple(A, B_packed, C, N, K, CLAMP_MIN, CLAMP_MAX);
    }
    bool verify() const override {
        return equal(1e-5, C, Ref); 
    }

};

DefineTest<VecmatTest<Reference>> VecmatTestInstance_0("vecmat.ref");
DefineTest<VecmatTest<RippleOpt>> VecmatTestInstance_1("vecmat.ripple");

} // namespace ripple_test_suite
