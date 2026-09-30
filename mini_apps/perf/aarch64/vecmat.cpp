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

static void fp32_pack_rhs(float *packed, float *b, float *bias, unsigned n_dim, unsigned k_dim, unsigned block_dim) {
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

// Weights b are K x N non-transposed (k outer, n inner).
static void fp32_vecmat_ref(float *a, float *b, float *c, unsigned n, unsigned k, float *bias,
                       float clamp_min, float clamp_max) {
    for (unsigned n_idx = 0; n_idx < n; ++n_idx) {
        c[n_idx] = bias[n_idx];
        for (unsigned k_idx = 0; k_idx < k; ++k_idx)
            c[n_idx] += a[k_idx] * b[k_idx * n + n_idx];
        c[n_idx] = std::max(clamp_min, std::min(clamp_max, c[n_idx]));
    }
}

#ifdef __ARM_FEATURE_SME
__arm_locally_streaming
#endif
void fp32_vecmat_ripple(float *a, float *b, float *c, unsigned n, unsigned k,
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

template <KernelT KT, unsigned N, unsigned K> class VecmatTest : public Test {
    static constexpr unsigned N_tiles = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    float CLAMP_MIN = randn() * 65504.0 - 65504.0;
    float CLAMP_MAX = randn() * 65504.0;
    float A[K], B[K * N], C[N], Ref[N], Bias[N];
    // Packed array has N_tiles, each tile has K+1 rows * BLOCK_DIM elts
    float B_packed[N_tiles * (K + 1) * BLOCK_DIM]{};

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
        if (KT != KernelT::Reference)
            fp32_pack_rhs(B_packed, B, Bias, N, K, BLOCK_DIM);

        fp32_vecmat_ref(A, B, Ref, N, K, Bias, CLAMP_MIN, CLAMP_MAX);
    }

    void run(unsigned) override {
        if (KT == KernelT::Reference)
            fp32_vecmat_ref(A, B, C, N, K, Bias, CLAMP_MIN, CLAMP_MAX);
        if (KT == KernelT::RippleOpt)
            fp32_vecmat_ripple(A, B_packed, C, N, K, CLAMP_MIN, CLAMP_MAX);
    }
    bool verify() const override {
        return equal(1e-5, C, Ref); 
    }

};

DefineTest<VecmatTest<RippleOpt, 1000, 1000>> VecmatTestInstance_1("fp32_vecmat_1000x1000.ripple");
DefineTest<VecmatTest<RippleOpt, 900, 1100>> VecmatTestInstance_2("fp32_vecmat_900x1100.ripple");
DefineTest<VecmatTest<RippleOpt, 1100, 900>> VecmatTestInstance_3("fp32_vecmat_1100x900.ripple");
DefineTest<VecmatTest<RippleOpt, 20, 20>> VecmatTestInstance_4("fp32_vecmat_20x20.ripple");
DefineTest<VecmatTest<RippleOpt, 10, 30>> VecmatTestInstance_5("fp32_vecmat_10x30.ripple");
DefineTest<VecmatTest<RippleOpt, 30, 10>> VecmatTestInstance_6("fp32_vecmat_30x10.ripple");
DefineTest<VecmatTest<RippleOpt, 512, 1>> VecmatTestInstance_7("fp32_vecmat_512x1.ripple");
DefineTest<VecmatTest<RippleOpt, 1, 512>> VecmatTestInstance_8("fp32_vecmat_1x512.ripple");

} // namespace ripple_test_suite
