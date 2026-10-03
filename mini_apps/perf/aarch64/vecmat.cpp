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

#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.h"

#define TILE_SIZE 16
#define BLOCK_DIM TILE_SIZE*2
#define VEC 0
#define UNUSED 0

namespace {
// Weights b are K x N non-transposed (k outer, n inner).
static void vecmat_f32_f32_f32_ref(float *a, float *b, float *c, unsigned n, unsigned k, float *bias,
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
void ripple_vecmat_f32_f32_f32p_ssve_fmla(float *a, float *b, float *c, unsigned n, unsigned k,
                   float clamp_min, float clamp_max) {
    const size_t block_dim = TILE_SIZE * 2;
    ripple_block_t B = ripple_set_block_shape(VEC, block_dim);
    float *tile = b;
    size_t stride = block_dim * k;
    // We tile across n, chunking it by block_dim.
    __builtin_prefetch(a, 0, /*locality*/ 2);
    for (size_t n_idx = 0; n_idx < n; n_idx += block_dim) {
        // Could have an incomplete block.
        size_t n_to_process = std::min(block_dim, n - n_idx);
        ripple_parallel(B, 0);
        for (size_t ti = 0; ti < n_to_process; ++ti) {
            // load bias
            float acc = __builtin_nontemporal_load(tile + ti);
            tile += block_dim;
            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                acc += a[k_idx] * tile[k_idx * block_dim + ti];
            }
            // clamp and store
            acc = std::max(clamp_min, std::min(clamp_max, acc));
            __builtin_nontemporal_store(acc, c + n_idx + ti);
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
        const size_t nr =
            kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla();
        const size_t kr =
            kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla();
        const size_t sr =
            kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla();
        for (unsigned i = 0; i < K; ++i) {
            A[i] = -100 + randn() * 200;
        }
        for (unsigned i = 0; i < N * K; ++i) {
            B[i] = -100 + randn() * 200;
        }
        for (unsigned i = 0; i < N; ++i) {
            Bias[i] = -10 + randn() * 20;
        }
        kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
                1, N, K, nr, kr, sr, N * sizeof(float), B, Bias,
                nullptr, B_packed, 0, nullptr);

        vecmat_f32_f32_f32_ref(A, B, Ref, N, K, Bias, CLAMP_MIN, CLAMP_MAX);
    }

    void run(unsigned) override {
        if (KT == KernelT::Reference)
            kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla(
                    1, N, K, A, UNUSED, B_packed, C, UNUSED, UNUSED,
                    CLAMP_MIN, CLAMP_MAX);
        if (KT == KernelT::RippleOpt)
            ripple_vecmat_f32_f32_f32p_ssve_fmla(A, B_packed, C, N, K,
                    CLAMP_MIN, CLAMP_MAX);
    }
    bool verify() const override {
        return equal(1e-5, C, Ref); 
    }

};

DefineTest<VecmatTest<RippleOpt, 1000, 1000>> VecmatTestInstance_1("fp32_vecmat_1000x1000.ripple");
DefineTest<VecmatTest<Reference, 1000, 1000>> RefVecmatTestInstance_1("fp32_vecmat_1000x1000.reference");
DefineTest<VecmatTest<RippleOpt, 900, 1100>> VecmatTestInstance_2("fp32_vecmat_900x1100.ripple");
DefineTest<VecmatTest<Reference, 900, 1100>> RefVecmatTestInstance_2("fp32_vecmat_900x1100.reference");
DefineTest<VecmatTest<RippleOpt, 1100, 900>> VecmatTestInstance_3("fp32_vecmat_1100x900.ripple");
DefineTest<VecmatTest<Reference, 1100, 900>> RefVecmatTestInstance_3("fp32_vecmat_1100x900.reference");
DefineTest<VecmatTest<RippleOpt, 20, 20>> VecmatTestInstance_4("fp32_vecmat_20x20.ripple");
DefineTest<VecmatTest<Reference, 20, 20>> RefVecmatTestInstance_4("fp32_vecmat_20x20.reference");
DefineTest<VecmatTest<RippleOpt, 10, 30>> VecmatTestInstance_5("fp32_vecmat_10x30.ripple");
DefineTest<VecmatTest<Reference, 10, 30>> RefVecmatTestInstance_5("fp32_vecmat_10x30.reference");
DefineTest<VecmatTest<RippleOpt, 30, 10>> VecmatTestInstance_6("fp32_vecmat_30x10.ripple");
DefineTest<VecmatTest<Reference, 30, 10>> RefVecmatTestInstance_6("fp32_vecmat_30x10.reference");
DefineTest<VecmatTest<RippleOpt, 512, 1>> VecmatTestInstance_7("fp32_vecmat_512x1.ripple");
DefineTest<VecmatTest<Reference, 512, 1>> RefVecmatTestInstance_7("fp32_vecmat_512x1.reference");
DefineTest<VecmatTest<RippleOpt, 1, 512>> VecmatTestInstance_8("fp32_vecmat_1x512.ripple");
DefineTest<VecmatTest<Reference, 1, 512>> RefVecmatTestInstance_8("fp32_vecmat_1x512.reference");

} // namespace ripple_test_suite
