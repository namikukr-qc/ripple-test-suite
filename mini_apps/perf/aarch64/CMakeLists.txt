enable_language(C ASM)

include(FetchContent)
include(CheckIncludeFileCXX)

set(KLEIDIAI_GIT_TAG "v1.31.0")

FetchContent_Declare(
  kleidiai
  GIT_REPOSITORY https://gitlab.arm.com/kleidi/kleidiai.git
  GIT_TAG        ${KLEIDIAI_GIT_TAG}
)

FetchContent_GetProperties(kleidiai)
if(NOT kleidiai_POPULATED)
  FetchContent_Populate(kleidiai)
endif()

# Ripple SME/SVE compiler flags.
set(RIPPLE_SME_SVE_FLAGS
  -fenable-ripple
  -march=armv9.2-a+sve2+sme
  -msve-streaming-vector-bits=512
  -O3
)

add_test_app(
  aarch64_perf_tests
  ../../../main.sample.cpp
)
target_compile_options(aarch64_perf_tests PRIVATE ${RIPPLE_SME_SVE_FLAGS})
target_include_directories(aarch64_perf_tests PRIVATE ${RIPPLE_TEST_SUITE_INCLUDE_DIR})

# Only add GEMM tests if ripple_sme_veclib.h is available

# KleidiAI source files
set(KLEIDIAI_SOURCES
  ${kleidiai_SOURCE_DIR}/kai/kai_common_sme_asm.S
  ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.c
  ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme_asm.S
  ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.c
  ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla_asm.S
)

# Ripple source files
set(RIPPLE_GEMV_SOURCES
  vecmat.cpp
)

target_sources(aarch64_perf_tests PRIVATE
  ${KLEIDIAI_SOURCES}
  ${RIPPLE_GEMV_SOURCES}
)
target_include_directories(aarch64_perf_tests PRIVATE ${kleidiai_SOURCE_DIR})
