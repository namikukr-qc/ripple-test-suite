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

# Check if ripple_sme_veclib.h is available.
list(JOIN RIPPLE_SME_SVE_FLAGS " " RIPPLE_SME_SVE_FLAGS_STRING)
set(CMAKE_REQUIRED_FLAGS "${RIPPLE_SME_SVE_FLAGS_STRING}")
check_include_file_cxx("ripple_sme_veclib.h" HAVE_RIPPLE_SME_VECLIB)
unset(CMAKE_REQUIRED_FLAGS)

add_test_app(
  aarch64_perf_tests
  ../../../main.sample.cpp
)
target_compile_options(aarch64_perf_tests PRIVATE ${RIPPLE_SME_SVE_FLAGS})
target_include_directories(aarch64_perf_tests PRIVATE ${RIPPLE_TEST_SUITE_INCLUDE_DIR})

# Only add GEMM tests if ripple_sme_veclib.h is available
if(HAVE_RIPPLE_SME_VECLIB)
  message(STATUS "ripple_sme_veclib.h found - enabling SME tests")

  # KleidiAI source files (f32 GEMM)
  set(KLEIDIAI_F32_SOURCES
    ${kleidiai_SOURCE_DIR}/kai/kai_common_sme_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme_asm.S
  )

  # KleidiAI source files (quantized int8 GEMM)
  set(KLEIDIAI_I8_SOURCES
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme_asm.S
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.c
    ${kleidiai_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme_asm.S
  )

  # Ripple source files (f32 GEMM)
  set(RIPPLE_GEMM_SOURCES
    ripple_matmul_f32_f32p_f32p_sme_mopa.cpp
    gemm_f32_test.cpp
  )

  # Ripple source files (quantized int8 GEMM)
  set(RIPPLE_GEMM_I8_SOURCES
    ripple_matmul_i8_i8p_i8p_sme_mopa.cpp
    gemm_i8_test.cpp
  )

  target_sources(aarch64_perf_tests PRIVATE
    ${KLEIDIAI_F32_SOURCES}
    ${KLEIDIAI_I8_SOURCES}
    ${RIPPLE_GEMM_SOURCES}
    ${RIPPLE_GEMM_I8_SOURCES}
  )
  target_include_directories(aarch64_perf_tests PRIVATE ${kleidiai_SOURCE_DIR})
else()
  message(STATUS "ripple_sme_veclib.h not found - skipping SME tests")
endif()
