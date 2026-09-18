add_test_app(
  aarch64_perf_tests
  vecmat.cpp
  ../../../main.sample.cpp
)

target_compile_options(aarch64_perf_tests PRIVATE -fenable-ripple)
target_include_directories(aarch64_perf_tests PRIVATE ${RIPPLE_TEST_SUITE_INCLUDE_DIR})
# TODO for now, we unconditionally use SVE, but potentially change in the future
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(aarch64_perf_tests PRIVATE -march=armv8-a+sve2+sme -O3)
endif()
