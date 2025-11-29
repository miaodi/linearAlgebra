include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

# Internal helper to probe a single SIMD flag for compiler and CPU support.
function(check_simd_support FLAG INSTRUCTION_SET COMPILE_VAR RUNTIME_VAR TEST_CODE)
  check_cxx_compiler_flag("${FLAG}" ${COMPILE_VAR})

  if(${COMPILE_VAR})
    if(NOT CMAKE_CROSSCOMPILING)
      set(CMAKE_REQUIRED_FLAGS "${FLAG}")
      check_cxx_source_runs("${TEST_CODE}" ${RUNTIME_VAR})
      set(CMAKE_REQUIRED_FLAGS "")
    else()
      # Assume target CPU matches the build host when cross compiling.
      set(${RUNTIME_VAR} TRUE)
    endif()

    if(${RUNTIME_VAR})
      message(STATUS "${INSTRUCTION_SET} enabled (${FLAG})")
    else()
      message(STATUS "${INSTRUCTION_SET} supported by compiler but not CPU; flag will not be used")
    endif()
  else()
    message(STATUS "${INSTRUCTION_SET} not supported by compiler")
    set(${RUNTIME_VAR} FALSE)
  endif()

  set(${COMPILE_VAR} ${${COMPILE_VAR}} PARENT_SCOPE)
  set(${RUNTIME_VAR} ${${RUNTIME_VAR}} PARENT_SCOPE)
endfunction()

# Probe FMA
set(SIMD_FLAG_FMA "-mfma")
check_simd_support(
  "${SIMD_FLAG_FMA}" "FMA" COMPILER_SUPPORTS_FMA FMA_CPU_SUPPORTS
  "
  #include <immintrin.h>
  int main() {
    __m128d a = _mm_set1_pd(1.0);
    __m128d b = _mm_set1_pd(2.0);
    __m128d c = _mm_fmadd_pd(a, b, a);
    (void)c;
    return 0;
  }
  "
)

# Probe AVX2
set(SIMD_FLAG_AVX2 "-mavx2")
check_simd_support(
  "${SIMD_FLAG_AVX2}" "AVX2" COMPILER_SUPPORTS_AVX2 AVX2_CPU_SUPPORTS
  "
  #include <immintrin.h>
  int main() {
    __m256 a = _mm256_set1_ps(1.0f);
    (void)a;
    return 0;
  }
  "
)

# Probe AVX-512F
set(SIMD_FLAG_AVX512 "-mavx512f")
check_simd_support(
  "${SIMD_FLAG_AVX512}" "AVX512F" COMPILER_SUPPORTS_AVX512F AVX512_CPU_SUPPORTS
  "
  #include <immintrin.h>
  int main() {
    __m512 a = _mm512_set1_ps(1.0f);
    (void)a;
    return 0;
  }
  "
)

# Attach the supported SIMD flags to a target.
# Usage: target_enable_simd(my_target [SCOPE <PRIVATE|PUBLIC|INTERFACE>])
function(target_enable_simd TARGET_NAME)
  set(options)
  set(oneValueArgs SCOPE)
  set(multiValueArgs)
  cmake_parse_arguments(SIMD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT SIMD_SCOPE)
    set(SIMD_SCOPE PRIVATE)
  endif()

  if(FMA_CPU_SUPPORTS)
    target_compile_options(${TARGET_NAME} ${SIMD_SCOPE} ${SIMD_FLAG_FMA})
    target_compile_definitions(${TARGET_NAME} ${SIMD_SCOPE} FMA_SUPPORTED)
  endif()

  if(AVX2_CPU_SUPPORTS)
    target_compile_options(${TARGET_NAME} ${SIMD_SCOPE} ${SIMD_FLAG_AVX2})
    target_compile_definitions(${TARGET_NAME} ${SIMD_SCOPE} AVX2_SUPPORTED)
  endif()

  if(AVX512_CPU_SUPPORTS)
    target_compile_options(${TARGET_NAME} ${SIMD_SCOPE} ${SIMD_FLAG_AVX512})
    target_compile_definitions(${TARGET_NAME} ${SIMD_SCOPE} AVX512_SUPPORTED)
  endif()
endfunction()
