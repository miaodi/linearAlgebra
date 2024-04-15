include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG origin/main
        GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(googletest)

set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "Turn off tests")

FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG origin/main
        GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(googlebenchmark)