target_sources(sparse_mat_op_cuda PRIVATE
    ilu_numeric.cu
    ilu_numeric_level_cta.cu
    ilu_numeric_persistent.cu
    ilu_update_cache.cpp
    ilu_update_cache_device.cu
    ilu_numeric_common.cuh
    ilu_numeric.cuh
    ilu_numeric_level_cta.cuh
    ilu_numeric_persistent.cuh
    ilu_update_cache.hpp
)
