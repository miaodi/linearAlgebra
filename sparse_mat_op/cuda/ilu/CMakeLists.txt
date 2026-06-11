target_sources(sparse_mat_op_cuda PRIVATE
    ilu_numeric.cu
    ilu_numeric_cta_granular.cu
    ilu_numeric_persistent.cu
    ilu_update_cache.cpp
    ilu_update_cache_device.cu
    ilu_numeric_common.cuh
    ilu_numeric.cuh
    ilu_numeric_cta_granular.cuh
    ilu_numeric_persistent.cuh
    ilu_update_cache.hpp
)
