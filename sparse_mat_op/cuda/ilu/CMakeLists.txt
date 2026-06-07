target_sources(sparse_mat_op_cuda PRIVATE
    ilu_numeric.cu
    ilu_numeric_workqueue.cu
    ilu_update_cache.cpp
    ilu_numeric_common.cuh
    ilu_numeric.cuh
    ilu_numeric_workqueue.cuh
    ilu_update_cache.hpp
)
