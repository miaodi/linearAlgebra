#pragma once

#include "../config.h"

#ifdef USE_CUDA
#include "cudss.h"
#include "mkl_solver.h"
#include "mkl_sparse_mat.h"
#include <cuda_runtime.h>
#include <memory>

namespace mkl_wrapper {

class cudss_solver : public mkl_solver {
public:
  cudss_solver(mkl_sparse_mat const *A);
  virtual bool solve(double const *const b, double *const x) override;
  virtual ~cudss_solver() override;
  bool factorize();

protected:
  cudaStream_t stream;
  cudssHandle_t handle;
  cudssConfig_t solverConfig;
  cudssData_t solverData;

  cudssMatrix_t cudaMat;
  cudssMatrix_t res, rhs;
  cudaError_t cuda_error = cudaSuccess;
  cudssStatus_t status = CUDSS_STATUS_SUCCESS;
  int64_t size;
  int *csr_offsets_d = NULL;
  int *csr_columns_d = NULL;
  double *csr_values_d = NULL;
  double *x_values_d = NULL, *b_values_d = NULL;
};
} // namespace mkl_wrapper

#endif