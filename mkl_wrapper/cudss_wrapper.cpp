#include "cudss_wrapper.h"

#ifdef USE_CUDA

#include <iostream>

namespace mkl_wrapper {

cudss_solver::cudss_solver(mkl_sparse_mat const *A) : mkl_solver() {

  size = A->rows();
  int64_t nnz = A->nnz();
  cudaMalloc(&csr_offsets_d, (size + 1) * sizeof(int));
  cudaMalloc(&csr_columns_d, nnz * sizeof(int));
  cudaMalloc(&csr_values_d, nnz * sizeof(double));
  cudaMalloc(&b_values_d, size * sizeof(double));
  cudaMalloc(&x_values_d, size * sizeof(double));

  cudaMemcpy(csr_offsets_d, A->get_ai().get(), (size + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMemcpy(csr_columns_d, A->get_aj().get(), nnz * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMemcpy(csr_values_d, A->get_av().get(), nnz * sizeof(double),
             cudaMemcpyHostToDevice);

  cudaStreamCreate(&stream);
  /* Creating the cuDSS library handle */
  cudssCreate(&handle);
  cudssSetStream(handle, stream);

  cudssConfigCreate(&solverConfig);
  cudssDataCreate(handle, &solverData);
  cudssIndexBase_t base = A->mkl_base() == SPARSE_INDEX_BASE_ZERO
                              ? CUDSS_BASE_ZERO
                              : CUDSS_BASE_ONE;
  /* Create a matrix object for the sparse input matrix. */
  cudssMatrixCreateCsr(&cudaMat, size, size, nnz, csr_offsets_d, NULL,
                       csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F,
                       CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, base);

  cudssMatrixCreateDn(&res, size, 1, size, x_values_d, CUDA_R_64F,
                      CUDSS_LAYOUT_COL_MAJOR);
}

bool cudss_solver::factorize() {

  status = cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                        cudaMat, 0, 0);
  if (status != CUDSS_STATUS_SUCCESS)
    std::cerr << "fucked\n";
  status = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                        solverData, cudaMat, 0, 0);
  if (status != CUDSS_STATUS_SUCCESS)
    std::cerr << "fucked\n";

  return true;
}

bool cudss_solver::solve(double const *const b, double *const x) {

  cudaMemcpy(b_values_d, b, size * sizeof(double), cudaMemcpyHostToDevice);
  cudssMatrixCreateDn(&rhs, size, 1, size, b_values_d, CUDA_R_64F,
                      CUDSS_LAYOUT_COL_MAJOR);
  cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, cudaMat,
               res, rhs);
  cudssMatrixDestroy(rhs);
  cudaStreamSynchronize(stream);

  cudaMemcpy(x, x_values_d, size * sizeof(double), cudaMemcpyDeviceToHost);
  return true;
}

cudss_solver::~cudss_solver() {
  cudssMatrixDestroy(cudaMat);
  cudssMatrixDestroy(res);
  cudssDataDestroy(handle, solverData);
  cudssConfigDestroy(solverConfig);
  cudssDestroy(handle);
  cudaStreamDestroy(stream);

  cudaFree(csr_offsets_d);
  cudaFree(csr_columns_d);
  cudaFree(csr_values_d);
  cudaFree(x_values_d);
  cudaFree(b_values_d);
}
} // namespace mkl_wrapper
#endif