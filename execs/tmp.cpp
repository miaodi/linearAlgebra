
#include "../mkl_wrapper/mkl_eigen.h"
#include "../mkl_wrapper/mkl_solver.h"
#include "../mkl_wrapper/mkl_sparse_mat.h"
#include "../utils/timer.h"
#include "../utils/utils.h"
#include "arpack.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
/*Mode 3: K * x = lambda * M * x, K symmetric, M symmetric positive
   semi-definite ===> OP = (inv[K - sigma * M]) * M and B = M. ===>
   Shift-and-Invert mode

*/
int main(int argc, char **argv) {

  std::ifstream fm("../../data/eigenvalue/parabolic_fem.mtx");
  std::ifstream fk("../../data/eigenvalue/parabolic_fem.mtx");

  std::vector<MKL_INT> k_csr_rows, k_csr_cols;
  std::vector<double> k_csr_vals;
  utils::read_matrix_market_csr(fk, k_csr_rows, k_csr_cols, k_csr_vals);
  std::shared_ptr<MKL_INT[]> k_csr_rows_ptr(k_csr_rows.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> k_csr_cols_ptr(k_csr_cols.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<double[]> k_csr_vals_ptr(k_csr_vals.data(), [](double[]) {});

  std::vector<MKL_INT> m_csr_rows, m_csr_cols;
  std::vector<double> m_csr_vals;
  utils::read_matrix_market_csr(fm, m_csr_rows, m_csr_cols, m_csr_vals);
  std::shared_ptr<MKL_INT[]> m_csr_rows_ptr(m_csr_rows.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<MKL_INT[]> m_csr_cols_ptr(m_csr_cols.data(),
                                            [](MKL_INT[]) {});
  std::shared_ptr<double[]> m_csr_vals_ptr(m_csr_vals.data(), [](double[]) {});

  const MKL_INT size = m_csr_rows.size() - 1;

  mkl_wrapper::mkl_sparse_mat k(size, size, k_csr_rows_ptr, k_csr_cols_ptr,
                                k_csr_vals_ptr);
  mkl_wrapper::mkl_sparse_mat m(size, size, m_csr_rows_ptr, m_csr_cols_ptr,
                                m_csr_vals_ptr);

  std::cout << "m: " << k.rows() << " , n: " << k.cols() << std::endl;
  // arpack params
  a_int ido{0};
  a_int n = size;
  char which[2] = {'L', 'A'};
  char bMat = 'I';
  a_int nev = 1;
  double tol = 1e-2;
  std::vector<double> resid(n);
  a_int ncv = 2;
  a_int ldv = n;
  std::vector<double> v(ldv * ncv);
  std::vector<a_int> iparam(11, 0);
  iparam[0] =
      1; // exact shifts with respect to the reduced tridiagonal matrix T.

  /*iparam[1] No longer referenced.*/
  int max_it = 1000;
  int mode = 3;
  iparam[2] = max_it;
  iparam[3] = 1; // NB: blocksize to be used in the recurrence. The code
                 // currently works only for NB = 1.
  iparam[4] = 0; // NCONV: number of "converged" Ritz values.
  /*iparam[5] IUPD No longer referenced. Implicit restarting is ALWAYS used. */
  iparam[6] = mode; // MODE On INPUT determines what type of eigenproblem is
                    // being solved. Must be 1,2,3,4,5;

  /*rest iparam are output*/

  std::vector<a_int> ipntr(11, 0);

  std::vector<double> workd(
      3 * n, 0.); // Distributed array to be used in the basic Arnoldi iteration
  // for reverse communication. The user should not use WORKD as
  // temporary workspace during the iteration.
  a_int lworkl = ncv * ncv + 8 * ncv;
  std::vector<double> workl(lworkl); // Private (replicated) array on each PE
                                     // or array allocated on the front end.

  a_int info = 0;

  /* create solver for y = inv(K)*x*/
  mkl_wrapper::mkl_direct_solver pardiso(&k);
  pardiso.factorize();
  // Implicitly Restarted Arnoldi Iteration
  do {
    dsaupd_c(&ido, &bMat, n, which, nev, tol, resid.data(), ncv, v.data(), ldv,
             iparam.data(), ipntr.data(), workd.data(), workl.data(), lworkl,
             &info);

    if (info != 0) {
      std::cerr << "fucked!" << std::endl;
      return 1;
    }

    a_int x_idx = ipntr[0] - 1; // 0-based (Fortran is 1-based).
    a_int y_idx = ipntr[1] - 1; // 0-based (Fortran is 1-based).

    double *X = workd.data() + x_idx; // Arpack provides X.
    double *Y = workd.data() + y_idx; // Arpack provides Y.

    // std::cout << "ido: " << ido << std::endl;
    if (ido == -1) {
      pardiso.solve(X, Y);
    } else if (ido == 1) {
      pardiso.solve(X, Y);
    } else if (ido == 2) {
      break;
    } else if (ido != 99) {
      std::cerr << "Error: unexpected ido " << ido << " - KO" << std::endl;
      return 1;
    }

  } while (ido != 99);

  // Extract eigen pairs
  std::cout << "nconv: " << iparam[4] << std::endl;
  std::cout << "niter: " << iparam[2] << std::endl;

  a_int rvec = 0;
  char howmny = 'A';
  std::vector<a_int> select(ncv, 1);
  std::vector<double> dr(
      nev + 1,
      0.); // D contains the Ritz value approximations to the eigenvalues
  std::vector<double> di(nev + 1, 0.);

  std::vector<double> z(n * (nev + 1), 0.); // Caution: nev+1 for dneupd.

  double sigma_real{0.}, sigma_image{0.};
  dseupd_c(rvec, &howmny, select.data(), dr.data(), z.data(), z.size(),
           sigma_real, &bMat, n, which, nev, tol, resid.data(), ncv, v.data(),
           v.size(), iparam.data(), ipntr.data(), workd.data(), workl.data(),
           lworkl, &info);
  for (int i = 0; i < nev; i++) {
    std::cout << dr[i] << std::endl;
  }
  return 0;
}