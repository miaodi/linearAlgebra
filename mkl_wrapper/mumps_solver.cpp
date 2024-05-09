#include "mumps_solver.h"

#ifdef USE_MUMPS_LIB

#include "mkl_sparse_mat.h"
#include <execution>
#include <iostream>

// Macro s.t. indices match MUMPS documentation
#define MUMPS_ICNTL(I) icntl[(I)-1]
#define MUMPS_CNTL(I) cntl[(I)-1]
#define MUMPS_INFO(I) info[(I)-1]
#define MUMPS_INFOG(I) infog[(I)-1]

namespace mkl_wrapper {
mumps_solver::mumps_solver(mkl_sparse_mat const *A) : mkl_solver() {

  // initialize mumps
  id.comm_fortran = 0; // should be ignored for MPI-free version
  id.par = 1;          // host version
  id.job = -1;         // Initialize MUMPS

  if (A->mkl_descr().type == SPARSE_MATRIX_TYPE_GENERAL) {
    id.sym = 0; // unsymmetric
  } else if (A->mkl_descr().type == SPARSE_MATRIX_TYPE_SYMMETRIC) {
    if (A->positive_definite()) {
      id.sym = 1; // symmetric positive definite
    } else {
      id.sym = 2; // general symmetric
    }
  } else {
    std::cerr << "Matrix type not supported by MUMPS" << std::endl;
  }
  dmumps_c(&id);

  // setup matrix
  id.n = A->rows();
  id.nz = A->nnz();

  row_ptr.reset(new int[id.nz]);
  col_ptr.reset(new int[id.nz]);

  for (int i = 0; i < id.n; i++) {
    for (int j = A->get_ai()[i] - A->mkl_base();
         j < A->get_ai()[i + 1] - A->mkl_base(); j++) {
      row_ptr[j] = i + 1;
      col_ptr[j] = A->get_aj()[j] + (1 - A->mkl_base());
    }
  }
  //   for (int i = 0; i < id.nz; i++) {
  //     std::cout << row_ptr[i] << " " << col_ptr[i] << std::endl;
  //   }
  id.irn = row_ptr.get();
  id.jcn = col_ptr.get();
  id.a = const_cast<double *>(A->get_av().get());
}

bool mumps_solver::factorize() {
  // Set MUMPS default parameters
  set_params();
  id.job = 1; // Analysis
  dmumps_c(&id);
  if (id.MUMPS_INFOG(1) != 0) {
    std::cerr << "WARNING mumps_solver Analysis "
              << " Error " << id.MUMPS_INFOG(1)
              << " returned in substitution dmumps()\n";
    return false;
  }

  id.job = 2; // Factorization
  {
    const int mem_relax_lim = 300;
    while (true) {
      dmumps_c(&id);
      if (id.MUMPS_INFOG(1) < 0) {
        if (id.MUMPS_INFOG(1) == -8 || id.MUMPS_INFOG(1) == -9) {
          id.MUMPS_ICNTL(14) += 20;
          if (id.MUMPS_ICNTL(14) > mem_relax_lim) {
            std::cerr
                << "Memory relaxation limit reached for MUMPS factorization\n";
            return false;
          }
          std::cout << "Re-running MUMPS factorization with memory relaxation "
                    << id.MUMPS_ICNTL(14) << '\n';
        } else {
          std::cerr << "WARNING mumps_solver Factorization "
                    << " Error " << id.MUMPS_INFOG(1)
                    << " returned in substitution dmumps()\n";
          return false;
        }
      } else {
        break;
      }
    }
  }
  row_ptr.reset();
  col_ptr.reset();
  return true;
}

void mumps_solver::set_params() {
  // Output stream for error messages
  id.MUMPS_ICNTL(1) = 6;
  // Output stream for diagnosting printing local to each proc
  id.MUMPS_ICNTL(2) = 0;
  // Output stream for global info
  id.MUMPS_ICNTL(3) = 6;
  // Level of error printing
  id.MUMPS_ICNTL(4) = _print_level;
  // Input matrix format (assembled)
  id.MUMPS_ICNTL(5) = 0;
  // Permutes the matrix to a zero-free diagonal and/or scale the matrix
  id.MUMPS_ICNTL(6) = 7; // auto
  // Computes a symmetric permutation (ordering)
  id.MUMPS_ICNTL(7) = 7; // auto
  // Describes the scaling strategy
  id.MUMPS_ICNTL(8) = 77; // auto
  // Use A or A^T
  id.MUMPS_ICNTL(9) = 1; // AX = B is solved.
  // Iterative refinement
  id.MUMPS_ICNTL(10) = 0; // no iterative refinement
  // Computes statistics related to an error analysis
  id.MUMPS_ICNTL(11) = 0; // no error analysis
  // Defines an ordering strategy for symmetric matrices (SYM = 2)
  id.MUMPS_ICNTL(12) = 0; // automatic choice
  // Controls the parallelism of the root node (enabling or not the use of
  // ScaLAPACK)
  id.MUMPS_ICNTL(13) = 0; //(parallel factorization on the root node)
  // Percentage increase of estimated workspace (default = 20%)
  id.MUMPS_ICNTL(14) = 20;
  // exploits compression of the input matrix resulting from a block format
  id.MUMPS_ICNTL(15) = 0; // no compression
  // Number of OpenMP threads (default)
  id.MUMPS_ICNTL(16) = 0; // MUMPS uses the number of OpenMP threads
                          // configured by the calling application
  // Matrix input format (distributed)
  id.MUMPS_ICNTL(18) = 0; // the input matrix is centralized on the host
  // Schur complement (no Schur complement matrix returned)
  id.MUMPS_ICNTL(19) = 0;
  // Determines the format (dense, sparse, or distributed) of the right-hand
  // sides
  id.MUMPS_ICNTL(20) = 0; // dense right-hand sides
  //  Determines the distribution (centralized or distributed) of the solution
  //  vectors
  id.MUMPS_ICNTL(21) = 0; // assembled centralized format
  // The in-core/out-of-core (OOC) factorization and solve.
  id.MUMPS_ICNTL(22) = 0; // In-core factorization and solution phases
  // Corresponds to the maximum size of the working memory in MegaBytes
  id.MUMPS_ICNTL(23) = 0; // each processor will allocate workspace based on
                          // the estimates computed during the analysis
  // Eetermines whether a sequential or parallel computation of the ordering
  // is performed
  id.MUMPS_ICNTL(28) = 0; // automatic choice
  // Defines the parallel ordering tool (when ICNTL(28)=1) to be used to
  // compute the fill-in reducing permutation
  id.MUMPS_ICNTL(29) = 0; // automatic choice
  //  Computes the determinant of the input matrix.
  id.MUMPS_ICNTL(33) =
      0; // the determinant of the input matrix is not computed.
}

bool mumps_solver::solve(double const *const b, double *const x) {
  id.job = 3; // Solve
  std::copy(std::execution::par, b, b + id.n, x);
  id.rhs = x;
  dmumps_c(&id);

  if (id.MUMPS_INFOG(1) != 0) {
    std::cerr << "WARNING mumps_solver Solve "
              << " Error " << id.MUMPS_INFOG(1)
              << " returned in substitution dmumps()\n";
    switch (id.MUMPS_INFOG(1)) {
    case -5:
      std::cerr << " out of memory allocation error\n";
    case -6:
      std::cerr
          << " cause: Matrix is Singular in Structure: check your model\n";
    case -7:
      std::cerr << " out of memory allocation error\n";
    case -8:
      std::cerr << "Work array too small; use -ICNTL14 option, the default is "
                   "-ICNTL 20 make 20 larger\n";
    case -9:
      std::cerr << "Work array too small; use -ICNTL14 option, the default is "
                   "-ICNTL 20 make 20 larger\n";
    case -10:
      std::cerr << " cause: Matrix is Singular Numerically\n";
    default:;
    }
    return false;
  }
  return true;
}
mumps_solver::~mumps_solver() {}
} // namespace mkl_wrapper

#endif