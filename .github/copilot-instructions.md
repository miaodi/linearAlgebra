# Linear Algebra Library - AI Agent Instructions

## Project Overview
High-performance numerical linear algebra library focused on sparse matrices, iterative solvers, and OpenMP-based parallelism. Future GPU support planned. Performance-critical code benchmarked with Google Benchmark.

## Architecture

### Core Modules (in dependency order)
1. **utils/** - Foundation utilities
   - `ObjectPool.hpp`: RAII object pooling for memory reuse (used heavily in reordering algorithms)
   - `CircularBuffer`: Fixed-size circular buffer for BFS/DFS operations
   - `BitVector`: Compact boolean arrays for graph algorithms
   - `traits.hpp`: C++20 concepts for type constraints

2. **sparse_mat_op/** - Sparse matrix operations & algorithms
   - CSR (Compressed Sparse Row) format is the primary data structure
   - `sparse_mat_traits.hpp`: Defines `CSRMatrixType` concept - all matrix types must satisfy this
   - `spmv.hpp`: SpMV implementations (serial and parallel)
   - `iterative_solver.hpp`: GMRES solver with left/right/no preconditioning
   - `graph_algs.hpp`: Graph algorithms on matrix structure (elimination trees, DAG detection, transitive reduction)
   - `permutation.hpp`: Matrix reordering operations `P*A*Q^T`

3. **mkl_wrapper/** - Intel MKL integration
   - `mkl_sparse_mat.h`: Wrapper around MKL sparse matrix types with RAII semantics
   - `mkl_sparse_mat_sym`: Upper triangular symmetric matrix specialization
   - Provides fallback for when custom implementations aren't needed

4. **reordering/** - Matrix reordering algorithms
   - `MinimumDegree.hpp`: AMD/EMD implementation using quotient graphs
   - `BFS.h`: Breadth-first search for Cuthill-McKee reordering
   - `UnionFind.h`: Disjoint-set data structure (supports parallel operations)

5. **factorization/** - Direct factorization methods
   - `Cholesky.hpp`: Symbolic and numeric Cholesky factorization
   - Elimination tree computation, post-ordering, skeleton graphs

## Build System

### CMake Presets (use these!)
```bash
# Configure with a preset
cmake --preset release-gcc    # Main build (uses GCC)
cmake --preset debug-gcc      # Debug with sanitizers
cmake --preset release-cuda   # CUDA-enabled build
cmake --preset release-intel  # Intel compiler

# Build
cmake --build release --parallel 16
```

**Output directories**: Presets create `debug/`, `release/`, `debug_cuda/`, `release_nv/` at project root.

### Key CMake Options
- `USE_CUDA=ON`: Enable CUDA support (required for cuDSS)
- `USE_CUDSS=ON`: cuDSS direct solver (requires USE_CUDA)
- `USE_AMGCL_LIB=ON`: AMG preconditioners
- `USE_MUMPS_LIB=ON`: MUMPS direct solver (non-MPI)
- `USE_METIS_LIB=ON`: METIS reordering
- `BUILD_BENCHMARKS=ON`: Build Google Benchmark tests

### Dependencies
- **Required**: Eigen3, Intel MKL (threading auto-detected), TBB, OpenMP, LAPACK
- **Fetched**: fast_matrix_market, Spectra, cxxopts, AMGCL (if enabled), Google Test/Benchmark
- **Python**: ssgetpy required for downloading test matrices from SuiteSparse

## Code Patterns

### CSR Matrix Interface
All matrix types satisfy the `CSRMatrixType` concept:
```cpp
template <typename T>
concept CSRMatrixType = requires(T obj) {
  typename T::ROWTYPE;  // Row pointer type (MKL_INT, int, etc.)
  typename T::COLTYPE;  // Column index type
  typename T::VALTYPE;  // Value type (double, float)
  { obj.AI() } -> std::same_as<typename T::ROWTYPE*>;  // Row pointers
  { obj.AJ() } -> std::same_as<typename T::COLTYPE*>;  // Column indices
  { obj.AV() } -> std::same_as<typename T::VALTYPE*>;  // Values
  { obj.rows };  // Number of rows
  { obj.cols };  // Number of columns
};
```

### Index Base Conventions
- **Internal algorithms**: 0-based indexing
- **MKL interface**: Configurable via `sparse_index_base_t` (usually 0-based)
- Matrix readers handle 1-based MTX format conversion automatically

### Namespace Organization
- `matrix_utils`: Sparse matrix operations, permutations, I/O
- `iterative_solver`: GMRES and iterative methods
- `reordering`: Graph reordering algorithms
- `factorization`: Direct factorization methods
- `mkl_wrapper`: MKL interface wrappers
- `utils`: Generic utilities
- `vec_ops`: Vector operations (often inlined)

### Parallelism
- OpenMP directives for SMP parallelism (use `omp_get_max_threads()`)
- Atomic operations for parallel union-find and graph coloring
- Thread-local storage patterns in symbolic factorization (`_ais(nthreads)`, `_ajs(nthreads)`)

## Testing

### Running Tests
```bash
# From build directory (e.g., release/)
ctest                    # Run all tests
ctest -R permutation    # Run tests matching regex
./bin/permutation_test  # Run specific test binary directly
```

### Test Matrices
Tests use SuiteSparse matrices downloaded via `download_sparse_matrix()` in CMakeLists.txt:
- `bcsstk17`: SPD, 10974×10974
- `ex5`: Small SPD, 27×27
- `rdist1`: Unsymmetric, 4134×4134

Matrices cached in `<build_dir>/data/`.

### Test Structure
- Google Test framework (`TEST(Suite, Name)`)
- Each `.cpp` file in `tests/` becomes a standalone executable
- Common pattern: Load matrix → Apply algorithm → Verify correctness

## Benchmarks

```bash
# Build with benchmarks enabled
cmake --preset release-gcc -DBUILD_BENCHMARKS=ON
cmake --build release

# Run benchmarks
./release/benchmarks/bin/spmv_bench
./release/benchmarks/bin/reordering_bench --benchmark_filter=AMD
```

Use `BENCHMARK_DEFINE_F` for fixture-based benchmarks with setup/teardown (see `unionfind_bench.cpp`).

## Common Development Tasks

### Adding a New Algorithm
1. Place implementation in appropriate module (`sparse_mat_op/`, `reordering/`, etc.)
2. If it operates on matrices, use `CSRMatrixType` concept for template parameters
3. Add unit test in `tests/` (use existing tests as templates)
4. Add benchmark in `benchmarks/` if performance-critical
5. Update module's CMakeLists.txt if new files added

### Working with MKL Sparse Matrices
```cpp
// Read from MTX file
auto [A, base] = mkl_wrapper::read_mtx("path/to/matrix.mtx");

// Operations
auto B = mkl_sparse_mult(A, C);           // Sparse-sparse product
auto D = mkl_sparse_sum(A, B, alpha);     // D = alpha*A + B
auto E = mkl_sparse_mult_ptap(A, P);      // E = P^T * A * P

// Convert to custom CSR format
matrix_utils::CSRMatrix<int, int, double> custom(A.rows(), A.cols(), 
                                                  A.get_ai(), A.get_aj(), A.get_av());
```

### Debugging Tips
- Use `debug-gcc` preset with sanitizers enabled
- Matrix visualization: `A.print_svg(std::cout)` or `A.print_gnuplot(std::cout)`
- Check matrix properties: `A.check()` validates CSR structure
- Enable MKL verbose mode by setting `SPARSE_STATUS_VERBOSE`

## File Naming Conventions
- Headers: `.hpp` for C++ template headers, `.h` for C-style or class declarations
- Tests: `*_test.cpp`
- Benchmarks: `*_bench.cpp`
- Executables: In `execs/` (many are experimental/debugging tools)

## Important Notes
- **Do not modify** `mkl_wrapper/` internals without understanding MKL handle lifecycle (use `sparse_matrix_t` via RAII wrapper)
- **Elimination trees** must be post-ordered before use in symbolic factorization
- **Symmetric matrices** assumed upper-triangular storage unless documented otherwise
- **ObjectPool** deleter returns objects to pool - don't bypass with raw pointers
- **CircularBuffer** size must be known at construction (pre-allocate for graph algorithms)
