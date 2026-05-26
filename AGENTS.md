# Linear Algebra Project Context

This is a C++20 numerical linear algebra and high-performance computing project.
It focuses on sparse matrix algorithms, iterative and direct solvers,
preconditioners, graph-based sparse computations, reordering, symbolic
factorization, and performance benchmarking.

The project is used for learning and experimenting with numerical linear algebra
and HPC techniques. Current emphasis includes shared-memory parallelism with
OpenMP/TBB, sparse matrix operations, preconditioners, symbolic factorization,
graph algorithms, and benchmark-driven performance analysis. CUDA support exists
optionally for GPU sparse operations and experiments.

## Repository Map

- `sparse_mat_op/`: sparse matrix operations, SpMV, SpGEMM, triangular solve,
  scaling, permutation, preconditioning, iterative solvers, and Matrix Market IO.
- `sparse_mat_op/cuda/`: optional CUDA implementations for GPU sparse operations,
  GMRES, ILU symbolic work, scaling, and CUDA preconditioners.
- `graph/`: graph traversal and tree algorithms used by sparse matrix and
  factorization code.
- `reordering/`: matrix and graph reordering algorithms.
- `factorization/`: symbolic factorization, especially Cholesky-related
  algorithms.
- `mkl_wrapper/`: wrappers around MKL, MUMPS, ARPACK, AMGCL, cuDSS, and related
  solver/eigen functionality.
- `solver/`: higher-level linear solver system and transformation pipeline.
- `utils/`: timers, traits, object pool, circular buffer, bit vectors, sorting
  utilities, and other shared helpers.
- `tests/`: GoogleTest-based unit tests.
- `benchmarks/`: Google Benchmark performance tests.
- `execs/`: standalone experiments and debugging executables.
- `scratch/`: temporary development experiments.
- `cmake/`: CMake modules, dependency setup, data download helpers, and build
  utilities.

## Build And Test

The project uses CMake.

The common local build directory is `release/`.

Common configure command:

```sh
cmake -S . -B release -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Common build command:

```sh
cmake --build release -j
```

Run tests:

```sh
ctest --test-dir release --output-on-failure
```

Useful CMake options:

- `USE_MKL=ON` by default.
- `USE_CUDA=OFF` by default.
- `BUILD_BENCHMARKS=OFF` by default.
- `ENABLE_TESTS=ON` by default.
- `ENABLE_DATA_DOWNLOADS=ON` by default.
- `BUILD_DATA=OFF` by default.
- Optional libraries include `USE_METIS_LIB`, `USE_MUMPS_LIB`,
  `USE_ARPACK_LIB`, `USE_AMGCL_LIB`, `USE_SPECTRA_LIB`, and `USE_CUDSS`.

Tests and data downloads may require the Python package `ssgetpy` in the Python
interpreter selected by CMake.

## Development Guidance

- Prefer small, focused C++ changes.
- Preserve the existing C++20 style and CMake structure.
- After generating or modifying C/C++ code, reformat the files changed with
  `clang-format` using the repository `.clang-format`.
- Be careful with sparse matrix indexing, base offsets, CSR invariants,
  row/column type aliases, and symmetry assumptions.
- Verify algorithmic changes with existing tests when possible.
- For performance-sensitive code, avoid unnecessary allocations, copies,
  synchronization, and virtual dispatch.
- For OpenMP/TBB changes, reason about thread safety, false sharing, load
  balance, and deterministic output when tests depend on ordering.
- For CUDA code, consider memory coalescing, occupancy, synchronization,
  launch overhead, and host-device transfer costs.
- For CMake changes, preserve optional dependency behavior.
- For Cholesky symbolic or multifrontal work, keep elimination-tree ordering,
  postorder mappings, frontal matrix assembly, and sparse index invariants
  explicit. Verify with the focused Cholesky tests when possible.
- Avoid changing benchmark behavior unless explicitly requested.
- Treat `execs/` and `scratch/` as experimental code unless the task explicitly
  says otherwise.

## AI Assistant Guidance

When using an AI tool on this project, provide the specific target area and the
verification expectation. Good requests include the files, algorithm, benchmark,
or failing test involved.

## AGENTS.md Maintenance Policy

When making major code changes, update `AGENTS.md` in the same change if the
work changes repository structure, build/test commands, important dependencies,
algorithm ownership, verification expectations, or project-specific development
guidance. Keep these updates focused on durable guidance rather than temporary
implementation notes.

Examples:

```text
Investigate why tests/triangular_solve_test.cpp fails and make the smallest fix.
Run the relevant test afterward.
```

```text
Optimize sparse_mat_op/spgemm.cpp for fewer allocations without changing public
behavior. Add or update a benchmark only if needed.
```

```text
Add a CUDA implementation for the existing CPU behavior in sparse_mat_op/ruiz_scale.cpp.
Keep USE_CUDA=OFF builds working.
```

For complex algorithm work, ask the AI to first identify invariants and compare
the implementation against the relevant paper or known sparse linear algebra
algorithm before editing code.
