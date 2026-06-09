This is a project for me to learn numerical linear algebra (solvers, multigrid, sparse matrix, spmv, etc.) and high performance computing. Currently, I am focusing on OpenMP based Shared-Memory Parallelism (SMP). May start implementing GPU based algorithm once I finish OMSCS [CS 8803 O21: GPU Hardware and Software]([/guides/content/editing-an-existing-page](https://omscs.gatech.edu/cs-8803-o21-gpu-hardware-and-software)). 

Algorithms I implemented are mainly from published articles or from other open source libraries. The performance of these algorithms will be tested by using Google Benchmark. 

## CUDA H100 Deployment

For H100 systems with an older driver than the CUDA toolkit, avoid relying on
PTX JIT. For example, CUDA 12.8-generated PTX is not JIT-compatible with an
R535 driver even though many CUDA 12.x binaries can run under minor-version
compatibility. Build native H100 cubins instead:

```sh
cmake --preset release-cuda-h100
cmake --build --preset release-cuda-h100
```

The preset uses `CMAKE_CUDA_ARCHITECTURES=90-real`, which emits `sm_90` code
without PTX fallback. To verify that a build is not relying on PTX JIT on an
H100 node, enable tests for a separate build and run the CUDA test subset with
PTX JIT disabled:

```sh
cmake -S . -B release_h100_tests \
  -DUSE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90-real \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=ON \
  -DENABLE_DATA_DOWNLOADS=OFF
cmake --build release_h100_tests -j
CUDA_DISABLE_PTX_JIT=1 ctest --test-dir release_h100_tests -R cuda --output-on-failure
```

If `CUDA_FORCE_PTX_JIT=1` fails on a cluster with an older driver, but
`CUDA_DISABLE_PTX_JIT=1` passes, the deployment is using native cubins as
intended.
