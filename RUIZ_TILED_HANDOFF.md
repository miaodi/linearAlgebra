RuizScale tiled optimization handoff

Current state

- Fixed clangd jump/parsing issues for CUDA files.
- Updated `.clangd` so CUDA-specific config actually matches `.cu` and `.cuh` files.
- Switched clangd CUDA parsing away from raw nvcc-only flags.
- Simplified CUDA error-message construction in CUDA code to avoid clangd parse failures.
- Added small accessor methods on `DeviceTileCOOMatrix` to avoid bogus clangd member-resolution failures.

Recent code changes

- `sparse_mat_op/cuda/cuda_ruiz_scale_impl.cu`
  - Added per-tile heuristics:
    - `compute_norms_tiled`: use shared memory when `tile_nnz >= 2 * tile_size`
    - `scale_tiled_values`: use shared memory when `tile_nnz >= 4 * tile_size`
  - Both kernels now have explicit `if (use_shared) { ... } else { ... }` structure.
- `sparse_mat_op/cuda/cuda_tiled_sparse_mat.cuh`
  - Added accessors like `tileRowIndData()`, `tileNnzPrefixData()`, `valuesData()`, `nnz()`.
- `benchmarks/cuda_ruiz_scale_bench.cu`
  - Fixed stale shared-memory guard in tiled benchmark:
    - old assumption: `warps_per_block * 2 * tile_size * sizeof(double)`
    - current kernel reality: `2 * tile_size * sizeof(double)`

Verification done

- `clangd --check sparse_mat_op/cuda/cuda_ruiz_scale_impl.cu --compile-commands-dir=. --enable-config`
  - passes with 0 errors after the fixes above

Latest benchmark result

```text
BM_RuizScaleCudaCSR                 41.7 ms         39.5 ms          177 items_per_second=4.34871G/s
BM_RuizScaleCudaTiled/k=2/2          216 ms          213 ms           34 items_per_second=803.839M/s n_tiles=4.26822M tile_k=2
BM_RuizScaleCudaTiled/k=4/4          114 ms          112 ms           63 items_per_second=1.53861G/s n_tiles=2.1624M tile_k=4
BM_RuizScaleCudaTiled/k=6/6         71.4 ms         69.0 ms          102 items_per_second=2.48788G/s n_tiles=1.10055M tile_k=6
BM_RuizScaleCudaTiled/k=8/8         64.7 ms         62.4 ms          112 items_per_second=2.7495G/s n_tiles=419.798k tile_k=8
BM_RuizScaleCudaTiled/k=10/10       52.4 ms         50.1 ms          100 items_per_second=3.42473G/s n_tiles=94.19k tile_k=10
```

Main conclusion so far

- The shared-vs-global heuristic helped.
- The next likely bottleneck is load imbalance from very sparse / almost-empty tiles.
- Current mapping is effectively one block per tile, which wastes work on low-nnz tiles.

Recommended next step

Implement sparse/dense tile partitioning based on `tile_nnz`.

Suggested plan

1. Compute `tile_nnz = tile_nnz_prefix[t + 1] - tile_nnz_prefix[t]`.
2. Partition tile IDs into at least two buckets:
   - sparse
   - dense
3. Keep current block-per-tile kernel for dense tiles.
4. Add a sparse path with multiple tiles per block, ideally one warp per tile.
5. For the sparse path, start without shared memory.
6. Apply this to `scale_tiled_values` first, then to `compute_norms_tiled`.

Why this next

- Many nearly empty tiles mean poor occupancy and block-level overhead dominates.
- The current heuristic only decides shared vs global access inside one tile.
- It does not fix underutilization caused by assigning a whole block to a tiny tile.

Notes

- `tile_nnz` is the preferred first heuristic because it is already available from `tile_nnz_prefix`.
- Better metadata like unique row/col counts may help later, but is not the first thing to add.
