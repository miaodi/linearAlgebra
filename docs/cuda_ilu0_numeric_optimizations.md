# CUDA ILU0 Numeric Optimizations

This note records the CUDA ILU(0) numeric factorization optimizations explored
on the `RTP_metis.bin` matrix and the main lessons from the current code paths.
It is intended as a durable reference for future tuning work, not as a complete
profiling report.

## Baseline Matrix

Representative benchmark command:

```sh
release/benchmarks/cuda_ilu0_bench \
  -f ~/repo/matrix_lib/RTP_metis.bin \
  --benchmark_min_time=5s
```

Observed matrix statistics:

| Statistic | Value |
| --- | ---: |
| `n` | 1.949482M |
| `nnz(A)` | 33.545908M |
| `nnz(LU0 pattern)` | 33.545908M |
| Topological levels | 314 |
| Strict lower nnz | 14.847128M |
| Cached updates | 124.009491M |

## Benchmark Machine

The RTP timings in this note were measured on the following local machine:

| Component | Value |
| --- | --- |
| Host | `miaodi-Z490-VISION-G` |
| OS | Ubuntu 26.04 LTS (`resolute`) |
| Kernel | Linux `7.0.0-22-generic` x86_64 |
| CPU | Intel Core i9-10900KF, 10 cores / 20 threads, up to 5.3 GHz |
| CPU cache | L1d 320 KiB, L1i 320 KiB, L2 2.5 MiB, L3 20 MiB |
| GPU | NVIDIA GeForce RTX 4070, 12282 MiB |
| NVIDIA driver | 595.71.05 |
| CUDA version reported by `nvidia-smi` | 13.2 |

Google Benchmark reported CPU scaling and ASLR warnings during the RTP runs, so
small timing differences should be treated as noisy unless repeated.

## Current Important Paths

| Path | Main files | Scheduling model |
| --- | --- | --- |
| Level scheduled numeric | `sparse_mat_op/cuda/ilu/ilu_numeric.cu` | One launch per topological level. |
| Persistent spin | `sparse_mat_op/cuda/ilu/ilu_numeric_persistent.cu` | Resident warps claim rows from `next_row`. |
| Persistent cached | `sparse_mat_op/cuda/ilu/ilu_numeric_persistent.cu` | Persistent row scheduler plus precomputed update cache. |
| CTA-granular | `sparse_mat_op/cuda/ilu/ilu_numeric_cta_granular.cu` | One grid block per 8-row chunk of a row permutation, with row-done polling for dependencies. |
| Shared helpers | `sparse_mat_op/cuda/ilu/ilu_numeric_common.cuh` | Shared row waiting, diagonal inverse, publish, and row update helpers. |

The benchmark driver is `benchmarks/cuda_ilu0_bench.cpp`.

## Benchmark Snapshot

Approximate RTP timings observed during development:

| Variant or experiment | Time | Notes |
| --- | ---: | --- |
| `binary_shared` | ~58.0 ms | Original level-scheduled binary search path with shared row staging. |
| Original persistent path | ~50.5 ms | Acquire/release `row_done` polling. |
| Persistent wait-path tuning | ~45.9 ms | Reduced polling/status overhead and delayed sleep. |
| Persistent with `diag_inv` | ~34.6 ms | Dependents multiply by published reciprocal diagonal. |
| Persistent cached | ~30.5 ms | Persistent scheduler plus lower-only update cache. |
| `cta_granular` | ~22.4 ms | Topological `level_perm` order with shared current-row column staging. |
| `cta_granular_identity` | ~44.3 ms | Identity row order with shared current-row column staging. |
| `cta_granular_global` | ~24.9 ms | Topological `level_perm` order, global row lookup, no shared row-column staging. |
| `cta_granular_global_identity` | ~56.0 ms | Identity row order with global row lookup. |
| `cta_granular_cached` | ~18.5 ms | Topological CTA scheduler plus lower-only update cache. |
| `persistent_spin_perm` | ~18.5 ms | Persistent scheduler with topological `level_perm` row order. |
| `persistent_cached_perm` | ~17.6 ms | Topological persistent scheduler plus lower-only update cache. |
| cuSPARSE `csrilu02` | ~21.8 ms | Reference library implementation on the same matrix. |

The most important recent experiment compared these variants:

```sh
release/benchmarks/cuda_ilu0_bench \
  -f ~/repo/matrix_lib/RTP_metis.bin \
  --benchmark_filter='ILU0Numeric/(persistent_spin|persistent_spin_perm|persistent_cached|persistent_cached_perm|cta_granular|cta_granular_identity|cta_granular_global|cta_granular_global_identity|cta_granular_cached)' \
  --benchmark_min_time=5s \
  --benchmark_counters_tabular=true
```

Observed result:

| Benchmark | Time |
| --- | ---: |
| `ILU0Numeric/cta_granular/real_time` | 22.4 ms |
| `ILU0Numeric/cta_granular_identity/real_time` | 44.3 ms |
| `ILU0Numeric/cta_granular_global/real_time` | 24.9 ms |
| `ILU0Numeric/cta_granular_global_identity/real_time` | 56.0 ms |
| `ILU0Numeric/cta_granular_cached/real_time` | 18.5 ms |
| `ILU0Numeric/persistent_spin/real_time` | 34.7 ms |
| `ILU0Numeric/persistent_spin_perm/real_time` | 18.5 ms |
| `ILU0Numeric/persistent_cached/real_time` | 30.6 ms |
| `ILU0Numeric/persistent_cached_perm/real_time` | 17.6 ms |

This experiment shows that topological row ordering is a major part of the
`cta_granular` speedup. It also shows that the fine-grained persistent scheduler
benefits even more from using the same topological order.

## Optimization: Topological Row Order

The CTA-granular path launches one grid block per 8-row chunk and maps each
claimed row slot through the caller-provided row permutation:

```cpp
const COLTYPE row_slot = atomicAdd(next_row, kCtaGranularWarpsPerBlock) + warp_in_block;
const COLTYPE row = row_perm[row_slot] - base;
```

The CTA-granular numeric kernel still uses row-level dependency waiting through
`row_done`. It no longer builds or uploads CTA predecessor/successor arrays; the
only runtime scheduling state is the row permutation and a monotonic `next_row`
counter. Each grid block atomically claims one 8-row chunk and then retires,
matching cuSPARSE's large-grid shape more closely and avoiding the previous
persistent CTA loop's repeated end-of-task block barrier.

What matters today is the row issue order:

| Schedule | Row order | Result on RTP |
| --- | --- | ---: |
| Topological CTA-granular | `level_perm[0..n)` | ~22.4 ms |
| Identity CTA-granular | `0, 1, 2, ...` | ~44.3 ms |

The identity experiment creates CTA tasks like:

```text
CTA 0: [0, 1, 2, 3, 4, 5, 6, 7]
CTA 1: [8, 9, 10, 11, 12, 13, 14, 15]
...
```

Even though this keeps the same 8-row CTA granularity, it regresses badly. The
likely cause is much more speculative dependency spinning and CTA-level barrier
drag when rows are issued before enough predecessors have completed.

Takeaway: preserve topological or dependency-depth ordering when changing the
CTA-granular path. Any new scheduler should measure how much time is spent waiting
on `row_done`.

## Optimization: Permuted Persistent Row Scheduling

The original persistent spin path claims raw row ids:

```cpp
row = atomicAdd(next_row, COLTYPE(1));
```

The permuted persistent experiments keep the same one-row-per-warp work
granularity but map the claimed work slot through `level_perm`:

```cpp
row_slot = atomicAdd(next_row, COLTYPE(1));
row = row_perm[row_slot] - base;
```

This adds one row-permutation load per row, but preserves the low-overhead
persistent structure. The baseline non-permuted path uses a separate compile-time
kernel instantiation, so it does not pay this row-permutation load or a runtime
branch.

Measured on RTP:

| Path | Time |
| --- | ---: |
| Raw-order persistent spin | 34.7 ms |
| Topological-order persistent spin | 18.5 ms |
| Raw-order persistent cached | 30.6 ms |
| Topological-order persistent cached | 17.6 ms |

This is currently the strongest evidence that row issue order dominates much of
the dependency-wait cost. Both topological persistent paths are faster than the
current CTA-granular path on RTP, despite doing one work-counter atomic per row.

Correctness requirement:

- `row_perm` must be dependency-topological. An arbitrary permutation can issue a
  row before one of its lower dependencies has even been claimed, which can lead
  to very long spinning or deadlock.

## Optimization: Diagonal Inverse Publishing

Earlier paths normalized a lower entry by reading and dividing by `U(k,k)` each
time a dependent row used row `k`. The optimized row-done paths publish the
reciprocal once when row `k` completes:

```cpp
diag_inv[row] = VALTYPE(1) / diagonal;
ready.store(1, cuda::memory_order_release);
```

Dependents wait with acquire semantics on `row_done[k]`, then multiply:

```cpp
aik *= diag_inv[k];
```

This helped the persistent path significantly on RTP:

| Stage | Approximate time |
| --- | ---: |
| Persistent before `diag_inv` | ~45.9 ms |
| Persistent after `diag_inv` | ~34.6 ms |

Correctness notes:

- A row stores `diag_inv[row]` before publishing `row_done[row]`.
- Dependents use acquire polling on `row_done[k]`, so they observe the published
  inverse and the completed row values.
- Using a reciprocal changes floating-point roundoff slightly compared with
  division by the diagonal.

## Optimization: Shared Row-Column Staging

Binary-search update paths repeatedly find whether a reference column `j` exists
in the current row. The shared lookup variant stages the current row's column
indices into per-warp shared memory before doing the binary searches:

```cpp
shared_row_cols[offset] = lu_aj[row_begin + offset] - base;
```

Then the binary search probes shared memory instead of global `lu_aj` for the
current row. This helps rows that fit in the shared cache because each current
row can be searched many times while processing its lower dependencies.

Important constraint:

- `kSharedRowColumnsPerWarp = 256`.
- The persistent path uses shared staging only when
  `row_end - row_begin <= kSharedRowColumnsPerWarp`.
- The default CTA-granular benchmark uses shared staging; the `cta_granular_global`
  variants use global lookup to compare against cuSPARSE's low-shared-memory path.
- Larger rows fall back to global lookup to avoid overflowing the per-warp shared
  buffer.

Takeaway: shared memory helps the binary-search path when the current row is
large enough to make repeated global-memory searches expensive, but still small
enough to fit in the per-warp shared row cache.

Measured CTA-granular lookup experiment on RTP:

| CTA-granular lookup mode | Topological order | Identity order |
| --- | ---: | ---: |
| Shared current-row staging | ~22.4 ms | ~44.3 ms |
| Global current-row lookup | ~24.9 ms | ~56.0 ms |

Removing shared staging reduced shared-memory footprint and register pressure,
but it increased repeated global lookup cost enough to regress the benchmark.

## Optimization: Cached CTA-Granular Updates

The `cta_granular_cached` path keeps the same 8-warps/block CTA scheduler as
`cta_granular`, but replaces runtime row-column searches with the lower-only
update cache:

```text
lower_row_ptr, update_ptr, update_jpos, update_pos
```

The cache stores the exact `(j_pos, pos_i)` update pairs for every strict-lower
entry. This removes the binary search inside the numeric kernel at the cost of a
large persistent cache. On RTP, the cache is about `1.01 GiB` for `124.0M`
updates.

Measured focused benchmark:

| Path | Time |
| --- | ---: |
| `cta_granular_global` | ~24.9 ms |
| `cta_granular` | ~22.3 ms |
| cuSPARSE `USE_LEVEL` | ~21.8 ms |
| `cta_granular_cached` | ~18.5 ms |
| `persistent_cached_perm` | ~17.6 ms |

Focused NCU reports:

| Metric | `cta_granular_global` | cuSPARSE `USE_LEVEL` | `cta_granular_cached` |
| --- | ---: | ---: | ---: |
| Kernel time | 27.609 ms | 23.792 ms | 19.486 ms |
| Grid/block | 243686 x 256 | 243686 x 256 | 243686 x 256 |
| Registers/thread | 24 | 32 | 36 |
| L2 sectors | 604.6M | 421.8M | 366.2M |
| L2 read sectors | 496.1M | 325.8M | 262.8M |
| DRAM read | 2.996 GB | 2.961 GB | 2.898 GB |
| DRAM write | 301.5 MB | 268.5 MB | 300.9 MB |
| Executed SM instructions | 2.892B | 2.898B | 1.569B |
| Thread instructions | 54.779B | 63.884B | 31.244B |
| Long-scoreboard not issued | 1.688M | 1.160M | 1.317M |

The cached CTA path does not eliminate memory latency, but it removes enough
row-search traffic and instructions to beat both the global lookup variant and
cuSPARSE on this matrix. This supports the earlier interpretation that
`cta_granular_global` was losing mostly to repeated global row lookup work, not
to occupancy or shared-memory footprint.

## Optimization: Persistent Row Scheduling

The persistent spin kernel keeps resident warps alive and lets lane 0 of each
warp claim rows from a monotonic counter:

```cpp
row = atomicAdd(next_row, COLTYPE(1));
```

Each warp factors one row and waits on lower dependencies with `row_done[k]`.
This avoids launching one independent block per row and keeps GPU work resident,
but it can also issue rows that are not dependency-ready yet.

Optimizations kept in this path:

- Lane 0 alone polls `row_done[k]`; the result is broadcast to the warp.
- Status checks use relaxed `cuda::atomic_ref` loads.
- `__nanosleep` is delayed until after initial spins.
- Rows with no lower dependencies skip row work and skip the row-update fence.
- Shared row-column staging is used when the current row fits the shared cache.
- `diag_inv` is used to replace repeated divisions with multiplication.

The persistent cached variant keeps the same scheduler but replaces per-update
binary searches with precomputed update arrays:

```text
lower_row_ptr, update_ptr, update_jpos, update_pos
```

This reduced the RTP time from roughly `34.6 ms` to `30.5 ms`.

## Optimization: CTA Granularity

The CTA-granular kernel launches one grid block for each CTA task. Each task
contains up to 8 rows by default, one row per warp:

```text
block claims one CTA task -> warp 0 handles row slot 0, ..., warp 7 handles row slot 7
```

Compared with persistent row scheduling:

| Path | Work claim granularity | Global scheduling atomics |
| --- | --- | --- |
| Persistent spin | 1 row per warp claim | About one per row. |
| CTA-granular | Up to 8 rows per CTA claim | About one per 8 rows. |

This coarser granularity reduces scheduler atomic traffic. Launching one block
per CTA task also removes the previous persistent-loop `__syncthreads()` after
each task; a focused NCU run on RTP reduced the barrier stall ratio from about
`20.66` to about `0.38` per issued instruction. The identity experiment still
shows that topological `level_perm` ordering is essential on RTP.

## Experiments To Avoid Repeating Blindly

| Experiment | Result | Decision |
| --- | ---: | --- |
| Row fetch chunking | ~337 ms | Rejected due to severe regression. |
| 32-warps/SM cap | ~54.6 ms | Rejected for the tuned persistent path. |
| Forced global lookup | ~53.0 ms | Rejected for the level-scheduled path; shared row-column staging remains useful. |
| CTA-granular global lookup | ~24.9 ms | Slower than the shared-staged CTA-granular variant; retained for cuSPARSE comparison. |
| CTA-granular cached lookup | ~18.5 ms | Retained as a high-memory comparison path using the existing lower-only update cache. |
| Busy spin without `__nanosleep` | ~49.5 ms | Rejected; delayed sleep performed better. |
| Forced 128/256 block size | Neutral | Not retained as a tuning knob. |
| Identity CTA-granular order | ~44.3 ms | Rejected; topological ordering matters. |

## Current Interpretation

The best current in-repo path for RTP is `persistent_cached_perm` at roughly
`17.6 ms`, ahead of `cta_granular_cached` at roughly `18.5 ms` and the measured
cuSPARSE `csrilu02` result of roughly `21.8 ms`.

The main lessons are:

- Topological row order is a major performance feature, not just a correctness
  convenience.
- Topological order helps both coarse CTA-granular scheduling and fine-grained
  persistent row scheduling.
- `diag_inv` is a major row-done path optimization.
- Shared row-column staging helps binary search when the current row fits the
  shared cache and is searched repeatedly.
- Precomputed update positions are the strongest known way to remove repeated
  current-row lookup work, but they require a large update cache.
- Coarser CTA scheduling reduces work-counter atomic traffic, but without
  topological ordering it can be slower than the finer persistent row scheduler.
- With topological ordering, the fine-grained persistent scheduler can beat the
  current CTA-granular path despite issuing one work-counter atomic per row.
- Combining topological persistent scheduling with the cached update path is the
  current fastest measured variant on RTP.
- Future scheduling work should measure dependency spin time directly, not just
  occupancy or total resident warps.

## Useful Verification Commands

Build the benchmark:

```sh
cmake --build release --target Bench_cuda_ilu0_bench -j
```

Run the focused topological-order experiment:

```sh
release/benchmarks/cuda_ilu0_bench \
  -f ~/repo/matrix_lib/RTP_metis.bin \
  --benchmark_filter='ILU0Numeric/(persistent_spin|persistent_spin_perm|persistent_cached|persistent_cached_perm|cta_granular|cta_granular_identity|cta_granular_global|cta_granular_global_identity|cta_granular_cached)' \
  --benchmark_min_time=5s \
  --benchmark_counters_tabular=true
```

Run correctness coverage for the CUDA ILU base paths:

```sh
cmake --build release --target TEST_cuda_ilu_base_test -j
ctest --test-dir release -R cuda_ilu_base_test --output-on-failure
```
