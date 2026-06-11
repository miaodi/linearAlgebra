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
| Level CTA | `sparse_mat_op/cuda/ilu/ilu_numeric_level_cta.cu` | Resident CTAs claim packed row bundles from `next_cta`. |
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
| `level_cta` | ~24.5 ms | Topological `level_perm` order, 8 rows per CTA task. |
| `level_cta_identity` | ~48.2 ms | Identity row order experiment, CTA chunks `[0..7]`, `[8..15]`, ... |
| `persistent_spin_perm` | ~18.5 ms | Persistent scheduler with topological `level_perm` row order. |
| `persistent_cached_perm` | ~17.6 ms | Topological persistent scheduler plus lower-only update cache. |
| cuSPARSE `csrilu02` | ~21.8 ms | Reference library implementation on the same matrix. |

The most important recent experiment compared these variants:

```sh
release/benchmarks/cuda_ilu0_bench \
  -f ~/repo/matrix_lib/RTP_metis.bin \
  --benchmark_filter='ILU0Numeric/(persistent_spin|persistent_spin_perm|persistent_cached|persistent_cached_perm|level_cta|level_cta_identity)' \
  --benchmark_min_time=5s \
  --benchmark_counters_tabular=true
```

Observed result:

| Benchmark | Time |
| --- | ---: |
| `ILU0Numeric/level_cta/real_time` | 24.5 ms |
| `ILU0Numeric/level_cta_identity/real_time` | 48.2 ms |
| `ILU0Numeric/persistent_spin/real_time` | 34.7 ms |
| `ILU0Numeric/persistent_spin_perm/real_time` | 18.5 ms |
| `ILU0Numeric/persistent_cached/real_time` | 30.6 ms |
| `ILU0Numeric/persistent_cached_perm/real_time` | 17.6 ms |

This experiment shows that topological row ordering is a major part of the
`level_cta` speedup. It also shows that the fine-grained persistent scheduler
benefits even more from using the same topological order.

## Optimization: Topological Row Order

The level CTA schedule is built by `BuildILULevelCtaSchedule`. It packs rows in
`level_perm` order, not raw row-id order:

```cpp
const COLTYPE row = checkedZeroBasedIndex(level_perm[pos], base, n, "level row");
schedule.cta_rows.push_back(row);
```

The level CTA numeric kernel still uses row-level dependency waiting through
`row_done`. The CTA predecessor/successor arrays are built and uploaded, but the
current numeric kernel does not use them for runtime dependency scheduling.

What matters today is the row issue order:

| Schedule | Row order | Result on RTP |
| --- | --- | ---: |
| Topological level CTA | `level_perm[0..n)` | ~24.5 ms |
| Identity level CTA | `0, 1, 2, ...` | ~48.2 ms |

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
level CTA path. Any new scheduler should measure how much time is spent waiting
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
current level CTA path on RTP, despite doing one work-counter atomic per row.

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
- The persistent and level CTA paths use shared staging only when
  `row_end - row_begin <= kSharedRowColumnsPerWarp`.
- Larger rows fall back to global lookup to avoid overflowing the per-warp shared
  buffer.

Takeaway: shared memory helps the binary-search path when the current row is
large enough to make repeated global-memory searches expensive, but still small
enough to fit in the per-warp shared row cache.

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

## Optimization: Level CTA Granularity

The level CTA kernel assigns one CTA task to a block. Each task contains up to 8
rows by default, one row per warp:

```text
block claims CTA task -> warp 0 handles row slot 0, ..., warp 7 handles row slot 7
```

Compared with persistent row scheduling:

| Path | Work claim granularity | Global scheduling atomics |
| --- | --- | --- |
| Persistent spin | 1 row per warp claim | About one per row. |
| Level CTA | Up to 8 rows per CTA claim | About one per 8 rows. |

This coarser granularity reduces scheduler atomic traffic, but the identity
experiment shows it is not the main reason `level_cta` is fast. The topological
`level_perm` ordering is essential on RTP.

## Experiments To Avoid Repeating Blindly

| Experiment | Result | Decision |
| --- | ---: | --- |
| Row fetch chunking | ~337 ms | Rejected due to severe regression. |
| 32-warps/SM cap | ~54.6 ms | Rejected for the tuned persistent path. |
| Forced global lookup | ~53.0 ms | Rejected; shared row-column staging remains useful. |
| Busy spin without `__nanosleep` | ~49.5 ms | Rejected; delayed sleep performed better. |
| Forced 128/256 block size | Neutral | Not retained as a tuning knob. |
| Identity level CTA order | ~48.2 ms | Rejected; topological ordering matters. |

## Current Interpretation

The best current in-repo path for RTP is `persistent_cached_perm` at roughly
`17.6 ms`, ahead of the measured cuSPARSE `csrilu02` result of roughly
`21.8 ms`.

The main lessons are:

- Topological row order is a major performance feature, not just a correctness
  convenience.
- Topological order helps both coarse level CTA scheduling and fine-grained
  persistent row scheduling.
- `diag_inv` is a major row-done path optimization.
- Shared row-column staging helps binary search when the current row fits the
  shared cache and is searched repeatedly.
- Coarser CTA scheduling reduces work-counter atomic traffic, but without
  topological ordering it can be slower than the finer persistent row scheduler.
- With topological ordering, the fine-grained persistent scheduler can beat the
  current level CTA path despite issuing one work-counter atomic per row.
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
  --benchmark_filter='ILU0Numeric/(persistent_spin|persistent_spin_perm|persistent_cached|persistent_cached_perm|level_cta|level_cta_identity)' \
  --benchmark_min_time=5s \
  --benchmark_counters_tabular=true
```

Run correctness coverage for the CUDA ILU base paths:

```sh
cmake --build release --target TEST_cuda_ilu_base_test -j
ctest --test-dir release -R cuda_ilu_base_test --output-on-failure
```
