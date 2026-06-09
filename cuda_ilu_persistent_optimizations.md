# CUDA ILU Persistent Optimization Notes

## Context

This note summarizes work after the committed checkpoint
`e5e712f7458ced81c9b1075961f70ee5442eb86a`
(`feat(cuda): add persistent ILU numeric path`) on the CUDA ILU(0) numeric
factorization non-cached persistent spin path.

Baseline matrix: `/home/miaodi/repo/matrix_lib/RTP_metis.bin`.

| Statistic | Value |
| --- | ---: |
| `n` | 1.949482M |
| `nnz(A)` | 33.545908M |
| `nnz(LU0 pattern)` | 33.545908M |
| Levels | 314 |
| Strict lower nnz | 14.847128M |
| Cached updates | 124.009491M |

## Benchmark Summary

Approximate benchmark means on RTP:

| Stage | `ours_persistent_spin` | `ours_binary_shared` | `cuSPARSE` | Notes |
| --- | ---: | ---: | ---: | --- |
| Original persistent path after `e5e712f` | 50.5 ms | 58.0 ms | -- | Acquire/release row-ready polling. |
| Wait-path tuning | 45.9 ms | 58.0 ms | -- | Lower polling and status-check overhead. |
| Diagonal inverse optimization | 34.6 ms | 58.0 ms | 21.8 ms | Dependents multiply by stored `diag_inv[k]`. |

The current persistent path is faster than the non-persistent binary shared path
on this matrix, but still trails cuSPARSE by about 12.8 ms.

## Current Persistent Algorithm Overview

- Resident warps pull rows from a monotonic global row counter.
- Before row `i` uses a lower dependency row `k`, the warp waits until
  `row_done[k]` is published.
- Monotonic row assignment ensures lower-index dependencies are scheduled before
  dependent rows can wait on them.
- The path remains non-cached: row updates use the existing binary-search row
  lookup, with shared row-column staging for rows that fit the shared buffer and
  global lookup otherwise.
- Each completed row checks the diagonal, stores `diag_inv[row] = 1 / U(row,row)`,
  then publishes `row_done[row]`.

Primary files:

- `sparse_mat_op/cuda/ilu/ilu_numeric_workqueue.cu`
- `sparse_mat_op/cuda/ilu/ilu_numeric_workqueue.cuh`
- `benchmarks/cuda_ilu0_bench.cpp`
- `tests/cuda_ilu_base_test.cpp`

## Optimizations Kept

- Lane 0 alone polls `row_done[k]`; the result is broadcast to the warp.
- Status reads use relaxed `cuda::atomic_ref` loads instead of read-modify-write
  atomics.
- The normal path no longer reloads status after every dependency wait.
- `__nanosleep` is delayed until after 64 failed polls.
- Row bounds are loaded once per row.
- Rows with no lower dependencies skip row work and the row-update fence.
- The persistent API takes a caller-owned `d_diag_inv` scratch buffer of size
  `n` and reuses it during the numeric factorization.
- Dependents multiply by `diag_inv[k]` instead of dividing by `U(k,k)`.

## Rejected Experiments

| Experiment | Result | Decision |
| --- | ---: | --- |
| Row fetch chunking | ~337 ms | Rejected due to severe regression. |
| 32-warps/SM cap | ~54.6 ms | Rejected; worse than the tuned persistent path. |
| Forced global lookup | ~53.0 ms | Rejected; shared row-column staging remains useful. |
| Busy spin without `__nanosleep` | ~49.5 ms | Rejected; delayed sleep performed better. |
| Forced 128/256 block size | Neutral | Not retained as a required tuning knob. |

## Correctness and Numerical Notes

- `d_diag_inv` must be allocated by the caller with length `n`.
- A row stores its diagonal inverse before publishing `row_done[row]`.
- Dependents use acquire polling on `row_done[k]`, so they observe the published
  row state and `diag_inv[k]` after the release store.
- Replacing division by multiplication with a stored reciprocal changes floating
  point roundoff slightly. The persistent test tolerance was relaxed from
  `1e-10` to `1e-8`.
- Verification already run after the diagonal-inverse change:
  - `ctest --test-dir release -R cuda_ilu_base_test --output-on-failure` passed.
  - `git diff --check` passed.

## Remaining Gap and Likely Next Steps

The current RTP result is roughly `34.6 ms` versus cuSPARSE at roughly
`21.8 ms`. Likely next steps are hypotheses until profiler data confirms them:

- Profile the persistent kernel to separate dependency-wait time from row-update
  work.
- Investigate scheduling or hybrid level/persistent schemes that reduce spin
  overhead without losing enough parallelism to regress.
- Reduce binary-search lookup and global-memory traffic in the update path.
- Revisit launch configuration only with profiler evidence; fixed block sizes
  were neutral and a hard resident-warp cap regressed.
