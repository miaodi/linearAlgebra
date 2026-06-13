# CUDA ILU(0) Row Matching Analysis

This note records the current understanding of CUDA ILU(0) numeric row matching
between a current row and a reference `U` row during left-looking updates. It is
intended as a durable engineering note for future matcher experiments in this
C++20/CUDA sparse linear algebra repository.

Related context is in [`cuda_ilu0_numeric_optimizations.md`](cuda_ilu0_numeric_optimizations.md).

## Problem Summary

During a left-looking ILU(0) update, the numeric kernel must match columns from a
reference `U` row against the current row. The existing CTA-granular global path
does this per update by binary-searching the current row for each reference-row
column.

The cached CTA path, `ILU0Numeric/cta_granular_cached`, removes that runtime row
matching by using precomputed update positions:

```text
lower_row_ptr, update_ptr, update_jpos, update_pos
```

This is faster, but the cache is large. On RTP, it stores about `124.0M` cached
updates and uses about `1.01 GiB` of device memory.

## RTP Benchmark Snapshot

These are measured facts from the RTP benchmark runs.

| Variant | Approximate time | Row-matching behavior |
| --- | ---: | --- |
| `cta_granular_global` | 24.9 ms | Binary-search current row from global memory. |
| `cta_granular` | 22.3 ms | Binary search with shared current-row staging. |
| cuSPARSE `USE_LEVEL` | 21.8 ms | Library reference path. |
| `cta_granular_cached` | 18.5 ms | Uses precomputed update positions. |
| `persistent_cached_perm` | 17.6 ms | Persistent topological scheduler plus update cache. |
| `cta_granular_cached_identity` | 35.7 ms | Cached matching with identity row order. |

The identity cached result shows that row issue order still matters even when row
matching is removed from the numeric kernel.

## NCU Snapshot

These are measured NCU facts from focused RTP runs.

| Metric | `cta_granular_global` | cuSPARSE `USE_LEVEL` | `cta_granular_cached` |
| --- | ---: | ---: | ---: |
| Kernel time | 27.609 ms | 23.792 ms | 19.486 ms |
| L2 sectors | 604.6M | 421.8M | 366.2M |
| L2 read sectors | 496.1M | 325.8M | 262.8M |
| SM instructions | 2.892B | 2.898B | 1.569B |
| Thread instructions | 54.779B | 63.884B | 31.244B |
| Long-scoreboard not-issued | 1.688M | 1.160M | 1.317M |

The cached path cuts both instruction count and L2 read traffic relative to the
global binary-search path. cuSPARSE sits between the global and cached paths in
time and L2 traffic, but it does not look like a full update-position cache.

## cuSPARSE Cache Hypothesis

This section is postulation, not a measured implementation fact.

cuSPARSE likely does not use a full update-position cache like
`cta_granular_cached`. Its observed buffer was about `22 MB`, and NCU showed very
low shared-memory use. That is inconsistent with storing all RTP update matches,
which takes about `1.01 GiB` in the in-repo cached path.

The more plausible hypothesis is that cuSPARSE uses a hybrid sorted-intersection
matcher. Candidate techniques include monotone cursor `lower_bound`,
galloping/exponential search, warp-cooperative search over sampled row positions,
merge-path or co-rank partitioning for similarly sized rows, row min/max range
rejection, and register/ballot/shuffle matching for small rows.

Reducing the search window alone may not help if the implementation still follows
the same serial dependent global-load chain. The GPU benefit must come from
reducing repeated current-row probes, reducing serial dependency depth, or both.

## One-Row Row-Matching Benchmark

A synthetic CUDA microbenchmark was created at
`benchmarks/cuda_row_match_bench.cu`. It simulates exactly one current-row and
reference-row pair, then repeats that same pair across many warps only to get
measurable timing.

The benchmark compares:

| Benchmark family | Meaning |
| --- | --- |
| `RowMatch/binary_global/...` | Binary-search current row from global memory. |
| `RowMatch/merge_tiled/...` | Current tile all-to-all merge-style matcher. |

The source should keep a short comment on each scenario explaining what it tests.

| Scenario | Intent |
| --- | --- |
| Tiny reference rows | Measures overhead when there is little work per row. |
| Short/medium reference subsets | Tests common subset-like reference access. |
| Sparse mixed hits | Mixes found and missing columns across a wider current row. |
| Interleaved misses | Tests miss-heavy probes interleaved with possible hits. |
| Disjoint-after misses | Tests sorted references beyond the current-row range. |
| `one_to_one` | Tests identical current and reference rows. |

Initial result: the current `merge_tiled` implementation loses to
`binary_global` in every tested one-row scenario.

## One-To-One Results

The one-to-one cases are especially important because the current and reference
rows are identical.

| Columns | `binary_global` | `merge_tiled` |
| ---: | ---: | ---: |
| 64 | ~31 us | ~75 us |
| 256 | ~106 us | ~266 us |
| 1024 | ~466 us | ~1030 us |
| 2048 | ~998 us | ~2046 us |

Interpretation: the current merge-tiled strategy is not a compelling replacement
for binary search. If it loses even when the two rows are identical, there is no
reason to use this implementation as-is. It likely compares warp-sized tiles in a
heavier way than a proper merge-path or co-rank sorted intersection.

## Galloping Search

Galloping search, also called exponential search, searches a sorted array from a
known cursor. Starting at the previous cursor, it jumps forward exponentially by
`1, 2, 4, 8, ...` until it brackets the target. It then binary-searches only
inside that bracket.

For row matching, reference columns arrive in sorted order, so the current-row
cursor never has to move backward. This can reduce repeated full-row binary
searches and avoid repeatedly probing the same current-row prefix.

The larger GPU benefit may be fewer dependent global-memory probes, not only
fewer comparisons. A cursor/gallop path should therefore be evaluated with both
timing and memory-latency metrics.

## Useful Commands

Build the one-row benchmark:

```sh
cmake --build release --target Bench_cuda_row_match_bench -j
```

Run only the one-to-one scenarios:

```sh
release/benchmarks/cuda_row_match_bench \
  --benchmark_filter='RowMatch/.*/one_to_one' \
  --benchmark_min_time=1x \
  --benchmark_repetitions=1 \
  --benchmark_counters_tabular=true
```

Run all row-matching microbenchmarks:

```sh
release/benchmarks/cuda_row_match_bench \
  --benchmark_min_time=1x \
  --benchmark_repetitions=1 \
  --benchmark_counters_tabular=true
```

## Recommended Next Experiments

| Experiment | Purpose |
| --- | --- |
| Add `binary_cursor_gallop` to `benchmarks/cuda_row_match_bench.cu`. | Test monotone cursor plus exponential bracketing. |
| Add a proper merge-path/co-rank one-row benchmark. | Compare against a real sorted-intersection design, not tile all-to-all merge. |
| Add a warp-cooperative stride/sample search benchmark. | Test whether cooperative probes reduce dependency depth. |
| Compare NCU long-scoreboard and L2 sectors for microbenchmarks and ILU variants. | Connect synthetic matcher behavior to full ILU behavior. |
| Test row-size based hybrid dispatch. | Use different matchers for small, medium, and large row pairs. |

## Current Takeaways

Measured facts show that precomputed update positions are fast but memory-heavy,
and that row order remains important after matching is cached. The current
merge-tiled matcher is not competitive in one-row tests. The next useful work is
to test cursor/gallop, real merge-path/co-rank, and warp-cooperative search
variants before changing the production ILU numeric matcher.
