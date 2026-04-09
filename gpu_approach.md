# `gpu_metadata` Algorithm Overview

The goal of `gpu_metadata` is to produce, for each **active instance**, the same
`InstanceMeta` that `cpu_fill` will later use to gather ops into the correct sorted
output positions. The GPU approach uses a **full 96M-entry histogram** with prefix sum —
feasible on GPU thanks to ~900 GB/s HBM bandwidth and thousands of threads hiding
atomic-add latency.

---

## Phase 1 — H2D + compact addresses + histogram (lines 840–870)

- **Pipelined** across 4 CUDA streams: for each chunk, queue an async H2D copy
  followed by `shift_and_histogram_kernel` on the same stream.
  Round-robin assignment (`stream = c % 4`) allows up to 4 chunks in flight.
- **`shift_and_histogram_kernel`** (lines 78–94): each thread processes ops in a
  stride loop:
  a) Read `ops[i]` (raw hardware address).
  b) Compute compact address in-place using the 3-region mapping.
  c) Write compact address back to `ops[i]`.
  d) `atomicAdd(&hist[compact], 1)`.
- The histogram `d_hist[96M]` accumulates across all chunk launches
  (zeroed once before the loop via `cudaMemset`).
- The last chunk is processed separately to measure "last chunk to ready" latency.

**Key GPU advantage**: the full 96M-entry histogram (384 MB) works because
GPU HBM bandwidth absorbs the random atomic writes. On CPU this would cause
constant cache misses (384 MB >> L3 cache).

---

## Phase 2 — Prefix sum (lines 872–878)

- **CUB `ExclusiveSum`** over `d_hist[96M]` → `d_prefix[96M+1]`.
- `d_prefix[addr]` = total number of ops with compact address `< addr`.
- `d_prefix[N_ADDR]` is set to `num_ops` (sentinel).

This is the equivalent of the CPU version's coarse prefix sum + per-instance
fine prefix sums, but computed in one shot over the full address space.

---

## Phase 3 — Region counts + active instances (lines 880–913)

- **Region boundaries**: read `d_prefix[N_ADDR_ROM]` and
  `d_prefix[N_ADDR_ROM + N_ADDR_INPUT]` via D2H copies.
  These give exact per-region op counts (no approximation needed — the full
  prefix sum is available).
- **Instance counts**: `num_inst[r] = ceil(region_n_ops[r] / INSTANCE_SIZE)`.
- **`pick_active_instances()`**: same bitmask logic as the CPU version.
  Uploads `active_local_ids[]` to GPU via `cudaMemcpy`.

**CPU difference**: the CPU version derives region counts from the coarse prefix
sum at bin boundaries. The GPU reads exact values from the full prefix sum.

---

## Phase 4 — Instance boundaries (lines 915–926)

- **`instance_boundaries_kernel`** (lines 109–170): launched as `<<<1, na>>>`
  per region (one thread per active instance in a single block).
- Each thread binary-searches `d_prefix[]` to find:
  a) `first_addr`: largest address where `prefix[addr] <= inst_start`
     (accounts for halo: `inst_start = base_pos - 1` for non-first instances).
  b) `last_addr`: largest address where `prefix[addr] < inst_end`.
  c) Trailing-empty trim: largest address where `prefix[addr] < prefix[last+1]`.
- Thread 0 computes `offset_starts[]` (serial prefix sum of per-instance address
  counts) after a `__syncthreads()`.

**CPU difference**: the CPU does the same binary searches, but on per-instance
fine histogram prefix sums (which only cover the approximate address range).
The GPU searches the full prefix sum directly.

---

## Phase 5 — FML counts (lines 928–935)

- **`chunk_fml_count_kernel`** (lines 181–240): single launch over all `num_ops`
  entries, processing all active instances in one pass.
- **Shared memory**: caches `active_first[]` and `active_last[]` for all instances.
- Each thread:
  a) Reads `ops[i]` (now compact), determines its chunk via `chunk_offsets`.
  b) Iterates active instances, early-exits with `__all_sync(addr < fa)`.
  c) Categorizes as first (0), middle (1), or last (2).
  d) **Warp-level optimization**: if entire warp is in the same chunk,
     uses `__ballot_sync` + `__popc` to reduce 32 per-thread atomics to
     one per category. Otherwise falls back to per-thread `atomicAdd`.
- Output: `d_fml[instance][chunk][category]`.

**CPU difference**: the CPU scans `compact_keys[]` in a separate Pass 3 with
a coarse-bin quick-reject bitmap. The GPU reads `d_ops[]` (already compact on
device) and uses warp-level intrinsics for efficient reduction.

---

## Phase 6 — Build metas + addr_offsets (lines 943–978)

Two kernels launched on **separate streams** for overlap:

### 6a — `build_metas_kernel` (meta_stream) (lines 255–456)
One block (256 threads) per active instance. Six phases inside each block:

1. **Compact non-empty chunks**: parallel prefix sum in shared memory to build
   a list of chunks where FML total > 0.
2. **Skip/include totals**: thread 0 computes `first_addr_total_skip` and
   `last_addr_total_include` from `d_prefix[]` and instance boundaries.
3. **Find `first_addr_chunk`**: parallel prefix sum over per-chunk first-address
   counts, then thread 0 binary-searches for the threshold crossing.
4. **Find `last_addr_chunk`**: same approach for last-address (or first-address
   in single-addr case).
5. **`nops_per_chunk`** with chunk elimination: each thread writes
   `nops[c] = F+M+L` only for chunks that are actually needed
   (middle always needed; first/last only within the chunk window).
6. **Write scalars**: `[fa_chunk, fa_skip, la_chunk, la_include]` to
   `d_meta_scalars[instance * 4]`.

### 6b — `compute_addr_offsets_kernel` (d2h_stream) (lines 471–499)
One block (1024 threads) per active instance.
- `addr_offsets[0]` = 0 (halo exists) or 1 (no halo).
- `addr_offsets[j]` = `prefix[first_addr + j] - (base_pos - 1)` for j > 0.

### 6c — Async D2H transfers
- `meta_scalars` and `result_nops` copied on `meta_stream`.
- `addr_offsets` copied on `d2h_stream`.
- Both streams synchronized before populating `InstanceMeta` on host.

**CPU difference**: the CPU computes addr_offsets from the per-instance fine
histogram prefix sums (serial loop per instance). The GPU reads the full
`d_prefix[]` directly with 1024 threads per instance.

---

## Phase 7 — Populate InstanceMeta (lines 982–996)

Host-side loop: for each active instance, assemble the `InstanceMeta` struct from:
- `h_meta_scalars[ai*4]`: `[fa_chunk, fa_skip, la_chunk, la_include]`
- `h_active_first[ai]` / `h_active_last[ai]`: expand to raw hardware addresses
- `h_result_nops[ai * num_chunks]`: span view for `nops_per_chunk`
- `h_offsets_buf[offset_starts[ai]]`: span view for `addr_offsets`

---

## Pipeline parallelism

| Stream | Phase 1 | Phase 5 | Phase 6 |
|--------|---------|---------|---------|
| streams[0..3] | H2D + histogram (round-robin) | — | — |
| default | — | FML kernel | — |
| meta_stream | — | — | build_metas → D2H scalars + nops |
| d2h_stream | — | — | addr_offsets → D2H offsets |

The H2D transfer overlaps with histogram computation (Phase 1).
The build_metas and addr_offsets kernels overlap with each other (Phase 6).

---

## Summary: GPU vs CPU approach

| Aspect | GPU | CPU |
|--------|-----|-----|
| Histogram | Full 96M entries, atomicAdd | Coarse (6K bins) + per-instance fine (~100K each) |
| Prefix sum | CUB on full 96M entries | Coarse prefix + local prefix per instance |
| Boundaries | Binary search on full prefix | Binary search on fine histogram prefix |
| FML | Single-pass kernel, warp ballot | Separate Pass 3, coarse-bin quick-reject |
| Build metas | GPU kernel (256 threads/instance) | Serial per instance |
| addr_offsets | GPU kernel (1024 threads/instance) | Serial prefix sum per instance |
| Overlap | H2D + histogram; metas + offsets | None (sequential passes) |
| **Bottleneck** | PCIe H2D transfer (97% of time) | Memory bandwidth (3 × 2.1 GB scans) |

---

## Summary of data products

| Field | Computed in | Used by |
|---|---|---|
| `d_ops[]` (compact in-place) | Phase 1 | Phase 5 |
| `d_hist[96M]` | Phase 1 | Phase 2 |
| `d_prefix[96M+1]` | Phase 2 | Phases 3, 4, 6a, 6b |
| `d_active_first/last[]` | Phase 4 | Phases 5, 6a, 6b |
| `d_fml[inst][chunk][cat]` | Phase 5 | Phase 6a |
| `d_result_nops[inst][chunk]` | Phase 6a | cpu_fill (via D2H) |
| `d_meta_scalars[inst][4]` | Phase 6a | Phase 7 (via D2H) |
| `d_addr_offsets[]` | Phase 6b | cpu_fill (via D2H) |
| `metas[].{first,last}_addr_{chunk,skip/include}` | Phase 7 | cpu_fill |
| `metas[].nops_per_chunk` | Phase 7 | cpu_fill |
| `metas[].addr_offsets` | Phase 7 | cpu_fill |
