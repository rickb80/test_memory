# `cpu_metadata` Algorithm Overview

The goal of `cpu_metadata` is to produce, for each **active instance**, the metadata that
`cpu_fill` will later use to gather ops into the correct sorted output positions.
An *instance* is a 4M-op slice of one memory region (ROM / INPUT / RAM).
An *active* instance is one assigned to this worker by the bitmask.

---

## Pass 1 — Compact addresses + coarse histogram (lines 290–333)

- **Convert** every raw hardware address to a *compact* index
  (`compact = (raw − region_base) >> 3`, packed across all three regions → 0…96M range).
  Store the compact key alongside each op for reuse in later passes.
- **Build** a per-thread coarse histogram: `COARSE_BINS = 6145` bins,
  each covering `2^14 = 16 384` consecutive compact addresses.
  Threads work on whole input-file chunks (round-robin), keeping writes local.
- **Merge** the per-thread histograms into one global coarse histogram.
- **Prefix-sum** the coarse histogram → `coarse_prefix[b]` = total ops in
  compact addresses `[0, b × 2^14)`.

---

## Post-1 — Region counts, instance counts, active instances, approximate ranges (lines 336–418)

- **Region op counts**: read directly from `coarse_prefix` at the ROM/INPUT/RAM
  boundary bins (no second scan needed). This works exactly because each region
  size (16M, 16M, 64M) is a multiple of the bin size 2^14, so region boundaries
  fall precisely on bin boundaries.
- **Instance counts**: `num_inst[r] = ceil(region_n_ops[r] / 4M)`.
- **Active instances**: built from the pre-computed bitmask; `pick_active_instances()`
  fills `active_local_ids[]`. *Local* here means local to the region — each region
  numbers its instances from 0 independently, so `lid = 2` in RAM means "3rd RAM
  instance", not 3rd instance globally. The global ID is recovered as
  `gid = gid_base + lid` when needed.
- **Approximate address range** for each active instance:
  a) The instance occupies sorted positions `[pos_start, pos_end)` globally.
     Because the first instance overlaps the previous one by one entry (*halo*),
     the starting position used is `pos_start − 1` (except for `lid == 0`).
  b) Linear-scan `coarse_prefix` to find the coarse bins that straddle
     `pos_start − 1` and `pos_end`.
  c) Expand by ±1 bin as a safety margin against bin-boundary precision loss.
  d) The resulting compact range `[approx_first, approx_last]` is a guaranteed
     superset of the instance's true address range.

---

## Pass 2 — Per-instance fine histograms (lines 421–467)

- **Quick-reject bitmask**: mark every coarse bin touched by at least one active
  instance's approximate range.
- **Parallel scan** of all ops: skip any op whose coarse bin is not marked,
  then check it against each active instance's approximate range and increment
  that instance's fine histogram (one entry per compact address).
  Each thread builds private histograms; a critical section merges them at the end.
  Result: `inst_hist[i][a]` = number of ops at compact address
  `approx_first[i] + a`.

---

## Post-3 — Exact boundaries + `addr_offsets` (lines 470–606)

For each active instance:

### 3a — Local prefix sum (lines 495–497)
Prefix-sum `inst_hist[i]` → `local_prefix[k]` = ops at compact addresses
`[approx_first, approx_first + k)`.

### 3b — Find exact `first_addr` (lines 501–512)
Binary-search `local_prefix` for the last index where the cumulative count is
`≤ halo_base − global_prefix_at_approx_first`.
This is the address whose ops include the halo entry.

### 3c — Find exact `last_addr` (lines 515–526)
Binary-search forward from `first_addr` for the last index where the cumulative
count is `< inst_end − global_prefix_at_approx_first`.
This is the last address that contributes at least one op to this instance.

### 3d — Trim trailing empty addresses (lines 528–536)
After step 3c, `last_addr` is often correct, but not always minimal: prefix-sum
searches can still stop at an address inside a flat (zero-count) tail.
This step removes that tail and keeps only addresses that actually have ops.

Equivalent view:
- Let `max_pref = local_prefix[last_addr_local_idx + 1]`.
- Any address where prefix is still `< max_pref` has at least one op before or at it.
- Once prefix becomes `== max_pref`, we are in trailing empty addresses.
- So we binary-search the rightmost index with prefix `< max_pref`.

Mini-example:
- Counts in candidate interval: `[3, 2, 0, 0]`
- Prefix: `[0, 3, 5, 5, 5]`
- Step 3c may return the last index (`3`) because prefix constraints still hold.
- Step 3d trims back to index `1`, which is the true last occupied address.

### 3e — Compute `addr_offsets` (lines 558–597)
For each address in `[first_addr, last_addr]`, `addr_offsets[k]` is the
**write position** (1-based, relative to the instance's output base) where
the *first* op of address `first_addr + k` should land.
- `addr_offsets[0]` = 0 if a halo entry exists, 1 otherwise.
- `addr_offsets[k]` = (global cumulative ops through address `first_addr + k − 1`)
  minus `(base_pos − 1)`.
  
`cpu_fill` increments `addr_offsets[k]` atomically as it writes each op,
so it naturally scatters ops to consecutive positions within each address slot.

---

## Pass 3 — FML (First / Middle / Last) counts per chunk (lines 609–648)

- Re-scan all ops with a coarse-bin quick-reject (now using exact ranges, much tighter).
- For each op that falls in an active instance's `[first_addr, last_addr]`, classify it:
  - **0 (First)** — compact address == `first_addr`
  - **1 (Middle)** — strictly interior address
  - **2 (Last)** — compact address == `last_addr`
- Accumulate `fml[instance][chunk][category]`.

The FML split matters because `first_addr` and `last_addr` each span multiple input
files (chunks) and only a sub-range of their occurrences belong to this instance.
Middle addresses are entirely owned by the instance so every occurrence is included.

---

## Post-4 — Build `InstanceMeta` (lines 652–760)

Using the FML counts, compute for each active instance:

### 4a — `first_addr_total_skip` and `last_addr_total_include`
- `first_addr_total_skip`: how many first-address occurrences (across all chunks) must
  be **skipped** before the instance's window starts.
  Formula: `halo_base − global_prefix_at_first_addr`.
- `last_addr_total_include`: how many last-address occurrences must be **included**
  after the instance's window begins at `last_addr`.
  For the single-address case both endpoints coincide and the counts are combined.

### 4b — `first_addr_chunk` / `first_addr_skip` (lines 710–722)
Walk the per-chunk `fml[i][c][0]` counts until the cumulative count exceeds
`first_addr_total_skip`. That chunk index is `first_addr_chunk`; the remainder
within that chunk is `first_addr_skip`.

### 4c — `last_addr_chunk` / `last_addr_include` (lines 724–737)
Same scan but over `fml[i][c][2]` (or `[0]` for the single-address case) until
the cumulative count reaches `last_addr_total_include`.

### 4d — `nops_per_chunk` with chunk elimination (lines 739–755)
For each chunk, set `nops_per_chunk[c]` = total ops (F+M+L) belonging to this
instance in chunk `c`, but **only** for chunks that are actually needed:
- A chunk with only middle ops is always needed.
- A chunk with first-address ops is needed only from `first_addr_chunk` onward.
- A chunk with last-address ops is needed only up to `last_addr_chunk`.
- Chunks outside that window are marked zero (skipped entirely in `cpu_fill`).

---

## Summary of data products

| Field | Computed in | Used by |
|---|---|---|
| `compact_keys[]` | Pass 1 | Passes 2, 3 |
| `coarse_prefix[]` | Pass 1 | Post-1, Post-3, Post-4 |
| `inst_info[].approx_{first,last}` | Post-1 | Pass 2, Post-3 |
| `inst_hist[i][]` | Pass 2 | Post-3 |
| `metas[i].{first_addr, last_addr}` | Post-3 | cpu_fill |
| `metas[i].addr_offsets[]` | Post-3 | cpu_fill |
| `fml[i][c][cat]` | Pass 3 | Post-4 |
| `metas[i].{first,last}_addr_{chunk,skip/include}` | Post-4 | cpu_fill |
| `metas[i].nops_per_chunk[]` | Post-4 | cpu_fill |
