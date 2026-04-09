# main_real.cu Documentation

## Overview

`main_real.cu` implements a GPU-accelerated pair sorting algorithm that distributes key-value pairs across multiple "instances" based on hardware memory addresses. It processes real trace data from files, handles three distinct memory regions (ROM, INPUT, RAM), and uses CUDA for GPU computation with OpenMP for CPU parallelism.

## Memory Regions

The system handles three hardware memory regions mapped to a compact address space:

| Region | Raw Address Range | Compact Range | Max Addresses | Max Instances |
|--------|------------------|---------------|---------------|---------------|
| ROM    | 0x80000000+ | 0 – 16M | 16M | 32 |
| INPUT  | 0x90000000+ | 16M – 32M | 16M | 32 |
| RAM    | 0xA0000000+ | 32M – 96M | 64M | 256 |

Address conversion: Raw addresses are 8-byte aligned, so `compact = (raw - base) >> 3`.

## Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `N_ADDR` | 96M | Total compact addresses (ROM + INPUT + RAM) |
| `INSTANCE_SIZE` | 2²² (4M) | Entries per instance |
| `MAX_INSTANCES` | 320 | Maximum total instances |
| `N_WORKERS` | 16 | Worker count for distributed selection |
| `MAX_ACTIVE` | 20 | Maximum active instances per worker |
| `MAX_CHUNKS` | 4096 | Maximum file chunks |

## GPU Kernels

### `shift_and_histogram_kernel`
Converts raw hardware addresses to compact addresses in-place while simultaneously building a histogram. Combines two operations for efficiency.

### `instance_boundaries_kernel`
Binary searches the prefix sum array to find the first and last compact addresses for each active instance within a region. Also computes `offset_starts` for the address offsets buffer.

### `chunk_fml_count_kernel`
For each active instance and chunk, counts:
- **First**: entries matching the instance's first address
- **Middle**: entries between first and last addresses
- **Last**: entries matching the instance's last address

Uses warp-level optimizations (`__ballot_sync`, `__all_sync`, `__any_sync`) for efficiency.

### `build_metas_kernel`
Processes FML counts to determine:
- Which chunks contain relevant data (chunk elimination)
- How many entries to skip at the first address boundary
- How many entries to include at the last address boundary

### `compute_addr_offsets_kernel`
Computes per-address write offsets within each instance using prefix sums, enabling correct scatter positioning during CPU fill.

## PairSortGPU Class

### Memory Layout
- **GPU**: Operations, histogram, prefix sums, per-region active instance data, FML counts, metadata, chunk offsets
- **Pinned Host**: Operations buffer, offset buffer, result arrays, metadata scalars
- **Regular Host**: Values, per-region output/reference arrays (6 pairs total)

### Workflow

```
1. generate(block_number)   → Load trace files from data/<block>/mem_addr_XXXX.bin
         ↓
2. create_active_mask()     → Select instances based on worker_id % N_WORKERS
         ↓
3. gpu_metadata()           → GPU pipeline:
   ├── H2D + shift + histogram (streamed per chunk)
   ├── CUB prefix sum
   ├── Compute per-region instance counts
   ├── Instance boundaries kernel (per region)
   ├── FML count kernel (single pass, all regions)
   ├── Build metas kernel (per region)
   └── Address offsets kernel + D2H transfers
         ↓
4. cpu_fill()               → Scatter pairs to per-region output arrays (OpenMP)
         ↓
5. verify() [optional]      → Compare against reference stable sort
```

### Key Data Structures

**`InstanceMeta`**: Per-instance metadata containing:
- `inst_id`: Local instance ID within region
- `type`: Region identifier (ROM/INPUT/RAM)
- `first_addr`, `last_addr`: Raw hardware address range
- `first_addr_chunk`, `first_addr_skip`: Starting chunk and skip count
- `last_addr_chunk`, `last_addr_include`: Ending chunk and include count
- `nops_per_chunk`: Span into pinned result buffer
- `addr_offsets`: Span into pinned offsets buffer

## Pipeline Strategy

1. **Streaming H2D + Histogram**: Uses 4 CUDA streams to overlap data transfer with address conversion and histogram computation
2. **Per-Region Processing**: Instance boundaries and metadata built separately for each memory region
3. **Overlapped Kernels**: `build_metas_kernel` and `compute_addr_offsets_kernel` run on separate streams
4. **Async D2H**: Metadata and offset transfers queued asynchronously

## Active Instance Selection

Instances are selected based on a worker ID pattern: `instance_gid % N_WORKERS == worker_id`. This simulates a distributed system where each worker processes a subset of instances.

## CPU Fill Logic

For each active instance (parallel via OpenMP):
1. Select the correct output array based on region type
2. Iterate through chunks with non-zero expected ops
3. For each entry matching the instance's address range:
   - Apply skip/include logic for first/last address boundaries
   - Write to output position using pre-computed offsets
4. Stop when instance is full

## Usage

```bash
./pair_sort_real_gpu <block_number> [-v]
```

- `block_number`: Directory under `data/` containing trace files
- `-v`: Enable verification against reference stable sort

## Input Format

Expects binary files at `data/<block>/mem_addr_XXXX.bin` containing sequences of 32-bit raw hardware addresses.

---

# main_real_cpu.cpp Documentation

## Overview

`main_real_cpu.cpp` is the CPU-only equivalent of `main_real.cu`. It produces identical metadata (`InstanceMeta`) and output using a coarse-to-fine histogram approach instead of the GPU's full 96M-entry histogram + prefix sum. The algorithm avoids the full histogram (384 MB, far exceeds L3 cache) by using a two-level scheme: a coarse histogram to narrow address ranges, then per-instance fine histograms covering only the relevant addresses.

## Algorithm: Coarse-to-Fine Histograms

The CPU version replaces the GPU's single-pass full histogram with multiple targeted passes:

1. **Pass 1**: Compact addresses + coarse histogram (COARSE_SHIFT=14, ~6K bins)
2. **Analysis**: Derive approximate address ranges per instance from coarse prefix sum
3. **Pass 2**: Build per-instance fine histograms (only within approximate ranges, ~100K addresses per instance)
4. **Boundaries**: Find exact first/last addresses via binary search on fine histogram prefix sums, compute addr_offsets
5. **Pass 3**: FML counts (first/middle/last per chunk) using exact boundaries with coarse-bin quick-reject
6. **Build metas**: Derive skip/include thresholds and nops_per_chunk from FML counts

### Why Not a Full Histogram

The GPU builds a single 96M-entry histogram (384 MB) using atomic adds with massive memory bandwidth (~900 GB/s HBM). On CPU:
- 384 MB >> L3 cache → nearly every write is a cache miss
- Thread-local histograms: 384 MB × N_threads → impractical memory

The coarse-to-fine approach keeps all working sets cache-friendly:
- Coarse histogram: ~24 KB (fits L1)
- Per-instance fine histograms: ~400 KB each (fits L2/L3)
- Coarse-bin quick-reject bitmaps: ~768 bytes (fits L1)

## GPU vs CPU: Key Differences

| GPU Concept | CPU Equivalent |
|-------------|---------------|
| Full 96M histogram via atomicAdd | Coarse histogram (~6K bins) + per-instance fine histograms |
| CUB prefix sum on 96M entries | Coarse prefix sum (~6K) + local prefix sums per instance |
| H2D/D2H streaming (4 CUDA streams) | Eliminated — data stays in RAM |
| Warp-level ballot/popc for FML | Scalar loop with coarse-bin quick-reject |
| instance_boundaries_kernel | Binary search on fine histogram prefix sums |
| compute_addr_offsets_kernel | Serial prefix sum per instance |

## PairSortCPU Class

### Memory Layout
- **Host only** — no GPU memory, no pinned memory, no CUDA streams
- `ops[MAX_OPS]`: raw hardware addresses from files
- `vals[num_ops]`: values (original indices)
- `compact_keys[num_ops]`: compact addresses (written in Pass 1, read in Pass 2 and 3)
- `result_nops[num_active * num_chunks]`: per-chunk expected entry counts
- `offsets_buf`: packed addr_offsets for all instances
- Output/reference arrays: same structure as GPU version

### Workflow

```
1. generate(block_number)    → Load trace files (same as GPU version)
         ↓
2. cpu_metadata()            → CPU pipeline:
   ├── Pass 1: Compact addresses + coarse histogram (OpenMP parallel)
   ├── Analysis: Region counts, approximate address ranges per instance
   ├── Pass 2: Per-instance fine histograms (OpenMP, coarse-bin quick-reject)
   ├── Boundaries: Binary search on fine prefix sums + addr_offsets
   ├── Pass 3: FML counts per chunk (OpenMP, coarse-bin quick-reject)
   └── Build metas: skip/include thresholds + nops_per_chunk
         ↓
3. cpu_fill()                → Scatter pairs to output arrays (OpenMP, same as GPU)
         ↓
4. verify() [optional]       → Compare against reference stable sort
```

### Key Data Structures

**`InstanceMeta`** (same fields as GPU version):
- `inst_id`: Local instance ID within region
- `type`: Region identifier (ROM/INPUT/RAM)
- `first_addr`, `last_addr`: Raw hardware address range
- `first_addr_chunk`, `first_addr_skip`: Starting chunk and skip count
- `last_addr_chunk`, `last_addr_include`: Ending chunk and include count
- `nops_per_chunk`: Span into result_nops buffer
- `addr_offsets`: Span into offsets_buf

## CPU Fill Logic

Same as GPU version's cpu_fill (parallel via OpenMP):
1. Select output array based on region type
2. Iterate chunks with non-zero `nops_per_chunk`
3. For each entry matching the instance's address range:
   - Apply skip/include logic for first/last address boundaries
   - Write to output position using pre-computed addr_offsets
4. Stop when instance is full

## Performance Comparison (block 24537399, 519M ops)

| Stage | GPU | CPU |
|-------|-----|-----|
| H2D + histogram | 97.8 ms | — (eliminated) |
| Compact + coarse | (included above) | 91 ms |
| Prefix sum | 0.55 ms | — (coarse only, ~0.01 ms) |
| Per-instance histograms | — | 128 ms |
| Boundaries + FML | 2.22 ms | 4 + 112 = 116 ms |
| Build metas | 0.59 ms | 0.4 ms |
| **Metadata total** | **101 ms** | **336 ms** |
| CPU fill | ~1-2 s | ~330 ms |

### CPU Bottleneck

The three data passes dominate (Pass 1: 91 ms, Pass 2: 128 ms, Pass 3: 112 ms), each scanning all 519M compact_keys (2.1 GB). These are memory-bandwidth bound at ~50 GB/s DDR5. The per-entry work (coarse-bin check, instance range comparison) overlaps with memory latency but cannot be fully hidden.

## Usage

```bash
./pair_sort_real_cpu <block_number> [-v]
```

- `block_number`: Directory under `data/` containing trace files
- `-v`: Enable verification against reference stable sort
