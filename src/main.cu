#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>
#include <cub/device/device_scan.cuh>

// ---------- Constants ----------
constexpr uint32_t N_OPS = 1u << 29;
constexpr uint32_t N_ADDR = 1u << 27;
constexpr uint32_t INSTANCE_SIZE = 1u << 22;
constexpr uint32_t NUM_INSTANCES = N_OPS / INSTANCE_SIZE;
constexpr uint32_t NUM_ACTIVE = NUM_INSTANCES / 16;  // how many instances to actually fill
constexpr uint32_t CHUNK_SIZE = 1u << 18;
constexpr uint32_t NUM_CHUNKS = N_OPS / CHUNK_SIZE;
constexpr int N_STREAMS = 4;

// ---------- GPU kernels ----------

__global__ void histogram_kernel(const uint32_t* ops, uint32_t* hist, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        atomicAdd(&hist[ops[i]], 1);
}

// first address and last address of each active instance, based on prefix sums
__global__ void instance_boundaries_kernel(
    const uint32_t* prefix,
    const uint32_t* d_active_ids,
    uint32_t* d_active_first,
    uint32_t* d_active_last,
    uint32_t* d_offset_starts,
    uint32_t num_active)
{
    uint32_t ai = threadIdx.x;
    if (ai >= num_active) return;

    uint32_t inst = d_active_ids[ai];
    uint32_t inst_start = (inst > 0) ? (inst * INSTANCE_SIZE - 1) : 0;
    uint32_t inst_end = (inst < NUM_INSTANCES - 1) ? (inst + 1) * INSTANCE_SIZE : N_OPS;

    // Find largest address <= inst_start
    uint32_t lo = 0, hi = N_ADDR;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo + 1) / 2;
        if (prefix[mid] <= inst_start) lo = mid;
        else hi = mid - 1;
    }
    d_active_first[ai] = lo;

    // Find largest address <= inst_end - 1
    lo = d_active_first[ai]; hi = N_ADDR;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo + 1) / 2;
        if (prefix[mid] < inst_end) lo = mid;
        else hi = mid - 1;
    }
    d_active_last[ai] = lo;

    // Compute offset_starts 
    __syncthreads();
    if (ai == 0) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < num_active; i++) {
            d_offset_starts[i] = offset;
            offset += d_active_last[i] - d_active_first[i] + 1;
        }
    }
}

// Counts first/middle/last entries for each active instance within each chunk
__global__ void chunk_fml_count_kernel(
    const uint32_t* ops,
    const uint32_t* d_active_first,
    const uint32_t* d_active_last,
    uint32_t* d_fml,  // [num_active][NUM_CHUNKS][3] = first/middle/last (instance-major)
    uint32_t num_active,
    uint32_t n)
{
    __shared__ uint32_t s_first[NUM_ACTIVE];
    __shared__ uint32_t s_last[NUM_ACTIVE];
    if (threadIdx.x < num_active) {
        s_first[threadIdx.x] = d_active_first[threadIdx.x];
        s_last[threadIdx.x] = d_active_last[threadIdx.x];
    }
    __syncthreads();

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    uint32_t lane = threadIdx.x & 31;

    for (; i < n; i += stride) {
        uint32_t addr = ops[i];
        uint32_t chunk_id = i / CHUNK_SIZE;

        // Check if entire warp is in same chunk
        uint32_t warp_base_i = i - lane;
        bool same_chunk = (warp_base_i / CHUNK_SIZE) == ((warp_base_i + 31) / CHUNK_SIZE);

        for (uint32_t ai = 0; ai < num_active; ai++) {
            uint32_t fa = s_first[ai];
            uint32_t la = s_last[ai];

            // Early exit: if ALL threads in warp have addr < fa
            if (__all_sync(0xFFFFFFFF, addr < fa)) break;

            bool in_range = (addr >= fa && addr <= la);

            // Skip if no thread in warp matches this instance
            if (!__any_sync(0xFFFFFFFF, in_range)) continue;

            // Categorize: 0=first, 1=middle, 2=last, 3=none
            uint32_t cat = 3;
            if (in_range) {
                if (addr == fa) cat = 0;
                else if (addr == la) cat = 2;
                else cat = 1;
            }

            if (same_chunk) {
                // Warp reduction
                for (uint32_t c = 0; c < 3; c++) {
                    unsigned mask = __ballot_sync(0xFFFFFFFF, cat == c);
                    if (mask && lane == 0) {
                        atomicAdd(&d_fml[(ai * NUM_CHUNKS + chunk_id) * 3 + c], __popc(mask));
                    }
                }
            } else if (in_range) {
                // Should never be used as chunk size is a large power of two
                atomicAdd(&d_fml[(ai * NUM_CHUNKS + chunk_id) * 3 + cat], 1);
            }
        }
    }
}

// Build metas on GPU: one block per active instance, 256 threads per block
__global__ void build_metas_kernel(
    const uint32_t* d_fml,           // [NUM_ACTIVE][NUM_CHUNKS][3], instance-major
    const uint32_t* d_prefix,        // global prefix sums [0..N_ADDR]
    const uint32_t* d_active_ids,    
    const uint32_t* d_active_first,  
    const uint32_t* d_active_last,   
    uint32_t* d_result_nops,         // output: [NUM_ACTIVE][NUM_CHUNKS]
    uint32_t* d_meta_scalars,        // output: [NUM_ACTIVE][4]
    uint32_t num_active)
{
    const uint32_t ai = blockIdx.x;
    if (ai >= num_active) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;

    // Shared memory for block communication
    __shared__ uint32_t s_total_compacted;
    __shared__ uint32_t s_first_addr_total_skip;
    __shared__ uint32_t s_last_addr_total_include;
    __shared__ bool s_single_addr;
    __shared__ uint32_t s_fa_chunk, s_fa_skip, s_la_chunk, s_la_include;
    __shared__ uint32_t s_scan[256];

    const uint32_t* fml_base = d_fml + (size_t)ai * NUM_CHUNKS * 3;
    uint32_t* scratch = d_result_nops + (size_t)ai * NUM_CHUNKS;

    // ---- Phase 1: find non-empty chunks
    // Each thread handles a contiguous range of chunks
    uint32_t chunks_per_thread = (NUM_CHUNKS + nthreads - 1) / nthreads;
    uint32_t c_start = tid * chunks_per_thread;
    uint32_t c_end = min(c_start + chunks_per_thread, (uint32_t)NUM_CHUNKS);

    // Pass 1: count non-empty chunks per thread
    uint32_t my_count = 0;
    for (uint32_t c = c_start; c < c_end; c++) {
        uint32_t idx = c * 3;
        if (fml_base[idx] + fml_base[idx + 1] + fml_base[idx + 2] > 0)
            my_count++;
    }

    // Block-level exclusive prefix sum of my_count
    s_scan[tid] = my_count;
    __syncthreads();
    if (tid == 0) {
        uint32_t total = 0;
        for (uint32_t i = 0; i < nthreads; i++) {
            uint32_t val = s_scan[i];
            s_scan[i] = total;
            total += val;
        }
        s_total_compacted = total;
    }
    __syncthreads();

    uint32_t my_offset = s_scan[tid];

    // Pass 2: write compacted chunk IDs
    uint32_t write_pos = my_offset;
    for (uint32_t c = c_start; c < c_end; c++) {
        uint32_t idx = c * 3;
        if (fml_base[idx] + fml_base[idx + 1] + fml_base[idx + 2] > 0)
            scratch[write_pos++] = c;
    }
    __syncthreads();

    uint32_t nc = s_total_compacted;

    // ---- Phase 2: s_first_addr_total_skip and s_last_addr_total_include
    if (tid == 0) {
        uint32_t inst = d_active_ids[ai];
        uint32_t fa = d_active_first[ai];
        uint32_t la = d_active_last[ai];
        bool single_addr = (fa == la);
        uint32_t n_addrs = la - fa + 1;

        s_single_addr = single_addr;

        uint32_t halo_base = (inst == 0) ? 0 : inst * INSTANCE_SIZE - 1;
        s_first_addr_total_skip = halo_base - d_prefix[fa];

        if (!single_addr) {
            uint32_t filled_before_last = d_prefix[fa + n_addrs - 1] - inst * INSTANCE_SIZE;
            s_last_addr_total_include = INSTANCE_SIZE - filled_before_last;
        } else {
            s_last_addr_total_include = INSTANCE_SIZE;
            if (inst > 0) s_last_addr_total_include++;  // +1 for halo row
        }
    }
    __syncthreads();

    uint32_t first_addr_total_skip = s_first_addr_total_skip;
    uint32_t last_addr_total_include = s_last_addr_total_include;
    bool single_addr = s_single_addr;

    // ---- Phase 3: Find fa_chunk/fa_skip (prefix-sum search over first-counts)
    uint32_t nc_per_thread = (nc + nthreads - 1) / nthreads;
    uint32_t ci_start = tid * nc_per_thread;
    uint32_t ci_end = min(ci_start + nc_per_thread, nc);

    uint32_t my_count_first = 0;

    for (uint32_t ci = ci_start; ci < ci_end; ci++)
        my_count_first += fml_base[scratch[ci] * 3 + 0];

    s_scan[tid] = my_count_first;
    __syncthreads();

    // Thread 0 does the prefix-sum search across thread totals
    if (tid == 0) {
        uint32_t cum = 0;
        for (uint32_t t = 0; t < nthreads; t++) {
            if (cum + s_scan[t] > first_addr_total_skip) {
                uint32_t t_start = t * nc_per_thread;
                uint32_t t_end = min(t_start + nc_per_thread, nc);
                uint32_t local_cum = cum;
                for (uint32_t ci = t_start; ci < t_end; ci++) {
                    uint32_t cf = fml_base[scratch[ci] * 3 + 0];
                    if (local_cum + cf > first_addr_total_skip) {
                        s_fa_chunk = scratch[ci];
                        s_fa_skip = first_addr_total_skip - local_cum;
                        break;
                    }
                    local_cum += cf;
                }
                break;
            }
            cum += s_scan[t];
        }
    }
    __syncthreads();

    // ---- Phase 4: Find la_chunk/la_include (parallel, same pattern as Phase 3)
    uint32_t la_cat = single_addr ? 0 : 2;
    uint32_t la_threshold = single_addr
        ? (first_addr_total_skip + last_addr_total_include)
        : last_addr_total_include;

    uint32_t my_cum_last = 0;
    for (uint32_t ci = ci_start; ci < ci_end; ci++)
        my_cum_last += fml_base[scratch[ci] * 3 + la_cat];

    s_scan[tid] = my_cum_last;
    __syncthreads();

    if (tid == 0) {
        uint32_t cum = 0;
        for (uint32_t t = 0; t < nthreads; t++) {
            if (cum + s_scan[t] >= la_threshold) {
                uint32_t t_start = t * nc_per_thread;
                uint32_t t_end = min(t_start + nc_per_thread, nc);
                uint32_t local_cum = cum;
                for (uint32_t ci = t_start; ci < t_end; ci++) {
                    uint32_t cv = fml_base[scratch[ci] * 3 + la_cat];
                    if (local_cum + cv >= la_threshold) {
                        s_la_chunk = scratch[ci];
                        s_la_include = la_threshold - local_cum;
                        break;
                    }
                    local_cum += cv;
                }
                break;
            }
            cum += s_scan[t];
        }
    }
    __syncthreads();

    uint32_t fa_chunk = s_fa_chunk;
    uint32_t fa_skip = s_fa_skip;
    uint32_t la_chunk = s_la_chunk;
    uint32_t la_include = s_la_include;

    // ---- Phase 5: Chunk elimination (sparse write) 
    // re-zero d_result_nops used previously as scratch
    uint32_t *results_nops = d_result_nops + (size_t)ai * NUM_CHUNKS;
    for (uint32_t c = tid; c < NUM_CHUNKS; c += nthreads)
        results_nops[c] = 0;
    __syncthreads();

    for (uint32_t c = c_start; c < c_end; c++) {
        uint32_t idx = c * 3;
        uint32_t cf = fml_base[idx + 0];
        uint32_t cm = fml_base[idx + 1];
        uint32_t cl = fml_base[idx + 2];
        if (cf + cm + cl == 0) continue;

        bool needed = (cm > 0);
        if (cf > 0) {
            if (single_addr) {
                if (c >= fa_chunk && c <= la_chunk) needed = true;
            } else {
                if (c >= fa_chunk) needed = true;
            }
        }
        if (cl > 0 && c <= la_chunk)
            needed = true;
        if (needed)
            results_nops[c] = cf + cm + cl;
    }

    // ---- Phase 6: Write scalar results
    if (tid == 0) {
        uint32_t* out = d_meta_scalars + ai * 4;
        out[0] = fa_chunk;
        out[1] = fa_skip;
        out[2] = la_chunk;
        out[3] = la_include;
    }
}

// Compute addr_offsets on GPU: addr_offsets[j] = prefix[fa+j] - inst*INSTANCE_SIZE + 1
// One block per instance, threads stride over addresses
__global__ void compute_addr_offsets_kernel(
    const uint32_t* d_prefix,
    const uint32_t* d_active_ids,
    const uint32_t* d_active_first,
    const uint32_t* d_active_last,
    uint32_t* d_addr_offsets,        // output: packed per-instance
    const uint32_t* d_offset_starts, // start position in d_addr_offsets for each instance
    uint32_t num_active)
{
    uint32_t ai = blockIdx.x;
    if (ai >= num_active) return;

    uint32_t inst = d_active_ids[ai];
    uint32_t fa = d_active_first[ai];
    uint32_t la = d_active_last[ai];
    uint32_t n_addrs = la - fa + 1;
    uint32_t base_val = inst * INSTANCE_SIZE - 1;
    uint32_t* out = d_addr_offsets + d_offset_starts[ai];

    uint32_t tid = threadIdx.x;
    uint32_t stride = blockDim.x;

    if (tid == 0)
        out[0] = (inst == 0) ? 1 : 0;

    for (uint32_t j = tid + 1; j < n_addrs; j += stride)
        out[j] = d_prefix[fa + j] - base_val;
}

struct InstanceMeta {
    uint32_t inst_id;
    uint32_t first_addr, last_addr;
    const uint32_t* nops_per_chunk;       // points into pinned h_result_nops (not owned), NUM_CHUNKS entries
    uint32_t* addr_offsets;               // points into pinned buffer (not owned)

    uint32_t first_addr_chunk;
    uint32_t first_addr_skip;

    uint32_t last_addr_chunk;
    uint32_t last_addr_include;
};

// ---------- PairSortGPU ----------

class PairSortGPU {
    // GPU memory
    uint32_t *d_ops, *d_hist, *d_prefix;
    void *d_temp;
    size_t d_temp_bytes;
    uint32_t *d_active_ids, *d_active_first, *d_active_last;
    uint32_t *d_fml;
    uint32_t *d_result_nops;     // [NUM_ACTIVE][NUM_CHUNKS]
    uint32_t *d_meta_scalars;    // [NUM_ACTIVE][4]: fa_chunk, fa_skip, la_chunk, la_include
    uint32_t *d_addr_offsets, *d_offset_starts; 

    cudaStream_t streams[N_STREAMS];
    cudaStream_t d2h_stream;
    cudaStream_t meta_stream;  
    
    // Host pinned
    uint32_t *h_ops;
    uint32_t *h_offsets_buf;
    size_t h_offsets_buf_size;  // in bytes
    uint32_t *h_result_nops;
    uint32_t *h_meta_scalars;

    // Host
    uint32_t *h_vals;
    uint32_t *out_ops, *out_vals;
    uint32_t *ref_ops, *ref_vals;
    std::vector<uint32_t> h_active_first, h_active_last;
    std::vector<uint32_t> active;
    std::vector<InstanceMeta> metas;

    size_t fml_size;

public:
    PairSortGPU();
    ~PairSortGPU();

    void pick_active_instances();
    void generate();
    void gpu_metadata();
    void cpu_fill();
    void reference_sort();
    void verify();
};

PairSortGPU::PairSortGPU()
    : h_active_first(NUM_ACTIVE), h_active_last(NUM_ACTIVE), metas(NUM_ACTIVE)
{
    fml_size = (size_t)NUM_CHUNKS * NUM_ACTIVE * 3;

    // GPU allocations
    cudaMalloc(&d_ops, (size_t)N_OPS * sizeof(uint32_t));
    cudaMalloc(&d_hist, (size_t)N_ADDR * sizeof(uint32_t));
    cudaMalloc(&d_prefix, (size_t)(N_ADDR + 1) * sizeof(uint32_t));

    d_temp = nullptr;
    d_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, d_temp_bytes, d_hist, d_prefix, N_ADDR);
    cudaMalloc(&d_temp, d_temp_bytes);

    cudaMalloc(&d_active_ids, NUM_ACTIVE * sizeof(uint32_t));
    cudaMalloc(&d_active_first, NUM_ACTIVE * sizeof(uint32_t));
    cudaMalloc(&d_active_last, NUM_ACTIVE * sizeof(uint32_t));
    cudaMalloc(&d_fml, fml_size * sizeof(uint32_t));
    cudaMalloc(&d_result_nops, (size_t)NUM_ACTIVE * NUM_CHUNKS * sizeof(uint32_t));
    cudaMalloc(&d_meta_scalars, (size_t)NUM_ACTIVE * 4 * sizeof(uint32_t));
    cudaMalloc(&d_addr_offsets, (size_t)NUM_ACTIVE * (INSTANCE_SIZE + 1) * sizeof(uint32_t));
    cudaMalloc(&d_offset_starts, (size_t)NUM_ACTIVE * sizeof(uint32_t));

    // Streams
    for (int s = 0; s < N_STREAMS; s++)
        cudaStreamCreate(&streams[s]);
    cudaStreamCreate(&d2h_stream);
    cudaStreamCreate(&meta_stream);

    // Host pinned
    cudaMallocHost(&h_ops, (size_t)N_OPS * sizeof(uint32_t));
    h_offsets_buf_size = 1ull << 28;  // 256 MB
    cudaMallocHost(&h_offsets_buf, h_offsets_buf_size);
    cudaMallocHost(&h_result_nops, (size_t)NUM_ACTIVE * NUM_CHUNKS * sizeof(uint32_t));
    cudaMallocHost(&h_meta_scalars, (size_t)NUM_ACTIVE * 4 * sizeof(uint32_t));

    // Host
    h_vals = new uint32_t[N_OPS];
    out_ops = new uint32_t[N_OPS]();
    out_vals = new uint32_t[N_OPS]();
    ref_ops = new uint32_t[N_OPS];
    ref_vals = new uint32_t[N_OPS];

    size_t gpu_bytes = ((size_t)N_OPS + N_ADDR + N_ADDR + 1) * sizeof(uint32_t)  // d_ops, d_hist, d_prefix
                     + d_temp_bytes                                              // d_temp (CUB)
                     + (size_t)NUM_ACTIVE * 3 * sizeof(uint32_t)                 // d_active_{ids,first,last}
                     + fml_size * sizeof(uint32_t)                               // d_fml
                     + (size_t)NUM_ACTIVE * NUM_CHUNKS * sizeof(uint32_t)         // d_result_nops
                     + (size_t)NUM_ACTIVE * 4 * sizeof(uint32_t)                 // d_meta_scalars
                     + (size_t)NUM_ACTIVE * (INSTANCE_SIZE + 1) * sizeof(uint32_t) // d_addr_offsets
                     + (size_t)NUM_ACTIVE * sizeof(uint32_t);                    // d_offset_starts
    size_t pinned_bytes = (size_t)N_OPS * sizeof(uint32_t)                       // h_ops
                        + h_offsets_buf_size                                       // h_offsets_buf
                        + (size_t)NUM_ACTIVE * NUM_CHUNKS * sizeof(uint32_t)      // h_result_nops
                        + (size_t)NUM_ACTIVE * 4 * sizeof(uint32_t);             // h_meta_scalars
    std::cout << "=== Setup ===" << std::endl
              << std::fixed << std::setprecision(1)
              << "  GPU:         " << gpu_bytes / (double)(1<<30) << " GB" << std::endl
              << "  Host pinned: " << pinned_bytes / (double)(1<<30) << " GB" << std::endl;
}

void PairSortGPU::pick_active_instances() {
    std::mt19937 rng(123);
    active.clear();
    while (active.size() < NUM_ACTIVE) {
        uint32_t r = rng() % NUM_INSTANCES;
        bool found = false;
        for (uint32_t v : active) if (v == r) { found = true; break; }
        if (!found) active.push_back(r);
    }
    std::sort(active.begin(), active.end()); //Important active are sorted
    std::cout << "  Active instances (" << NUM_ACTIVE << " of " << NUM_INSTANCES << "):";
    for (uint32_t id : active) std::cout << " " << id;
    std::cout << std::endl;

    cudaMemcpy(d_active_ids, active.data(), NUM_ACTIVE * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

PairSortGPU::~PairSortGPU() {
    // GPU
    cudaFree(d_ops);
    cudaFree(d_hist);
    cudaFree(d_prefix);
    cudaFree(d_temp);
    cudaFree(d_active_ids);
    cudaFree(d_active_first);
    cudaFree(d_active_last);
    cudaFree(d_fml);
    cudaFree(d_result_nops);
    cudaFree(d_meta_scalars);
    cudaFree(d_addr_offsets);
    cudaFree(d_offset_starts);

    // Streams
    for (int s = 0; s < N_STREAMS; s++)
        cudaStreamDestroy(streams[s]);
    cudaStreamDestroy(d2h_stream);
    cudaStreamDestroy(meta_stream);

    // Host
    cudaFreeHost(h_ops);
    cudaFreeHost(h_offsets_buf);
    cudaFreeHost(h_result_nops);
    cudaFreeHost(h_meta_scalars);
    delete[] h_vals;
    delete[] out_ops;
    delete[] out_vals;
    delete[] ref_ops;
    delete[] ref_vals;
}

void PairSortGPU::generate() {
    std::cout << std::endl << "=== Generate ===" << std::endl;
    double t = omp_get_wtime();
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> addr_dist(0, N_ADDR - 1);
    std::uniform_int_distribution<uint32_t> val_dist(0, UINT32_MAX);
    for (uint32_t i = 0; i < N_OPS; i++) {
        h_ops[i] = addr_dist(rng);
        h_vals[i] = val_dist(rng);
    }
    std::cout << std::fixed << std::setprecision(2)
              << "  " << N_OPS << " pairs in " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;
}

void PairSortGPU::gpu_metadata() {
    std::cout << std::endl << "=== GPU Metadata ===" << std::endl;
    double t_total = omp_get_wtime(), t;

    // Clear histogram
    cudaMemset(d_hist, 0, (size_t)N_ADDR * sizeof(uint32_t));

    // H2D + histogram pipelined by chunks
    t = omp_get_wtime();
    const size_t chunk_bytes = CHUNK_SIZE * sizeof(uint32_t);
    int hist_block, hist_grid;
    cudaOccupancyMaxPotentialBlockSize(&hist_grid, &hist_block, histogram_kernel, 0, 0);
    for (uint32_t c = 0; c < NUM_CHUNKS - 1; c++) {
        int s = c % N_STREAMS;
        size_t offset = (size_t)c * CHUNK_SIZE;
        cudaMemcpyAsync(d_ops + offset, h_ops + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[s]);
        histogram_kernel<<<hist_grid, hist_block, 0, streams[s]>>>(d_ops + offset, d_hist, CHUNK_SIZE);
    }
    cudaDeviceSynchronize();  // all prior chunks done (simulates real-time generation)

    double t_last_chunk = omp_get_wtime();
    {
        uint32_t c = NUM_CHUNKS - 1;
        int s = c % N_STREAMS;
        size_t offset = (size_t)c * CHUNK_SIZE;
        cudaMemcpyAsync(d_ops + offset, h_ops + offset, chunk_bytes,
                        cudaMemcpyHostToDevice, streams[s]);
        histogram_kernel<<<hist_grid, hist_block, 0, streams[s]>>>(d_ops + offset, d_hist, CHUNK_SIZE);
    }
    cudaDeviceSynchronize();
    std::cout << std::fixed << std::setprecision(2)
              << "  H2D + histogram:  " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // Prefix sum
    t = omp_get_wtime();
    cub::DeviceScan::ExclusiveSum(d_temp, d_temp_bytes, d_hist, d_prefix, N_ADDR);
    uint32_t n_ops_val = N_OPS;
    cudaMemcpy(d_prefix + N_ADDR, &n_ops_val, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    std::cout << std::fixed << std::setprecision(2)
              << "  Prefix sum:       " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // GPU: active instance boundaries + chunk FML counts
    t = omp_get_wtime();
    uint32_t num_active = NUM_ACTIVE;
    const size_t active_bytes = num_active * sizeof(uint32_t);

    instance_boundaries_kernel<<<1, num_active>>>(
        d_prefix, d_active_ids, d_active_first, d_active_last, d_offset_starts, num_active);

    // FML kernel
    cudaMemset(d_fml, 0, fml_size * sizeof(uint32_t));
    int fml_block, fml_grid;
    cudaOccupancyMaxPotentialBlockSize(&fml_grid, &fml_block, chunk_fml_count_kernel, 0, 0);
    chunk_fml_count_kernel<<<fml_grid, fml_block>>>(
        d_ops, d_active_first, d_active_last, d_fml, num_active, N_OPS);
    cudaDeviceSynchronize();

    // D2H: active boundaries (needed for prefix D2H offsets)
    cudaMemcpy(h_active_first.data(), d_active_first, active_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_active_last.data(), d_active_last, active_bytes, cudaMemcpyDeviceToHost);
    std::cout << std::fixed << std::setprecision(2)
              << "  Boundaries + FML: " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // --- Build metas + addr_offsets: all on GPU, overlapped ---
    t = omp_get_wtime();
    const size_t n_inst = active.size();

    // Compute offset_starts + total_addrs from already-D2H'd boundaries
    std::vector<uint32_t> h_offset_starts(n_inst);
    uint32_t total_addrs = 0;
    for (size_t i = 0; i < n_inst; i++) {
        h_offset_starts[i] = total_addrs;
        total_addrs += h_active_last[i] - h_active_first[i] + 1;
    }
    size_t total_offsets_bytes = (size_t)total_addrs * sizeof(uint32_t);

    // Launch build_metas_kernel on meta_stream
    build_metas_kernel<<<num_active, 256, 0, meta_stream>>>(
        d_fml, d_prefix, d_active_ids, d_active_first, d_active_last,
        d_result_nops, d_meta_scalars, num_active);

    // Launch addr_offsets kernel on d2h_stream (overlaps with build_metas)
    compute_addr_offsets_kernel<<<num_active, 1024, 0, d2h_stream>>>(
        d_prefix, d_active_ids, d_active_first, d_active_last,
        d_addr_offsets, d_offset_starts, num_active);

    // Queue D2H for meta results on meta_stream (after build_metas_kernel completes)
    cudaMemcpyAsync(h_meta_scalars, d_meta_scalars, num_active * 4 * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, meta_stream);
    cudaMemcpyAsync(h_result_nops, d_result_nops, (size_t)NUM_ACTIVE * NUM_CHUNKS * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, meta_stream);

    // Queue addr_offsets D2H after kernel completes (same stream)
    cudaMemcpyAsync(h_offsets_buf, d_addr_offsets, total_offsets_bytes,
                    cudaMemcpyDeviceToHost, d2h_stream);

    // Wait for both streams
    cudaStreamSynchronize(meta_stream);
    cudaStreamSynchronize(d2h_stream);

    // Populate metas from GPU results
    for (size_t i = 0; i < n_inst; i++) {
        uint32_t* sc = h_meta_scalars + i * 4;
        metas[i].inst_id = active[i];
        metas[i].first_addr = h_active_first[i];
        metas[i].last_addr = h_active_last[i];
        metas[i].first_addr_chunk = sc[0];
        metas[i].first_addr_skip = sc[1];
        metas[i].last_addr_chunk = sc[2];
        metas[i].last_addr_include = sc[3];
        metas[i].addr_offsets = h_offsets_buf + h_offset_starts[i];
        metas[i].nops_per_chunk = h_result_nops + i * NUM_CHUNKS;
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  Build metas:      " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl
              << "  TOTAL:            " << (omp_get_wtime() - t_total) * 1e3 << " ms" << std::endl
              << "  Last chunk to ready: " << (omp_get_wtime() - t_last_chunk) * 1e3 << " ms" << std::endl;
}

void PairSortGPU::cpu_fill() {
    std::cout << std::endl << "=== CPU Fill ===" << std::endl;
    double t_total = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t idx = 0; idx < metas.size(); idx++) {
        auto& m = metas[idx];
        bool single_addr = (m.first_addr == m.last_addr);

        uint32_t out_base = m.inst_id * INSTANCE_SIZE;
        uint32_t total_written = 0;
        for (uint32_t chunk_gid = 0; chunk_gid < NUM_CHUNKS; chunk_gid++) {
            uint32_t expected = m.nops_per_chunk[chunk_gid];
            if (expected == 0) continue;
            size_t base = (size_t)chunk_gid * CHUNK_SIZE;
            uint32_t found = 0;
            uint32_t first_found = 0;
            uint32_t last_found = 0;

            for (uint32_t j = 0; j < CHUNK_SIZE && found < expected; j++) {
                uint32_t addr = h_ops[base + j];
                if (addr < m.first_addr || addr > m.last_addr) continue;
                uint32_t ind = addr - m.first_addr;
                found++;

                // Skip logic for first/last address boundaries
                bool skip = false;
                if (addr == m.first_addr) {
                    first_found++;
                    if (chunk_gid < m.first_addr_chunk) skip = true;
                    else if (chunk_gid == m.first_addr_chunk && first_found <= m.first_addr_skip) skip = true;
                    else if (single_addr) {
                        if (chunk_gid > m.last_addr_chunk) skip = true;
                        else if (chunk_gid == m.last_addr_chunk && first_found > m.last_addr_include) skip = true;
                    }
                } else if (addr == m.last_addr) {
                    last_found++;
                    if (chunk_gid > m.last_addr_chunk) skip = true;
                    else if (chunk_gid == m.last_addr_chunk && last_found > m.last_addr_include) skip = true;
                }
                if (skip) continue;

                uint32_t pos = m.addr_offsets[ind]++;
                if (pos == 0) continue;  // halo entry (inst > 0); inst==0 starts at 1
                uint32_t out_pos = out_base + pos - 1;
                out_ops[out_pos] = addr;
                out_vals[out_pos] = h_vals[base + j];
                total_written++;
                if (total_written >= INSTANCE_SIZE) break;
            }
            if (total_written >= INSTANCE_SIZE) break;
        }
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  " << NUM_ACTIVE << " instances in " << (omp_get_wtime() - t_total) * 1e3 << " ms" << std::endl;
}

void PairSortGPU::reference_sort() {
    std::cout << std::endl << "=== Verify ===" << std::endl;
    double t = omp_get_wtime();
    struct Triple { uint32_t key, pos, val; };
    std::vector<Triple> triples(N_OPS);
    for (uint32_t i = 0; i < N_OPS; i++)
        triples[i] = {h_ops[i], i, h_vals[i]};
    std::sort(triples.begin(), triples.end(), [](const Triple& a, const Triple& b) {
        return a.key < b.key || (a.key == b.key && a.pos < b.pos);
    });
    for (uint32_t i = 0; i < N_OPS; i++) {
        ref_ops[i] = triples[i].key;
        ref_vals[i] = triples[i].val;
    }
    std::cout << std::fixed << std::setprecision(2)
              << "  Reference sort:   " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;
}

void PairSortGPU::verify() {
    double t = omp_get_wtime();
    uint32_t total_verified = 0;
    for (const auto& m : metas) {
        uint32_t inst = m.inst_id;
        uint32_t start = inst * INSTANCE_SIZE;

        for (uint32_t j = 0; j < INSTANCE_SIZE; j++) {
            uint32_t ind = start + j;
            if (out_ops[ind] != ref_ops[ind] || out_vals[ind] != ref_vals[ind]) {
                std::cout << "MISMATCH at ref position " << ind << " (instance " << inst
                          << ", local " << j << "): got (" << out_ops[ind] << "," << out_vals[ind]
                          << ") expected (" << ref_ops[ind] << "," << ref_vals[ind] << ")" << std::endl;
                return;
            }
        }
        total_verified += INSTANCE_SIZE;
    }
    std::cout << std::fixed << std::setprecision(2)
              << "  Verify:           " << (omp_get_wtime() - t) * 1e3
              << " ms -- " << total_verified << " entries, " << metas.size() << " instances OK" << std::endl;
}

// =====================================================================

int main(int argc, char** argv) {
    bool do_verify = false;
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "-v") do_verify = true;

    PairSortGPU app;
    app.pick_active_instances();
    app.generate();
    app.gpu_metadata();
    app.cpu_fill();

    if (do_verify) {
        app.reference_sort();
        app.verify();
    }
}
