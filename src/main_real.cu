#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>
#include <cub/device/device_scan.cuh>

// ---------- Constants ----------
constexpr uint32_t N_ADDR = 1u << 26;
constexpr uint32_t RAM_BASE = 0xA0000000u;
constexpr uint32_t RAM_END  = 0xBFFFFFFFu;
constexpr uint32_t ADDR_SHIFT = 3;
constexpr uint32_t INSTANCE_SIZE = 1u << 22;
constexpr uint32_t MAX_INSTANCES = 256;
constexpr uint32_t MASK_INSTANCES_WORDS = MAX_INSTANCES / 32;
constexpr uint32_t MAX_OPS = (uint32_t)MAX_INSTANCES * INSTANCE_SIZE;  // 2^30
constexpr uint32_t N_WORKERS = 16;
constexpr uint32_t MAX_ACTIVE = (MAX_INSTANCES + N_WORKERS - 1) / N_WORKERS;  // 16
constexpr uint32_t MAX_CHUNKS = 4096;
constexpr int N_STREAMS = 4;

// ---------- GPU kernels ----------

__global__ void extract_active_ids_kernel(
    const uint32_t* d_mask, uint32_t num_instances,
    uint32_t* d_active_ids, uint32_t* d_num_active)
{
    uint32_t count = 0;
    for (uint32_t i = 0; i < num_instances; i++) {
        if (d_mask[i / 32] & (1u << (i % 32)))
            d_active_ids[count++] = i;
    }
    *d_num_active = count;
}

__global__ void shift_addresses_kernel(uint32_t* ops, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        ops[i] = (ops[i] - RAM_BASE) >> ADDR_SHIFT;
}

__global__ void histogram_kernel(const uint32_t* ops, uint32_t* hist, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        atomicAdd(&hist[ops[i]], 1);
}

__global__ void instance_boundaries_kernel(
    const uint32_t* prefix,
    const uint32_t* d_active_ids,
    uint32_t* d_active_first,
    uint32_t* d_active_last,
    uint32_t* d_offset_starts,
    uint32_t num_active,
    uint32_t num_instances,
    uint32_t n_ops)
{
    uint32_t ai = threadIdx.x;
    if (ai >= num_active) return;

    uint32_t inst = d_active_ids[ai];
    uint32_t inst_start = (inst > 0) ? (inst * INSTANCE_SIZE - 1) : 0;
    uint32_t inst_end = (inst < num_instances - 1) ? (inst + 1) * INSTANCE_SIZE : n_ops;

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
    const uint32_t* d_chunk_offsets,
    uint32_t* d_fml,  // [num_active][num_chunks][3] = first/middle/last (instance-major)
    uint32_t num_active,
    uint32_t num_chunks,
    uint32_t n_ops)
{
    __shared__ uint32_t s_first[MAX_ACTIVE];
    __shared__ uint32_t s_last[MAX_ACTIVE];
    if (threadIdx.x < num_active) {
        s_first[threadIdx.x] = d_active_first[threadIdx.x];
        s_last[threadIdx.x] = d_active_last[threadIdx.x];
    }
    __syncthreads();

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    uint32_t lane = threadIdx.x & 31;

    uint32_t chunk_id = 0;

    for (; i < n_ops; i += stride) {
        while (chunk_id + 1 < num_chunks && i >= d_chunk_offsets[chunk_id + 1])
            chunk_id++;

        uint32_t addr = ops[i];

        // Check if entire warp is in same chunk
        uint32_t warp_chunk_first = __shfl_sync(0xFFFFFFFF, chunk_id, 0);
        uint32_t warp_chunk_last = __shfl_sync(0xFFFFFFFF, chunk_id, 31);
        bool same_chunk = (warp_chunk_first == warp_chunk_last);

        for (uint32_t ai = 0; ai < num_active; ai++) {
            uint32_t fa = s_first[ai];
            uint32_t la = s_last[ai];

            if (__all_sync(0xFFFFFFFF, addr < fa)) break;

            bool in_range = (addr >= fa && addr <= la);

            if (!__any_sync(0xFFFFFFFF, in_range)) continue;

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
                        atomicAdd(&d_fml[(ai * num_chunks + chunk_id) * 3 + c], __popc(mask));
                    }
                }
            } else if (in_range) {
                atomicAdd(&d_fml[(ai * num_chunks + chunk_id) * 3 + cat], 1);
            }
        }
    }
}

// Build metas on GPU: one block per active instance, 256 threads per block
__global__ void build_metas_kernel(
    const uint32_t* d_fml,           // [num_active][num_chunks][3], instance-major
    const uint32_t* d_prefix,        // global prefix sums [0..N_ADDR]
    const uint32_t* d_active_ids,
    const uint32_t* d_active_first,
    const uint32_t* d_active_last,
    uint32_t* d_result_nops,         // output + scratch: [num_active][num_chunks]
    uint32_t* d_meta_scalars,        // output: [num_active][4]
    uint32_t num_active,
    uint32_t num_chunks,
    uint32_t n_ops)
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

    const uint32_t* fml_base = d_fml + (size_t)ai * num_chunks * 3;
    uint32_t* scratch = d_result_nops + (size_t)ai * num_chunks;

    // ---- Phase 1: find non-empty chunks
    uint32_t chunks_per_thread = (num_chunks + nthreads - 1) / nthreads;
    uint32_t c_start = tid * chunks_per_thread;
    uint32_t c_end = min(c_start + chunks_per_thread, num_chunks);

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

        uint32_t inst_size = min(INSTANCE_SIZE, n_ops - inst * INSTANCE_SIZE);
        uint32_t halo_base = (inst == 0) ? 0 : inst * INSTANCE_SIZE - 1;
        s_first_addr_total_skip = halo_base - d_prefix[fa];

        if (!single_addr) {
            uint32_t filled_before_last = d_prefix[fa + n_addrs - 1] - inst * INSTANCE_SIZE;
            s_last_addr_total_include = inst_size - filled_before_last;
        } else {
            s_last_addr_total_include = inst_size;
            if (inst > 0) s_last_addr_total_include++;  // +1 for halo row
        }
    }
    __syncthreads();

    uint32_t first_addr_total_skip = s_first_addr_total_skip;
    uint32_t last_addr_total_include = s_last_addr_total_include;
    bool single_addr = s_single_addr;

    // ---- Phase 3: Find fa_chunk/fa_skip
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

    // ---- Phase 4: Find la_chunk/la_include
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
    uint32_t *results_nops = d_result_nops + (size_t)ai * num_chunks;
    for (uint32_t c = tid; c < num_chunks; c += nthreads)
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

__global__ void compute_addr_offsets_kernel(
    const uint32_t* d_prefix,
    const uint32_t* d_active_ids,
    const uint32_t* d_active_first,
    const uint32_t* d_active_last,
    uint32_t* d_addr_offsets,
    const uint32_t* d_offset_starts,
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
    const uint32_t* nops_per_chunk;       // points into pinned h_result_nops (not owned)
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
    uint32_t *d_active_mask;  // GPU copy of bitmask
    uint32_t *d_fml;
    uint32_t *d_result_nops;
    uint32_t *d_meta_scalars;
    uint32_t *d_addr_offsets, *d_offset_starts;
    uint32_t *d_chunk_offsets;

    cudaStream_t streams[N_STREAMS];
    cudaStream_t d2h_stream;
    cudaStream_t meta_stream;

    // Host pinned
    uint32_t *h_ops;
    uint32_t *h_offsets_buf;
    size_t h_offsets_buf_size;
    uint32_t *h_result_nops;
    uint32_t *h_meta_scalars;

    // Host
    uint32_t *h_vals;
    uint32_t *out_ops, *out_vals;
    uint32_t *ref_ops, *ref_vals;
    std::vector<uint32_t> h_active_first, h_active_last;
    uint32_t active_mask[MASK_INSTANCES_WORDS];
    std::vector<InstanceMeta> metas;

    // Runtime state
    uint32_t n_ops;
    uint32_t num_chunks;
    uint32_t num_instances;
    uint32_t num_active;
    std::vector<uint32_t> chunk_offsets;

    size_t fml_size;

public:
    PairSortGPU();
    ~PairSortGPU();

    void pick_active_instances();
    void generate(uint32_t block_number);
    void gpu_metadata();
    void cpu_fill();
    void reference_sort();
    void verify();
};

PairSortGPU::PairSortGPU()
    : h_active_first(MAX_ACTIVE), h_active_last(MAX_ACTIVE), metas(MAX_ACTIVE),
      n_ops(0), num_chunks(0), num_instances(0), num_active(0)
{
    fml_size = (size_t)MAX_CHUNKS * MAX_ACTIVE * 3;

    // GPU allocations
    cudaMalloc(&d_ops, (size_t)MAX_OPS * sizeof(uint32_t));
    cudaMalloc(&d_hist, (size_t)N_ADDR * sizeof(uint32_t));
    cudaMalloc(&d_prefix, (size_t)(N_ADDR + 1) * sizeof(uint32_t));

    d_temp = nullptr;
    d_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, d_temp_bytes, d_hist, d_prefix, N_ADDR);
    cudaMalloc(&d_temp, d_temp_bytes);

    cudaMalloc(&d_active_ids, MAX_ACTIVE * sizeof(uint32_t));
    cudaMalloc(&d_active_mask, MASK_INSTANCES_WORDS * sizeof(uint32_t));
    cudaMalloc(&d_active_first, MAX_ACTIVE * sizeof(uint32_t));
    cudaMalloc(&d_active_last, MAX_ACTIVE * sizeof(uint32_t));
    cudaMalloc(&d_fml, fml_size * sizeof(uint32_t));
    cudaMalloc(&d_result_nops, (size_t)MAX_ACTIVE * MAX_CHUNKS * sizeof(uint32_t));
    cudaMalloc(&d_meta_scalars, (size_t)MAX_ACTIVE * 4 * sizeof(uint32_t));
    cudaMalloc(&d_addr_offsets, (size_t)MAX_ACTIVE * (INSTANCE_SIZE + 1) * sizeof(uint32_t));
    cudaMalloc(&d_offset_starts, (size_t)MAX_ACTIVE * sizeof(uint32_t));
    cudaMalloc(&d_chunk_offsets, (size_t)(MAX_CHUNKS + 1) * sizeof(uint32_t));

    // Streams
    for (int s = 0; s < N_STREAMS; s++)
        cudaStreamCreate(&streams[s]);
    cudaStreamCreate(&d2h_stream);
    cudaStreamCreate(&meta_stream);

    // Host pinned
    cudaMallocHost(&h_ops, (size_t)MAX_OPS * sizeof(uint32_t));
    h_offsets_buf_size = 1ull << 28;  // 256 MB
    cudaMallocHost(&h_offsets_buf, h_offsets_buf_size);
    cudaMallocHost(&h_result_nops, (size_t)MAX_ACTIVE * MAX_CHUNKS * sizeof(uint32_t));
    cudaMallocHost(&h_meta_scalars, (size_t)MAX_ACTIVE * 4 * sizeof(uint32_t));

    // Host
    h_vals = new uint32_t[MAX_OPS];
    out_ops = new uint32_t[MAX_OPS]();
    out_vals = new uint32_t[MAX_OPS]();
    ref_ops = new uint32_t[MAX_OPS];
    ref_vals = new uint32_t[MAX_OPS];

    size_t gpu_bytes = ((size_t)MAX_OPS + N_ADDR + N_ADDR + 1) * sizeof(uint32_t)
                     + d_temp_bytes
                     + (size_t)MAX_ACTIVE * 3 * sizeof(uint32_t)
                     + fml_size * sizeof(uint32_t)
                     + (size_t)MAX_ACTIVE * MAX_CHUNKS * sizeof(uint32_t)
                     + (size_t)MAX_ACTIVE * 4 * sizeof(uint32_t)
                     + (size_t)MAX_ACTIVE * (INSTANCE_SIZE + 1) * sizeof(uint32_t)
                     + (size_t)MAX_ACTIVE * sizeof(uint32_t)
                     + (size_t)(MAX_CHUNKS + 1) * sizeof(uint32_t);
    size_t pinned_bytes = (size_t)MAX_OPS * sizeof(uint32_t)
                        + h_offsets_buf_size
                        + (size_t)MAX_ACTIVE * MAX_CHUNKS * sizeof(uint32_t)
                        + (size_t)MAX_ACTIVE * 4 * sizeof(uint32_t);
    std::cout << "=== Setup ===" << std::endl
              << std::fixed << std::setprecision(1)
              << "  GPU:         " << gpu_bytes / (double)(1<<30) << " GB" << std::endl
              << "  Host pinned: " << pinned_bytes / (double)(1<<30) << " GB" << std::endl;
}

void PairSortGPU::pick_active_instances() {
    std::mt19937 rng(123);
    std::uniform_int_distribution<uint32_t> worker_dist(0, N_WORKERS - 1);
    uint32_t worker_id = worker_dist(rng);

    memset(active_mask, 0, sizeof(active_mask));
    num_active = 0;
    for (uint32_t i = 0; i < num_instances; i++) {
        if (i % N_WORKERS == worker_id) {
            active_mask[i / 32] |= (1u << (i % 32));
            num_active++;
        }
    }

    std::cout << "  Instances: " << num_instances << ", Worker: " << worker_id
              << ", Active: " << num_active << std::endl;
    std::cout << "  Active IDs:";
    for (uint32_t i = 0; i < num_instances; i++)
        if (active_mask[i / 32] & (1u << (i % 32)))
            std::cout << " " << i;
    std::cout << std::endl;

    cudaMemcpy(d_active_mask, active_mask, MASK_INSTANCES_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    uint32_t* d_num_active_ptr;
    cudaMalloc(&d_num_active_ptr, sizeof(uint32_t));
    extract_active_ids_kernel<<<1, 1>>>(d_active_mask, num_instances, d_active_ids, d_num_active_ptr);
    cudaDeviceSynchronize();
    cudaFree(d_num_active_ptr);
}

PairSortGPU::~PairSortGPU() {
    // GPU
    cudaFree(d_ops);
    cudaFree(d_hist);
    cudaFree(d_prefix);
    cudaFree(d_temp);
    cudaFree(d_active_ids);
    cudaFree(d_active_mask);
    cudaFree(d_active_first);
    cudaFree(d_active_last);
    cudaFree(d_fml);
    cudaFree(d_result_nops);
    cudaFree(d_meta_scalars);
    cudaFree(d_addr_offsets);
    cudaFree(d_offset_starts);
    cudaFree(d_chunk_offsets);

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

void PairSortGPU::generate(uint32_t block_number) {
    std::cout << std::endl << "=== Generate (block " << block_number << ") ===" << std::endl;
    double t = omp_get_wtime();

    chunk_offsets.clear();
    chunk_offsets.push_back(0);
    uint32_t total_ops = 0;

    std::vector<uint32_t> raw_buf;
    char path[512];

    for (uint32_t file_idx = 0; ; file_idx++) {
        snprintf(path, sizeof(path), "data/%u/mem_addr_%04u.bin", block_number, file_idx);
        FILE* f = fopen(path, "rb");
        if (!f) break;

        fseek(f, 0, SEEK_END);
        size_t file_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint32_t n_entries = file_size / sizeof(uint32_t);

        raw_buf.resize(n_entries);
        (void)fread(raw_buf.data(), sizeof(uint32_t), n_entries, f);
        fclose(f);

        // Filter RAM addresses, store raw in h_ops
        uint32_t chunk_count = 0;
        for (uint32_t i = 0; i < n_entries; i++) {
            uint32_t addr = raw_buf[i];
            if (addr >= RAM_BASE && addr <= RAM_END) {
                h_ops[total_ops + chunk_count] = addr;
                chunk_count++;
            }
        }

        if (chunk_count > 0) {
            total_ops += chunk_count;
            chunk_offsets.push_back(total_ops);
        }

        if (chunk_offsets.size() - 1 >= MAX_CHUNKS) break;
    }

    n_ops = total_ops;
    num_instances = (n_ops + INSTANCE_SIZE - 1) / INSTANCE_SIZE;
    if (num_instances == 0) {
        std::cerr << "  ERROR: no RAM ops found" << std::endl;
        exit(1);
    }
    num_chunks = chunk_offsets.size() - 1;

    // Set values (global index)
    for (uint32_t i = 0; i < n_ops; i++)
        h_vals[i] = i;

    // Copy chunk_offsets to GPU
    cudaMemcpy(d_chunk_offsets, chunk_offsets.data(),
               (num_chunks + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t last_inst_size = n_ops - (num_instances - 1) * INSTANCE_SIZE;
    std::cout << std::fixed << std::setprecision(2)
              << "  " << n_ops << " ops in " << num_chunks << " chunks, "
              << num_instances << " instances (last: " << last_inst_size << " ops)"
              << " (" << (omp_get_wtime() - t) * 1e3 << " ms)" << std::endl;
}

void PairSortGPU::gpu_metadata() {
    std::cout << std::endl << "=== GPU Metadata ===" << std::endl;
    double t_total = omp_get_wtime(), t;

    // Clear histogram
    cudaMemset(d_hist, 0, (size_t)N_ADDR * sizeof(uint32_t));

    // H2D + shift + histogram pipelined by chunks
    t = omp_get_wtime();
    int hist_block, hist_grid;
    cudaOccupancyMaxPotentialBlockSize(&hist_grid, &hist_block, histogram_kernel, 0, 0);
    int shift_block, shift_grid;
    cudaOccupancyMaxPotentialBlockSize(&shift_grid, &shift_block, shift_addresses_kernel, 0, 0);
    for (uint32_t c = 0; c < num_chunks - 1; c++) {
        int s = c % N_STREAMS;
        uint32_t offset = chunk_offsets[c];
        uint32_t chunk_size = chunk_offsets[c + 1] - offset;
        size_t chunk_bytes = (size_t)chunk_size * sizeof(uint32_t);
        cudaMemcpyAsync(d_ops + offset, h_ops + offset, chunk_bytes,
                        cudaMemcpyHostToDevice, streams[s]);
        shift_addresses_kernel<<<shift_grid, shift_block, 0, streams[s]>>>(
            d_ops + offset, chunk_size);
        histogram_kernel<<<hist_grid, hist_block, 0, streams[s]>>>(
            d_ops + offset, d_hist, chunk_size);
    }
    cudaDeviceSynchronize();  // all prior chunks done

    double t_last_chunk = omp_get_wtime();
    {
        uint32_t c = num_chunks - 1;
        int s = c % N_STREAMS;
        uint32_t offset = chunk_offsets[c];
        uint32_t chunk_size = chunk_offsets[c + 1] - offset;
        size_t chunk_bytes = (size_t)chunk_size * sizeof(uint32_t);
        cudaMemcpyAsync(d_ops + offset, h_ops + offset, chunk_bytes,
                        cudaMemcpyHostToDevice, streams[s]);
        shift_addresses_kernel<<<shift_grid, shift_block, 0, streams[s]>>>(
            d_ops + offset, chunk_size);
        histogram_kernel<<<hist_grid, hist_block, 0, streams[s]>>>(
            d_ops + offset, d_hist, chunk_size);
    }
    cudaDeviceSynchronize();
    std::cout << std::fixed << std::setprecision(2)
              << "  H2D + histogram:  " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // Prefix sum
    t = omp_get_wtime();
    cub::DeviceScan::ExclusiveSum(d_temp, d_temp_bytes, d_hist, d_prefix, N_ADDR);
    cudaMemcpy(d_prefix + N_ADDR, &n_ops, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    std::cout << std::fixed << std::setprecision(2)
              << "  Prefix sum:       " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // GPU: active instance boundaries + chunk FML counts
    t = omp_get_wtime();
    const size_t active_bytes = num_active * sizeof(uint32_t);

    instance_boundaries_kernel<<<1, num_active>>>(
        d_prefix, d_active_ids, d_active_first, d_active_last,
        d_offset_starts, num_active, num_instances, n_ops);

    // FML kernel
    size_t runtime_fml_size = (size_t)num_active * num_chunks * 3;
    cudaMemset(d_fml, 0, runtime_fml_size * sizeof(uint32_t));
    int fml_block, fml_grid;
    cudaOccupancyMaxPotentialBlockSize(&fml_grid, &fml_block, chunk_fml_count_kernel, 0, 0);
    chunk_fml_count_kernel<<<fml_grid, fml_block>>>(
        d_ops, d_active_first, d_active_last, d_chunk_offsets,
        d_fml, num_active, num_chunks, n_ops);
    cudaDeviceSynchronize();

    // D2H: active boundaries
    cudaMemcpy(h_active_first.data(), d_active_first, active_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_active_last.data(), d_active_last, active_bytes, cudaMemcpyDeviceToHost);
    std::cout << std::fixed << std::setprecision(2)
              << "  Boundaries + FML: " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // --- Build metas + addr_offsets: all on GPU, overlapped ---
    t = omp_get_wtime();

    // Compute offset_starts + total_addrs from already-D2H'd boundaries
    std::vector<uint32_t> h_offset_starts(num_active);
    uint32_t total_addrs = 0;
    for (uint32_t i = 0; i < num_active; i++) {
        h_offset_starts[i] = total_addrs;
        total_addrs += h_active_last[i] - h_active_first[i] + 1;
    }
    size_t total_offsets_bytes = (size_t)total_addrs * sizeof(uint32_t);

    // Launch build_metas_kernel on meta_stream
    build_metas_kernel<<<num_active, 256, 0, meta_stream>>>(
        d_fml, d_prefix, d_active_ids, d_active_first, d_active_last,
        d_result_nops, d_meta_scalars, num_active, num_chunks, n_ops);

    // Launch addr_offsets kernel on d2h_stream
    compute_addr_offsets_kernel<<<num_active, 1024, 0, d2h_stream>>>(
        d_prefix, d_active_ids, d_active_first, d_active_last,
        d_addr_offsets, d_offset_starts, num_active);

    // Queue D2H for meta results on meta_stream
    cudaMemcpyAsync(h_meta_scalars, d_meta_scalars, num_active * 4 * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, meta_stream);
    cudaMemcpyAsync(h_result_nops, d_result_nops, (size_t)num_active * num_chunks * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, meta_stream);

    // Queue addr_offsets D2H
    cudaMemcpyAsync(h_offsets_buf, d_addr_offsets, total_offsets_bytes,
                    cudaMemcpyDeviceToHost, d2h_stream);

    // Wait for both streams
    cudaStreamSynchronize(meta_stream);
    cudaStreamSynchronize(d2h_stream);

    uint32_t ai = 0;
    for (uint32_t i = 0; i < num_instances && ai < num_active; i++) {
        if (!(active_mask[i / 32] & (1u << (i % 32)))) continue;
        uint32_t* sc = h_meta_scalars + ai * 4;
        metas[ai].inst_id = i;
        metas[ai].first_addr = h_active_first[ai];
        metas[ai].last_addr = h_active_last[ai];
        metas[ai].first_addr_chunk = sc[0];
        metas[ai].first_addr_skip = sc[1];
        metas[ai].last_addr_chunk = sc[2];
        metas[ai].last_addr_include = sc[3];
        metas[ai].addr_offsets = h_offsets_buf + h_offset_starts[ai];
        metas[ai].nops_per_chunk = h_result_nops + (size_t)ai * num_chunks;
        ai++;
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
    for (size_t idx = 0; idx < num_active; idx++) {
        auto& m = metas[idx];
        bool single_addr = (m.first_addr == m.last_addr);

        uint32_t inst_size = std::min(INSTANCE_SIZE, n_ops - m.inst_id * INSTANCE_SIZE);
        uint32_t out_base = m.inst_id * INSTANCE_SIZE;
        uint32_t total_written = 0;
        for (uint32_t chunk_gid = 0; chunk_gid < num_chunks; chunk_gid++) {
            uint32_t expected = m.nops_per_chunk[chunk_gid];
            if (expected == 0) continue;
            uint32_t chunk_start = chunk_offsets[chunk_gid];
            uint32_t chunk_size = chunk_offsets[chunk_gid + 1] - chunk_start;
            uint32_t found = 0;
            uint32_t first_found = 0;
            uint32_t last_found = 0;

            for (uint32_t j = 0; j < chunk_size && found < expected; j++) {
                uint32_t raw = h_ops[chunk_start + j];
                uint32_t addr = (raw - RAM_BASE) >> ADDR_SHIFT;
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
                out_ops[out_pos] = raw;
                out_vals[out_pos] = h_vals[chunk_start + j];
                total_written++;
                if (total_written >= inst_size) break;
            }
            if (total_written >= inst_size) break;
        }
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  " << num_active << " instances in " << (omp_get_wtime() - t_total) * 1e3 << " ms" << std::endl;
}

void PairSortGPU::reference_sort() {
    std::cout << std::endl << "=== Verify ===" << std::endl;
    double t = omp_get_wtime();
    struct Triple { uint32_t key, pos, val; };
    std::vector<Triple> triples(n_ops);
    for (uint32_t i = 0; i < n_ops; i++)
        triples[i] = {h_ops[i], i, h_vals[i]};
    std::sort(triples.begin(), triples.end(), [](const Triple& a, const Triple& b) {
        return a.key < b.key || (a.key == b.key && a.pos < b.pos);
    });
    for (uint32_t i = 0; i < n_ops; i++) {
        ref_ops[i] = triples[i].key;
        ref_vals[i] = triples[i].val;
    }
    std::cout << std::fixed << std::setprecision(2)
              << "  Reference sort:   " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;
}

void PairSortGPU::verify() {
    double t = omp_get_wtime();
    uint32_t total_verified = 0;
    for (uint32_t idx = 0; idx < num_active; idx++) {
        auto& m = metas[idx];
        uint32_t inst = m.inst_id;
        uint32_t inst_size = std::min(INSTANCE_SIZE, n_ops - inst * INSTANCE_SIZE);
        uint32_t start = inst * INSTANCE_SIZE;

        for (uint32_t j = 0; j < inst_size; j++) {
            uint32_t ind = start + j;
            if (out_ops[ind] != ref_ops[ind] || out_vals[ind] != ref_vals[ind]) {
                std::cout << "MISMATCH at ref position " << ind << " (instance " << inst
                          << ", local " << j << "): got (" << out_ops[ind] << "," << out_vals[ind]
                          << ") expected (" << ref_ops[ind] << "," << ref_vals[ind] << ")" << std::endl;
                return;
            }
        }
        total_verified += inst_size;
    }
    std::cout << std::fixed << std::setprecision(2)
              << "  Verify:           " << (omp_get_wtime() - t) * 1e3
              << " ms -- " << total_verified << " entries, " << num_active << " instances OK" << std::endl;
}

// =====================================================================

int main(int argc, char** argv) {
    bool do_verify = false;
    uint32_t block_number = 0;
    bool have_block = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-v") do_verify = true;
        else { block_number = std::strtoul(argv[i], nullptr, 10); have_block = true; }
    }
    if (!have_block) {
        std::cerr << "Usage: " << argv[0] << " <block_number> [-v]" << std::endl;
        return 1;
    }

    PairSortGPU app;
    app.generate(block_number);
    app.pick_active_instances();
    app.gpu_metadata();
    app.cpu_fill();

    if (do_verify) {
        app.reference_sort();
        app.verify();
    }
}
