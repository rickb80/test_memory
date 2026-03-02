#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <vector>
#include <cub/device/device_scan.cuh>

// ---------- Constants ----------
constexpr uint32_t N_ACCS        = 1u << 30;
constexpr uint32_t N_ADDR        = 1u << 27;
constexpr uint32_t INSTANCE_SIZE = 1u << 22;
constexpr uint32_t NUM_INSTANCES = N_ACCS / INSTANCE_SIZE;  // 128
constexpr uint32_t NUM_ACTIVE    = NUM_INSTANCES / 16;      // how many instances to actually fill
constexpr uint32_t CHUNK_SIZE    = 1u << 18;
constexpr uint32_t NUM_CHUNKS    = N_ACCS / CHUNK_SIZE;  // 1024
    
// ---------- GPU kernels ----------

__global__ void histogram_kernel(const uint32_t* accs, uint32_t* hist, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        atomicAdd(&hist[accs[i]], 1);
}

__global__ void instance_boundaries_kernel(
    const uint32_t* prefix,
    uint32_t* d_first_addr,
    uint32_t* d_last_addr)
{
    uint32_t inst = threadIdx.x;
    if (inst >= NUM_INSTANCES) return;

    uint32_t inst_start = inst * INSTANCE_SIZE;
    uint32_t inst_end = (inst < NUM_INSTANCES - 1) ? (inst + 1) * INSTANCE_SIZE : N_ACCS;

    // Find largest address <= inst_start
    uint32_t lo = 0, hi = N_ADDR;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo + 1) / 2;
        if (prefix[mid] <= inst_start) lo = mid;
        else hi = mid - 1;
    }
    uint32_t first_addr = lo;
    d_first_addr[inst] = first_addr;

    // Find largest address <= inst_end - 1
    lo = first_addr; hi = N_ADDR;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo + 1) / 2;
        if (prefix[mid] < inst_end) lo = mid;
        else hi = mid - 1;
    }
    d_last_addr[inst] = lo;
}

__global__ void chunk_inst_count_kernel(
    const uint32_t* accs,
    const uint32_t* d_first_addr,
    const uint32_t* d_last_addr,
    uint32_t* chunk_inst,
    uint32_t n)
{
    __shared__ uint32_t s_first[NUM_INSTANCES];
    __shared__ uint32_t s_last[NUM_INSTANCES];
    if (threadIdx.x < NUM_INSTANCES) {
        s_first[threadIdx.x] = d_first_addr[threadIdx.x];
        s_last[threadIdx.x] = d_last_addr[threadIdx.x];
    }
    __syncthreads();

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        uint32_t addr = accs[i];
        uint32_t chunk_id = i / CHUNK_SIZE;

        // Find first instance where addr <= last_addr (first containing)
        uint32_t lo = 0, hi = NUM_INSTANCES - 1;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (s_last[mid] < addr) lo = mid + 1;
            else hi = mid;
        }
        uint32_t first_inst = lo;

        // Find last instance where first_addr <= addr (last containing)
        lo = first_inst; hi = NUM_INSTANCES - 1;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo + 1) / 2;
            if (s_first[mid] <= addr) lo = mid;
            else hi = mid - 1;
        }
        uint32_t last_inst = lo;

        // Count toward all instances this address spans
        for (uint32_t inst = first_inst; inst <= last_inst; inst++)
            atomicAdd(&chunk_inst[chunk_id * NUM_INSTANCES + inst], 1);
    }
}


struct InstanceMeta {
    uint32_t inst_id;
    uint32_t first_addr, last_addr;
    std::vector<uint32_t> chunks;         // which input chunks to scan
    std::vector<uint32_t> accs_expected; 
    std::vector<int32_t> instance_offsets; // local offsets within instance (0-based, can be negative)
};

// ---------- CPU helpers ----------

std::vector<uint32_t> pick_random_instances() {
    std::mt19937 rng(123);
    std::vector<uint32_t> active;
    while (active.size() < NUM_ACTIVE) {
        uint32_t r = rng() % NUM_INSTANCES;
        bool found = false;
        for (uint32_t v : active) if (v == r) { found = true; break; }
        if (!found) active.push_back(r);
    }
    std::sort(active.begin(), active.end());
    std::cout << "Active instances (" << NUM_ACTIVE << " of " << NUM_INSTANCES << "):";
    for (uint32_t id : active) std::cout << " " << id;
    std::cout << std::endl;
    return active;
}

void generate_pairs(uint32_t* accs, uint32_t* vals) {
    std::cout << "Generating " << N_ACCS << " pairs..." << std::endl;
    double t = omp_get_wtime();
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> addr_dist(0, N_ADDR - 1);
    std::uniform_int_distribution<uint32_t> val_dist(0, UINT32_MAX);
    for (uint32_t i = 0; i < N_ACCS; i++) {
        accs[i] = addr_dist(rng);
        vals[i] = val_dist(rng);
    }
    std::cout << std::fixed << std::setprecision(2)
              << "  Generated in " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;
}

void gpu_metadata(const uint32_t* h_accs,
                  const std::vector<uint32_t>& active_instances,
                  std::vector<InstanceMeta>& metas) {
    std::cout << std::endl << "=== GPU Metadata Phase ===" << std::endl;
    double t_total = omp_get_wtime(), t;

    // Allocations
    t = omp_get_wtime();
    const size_t accs_bytes  = (size_t)N_ACCS * sizeof(uint32_t);
    const size_t hist_bytes = (size_t)N_ADDR * sizeof(uint32_t);
    const size_t pfx_bytes  = (size_t)(N_ADDR + 1) * sizeof(uint32_t);
    const size_t ci_bytes   = (size_t)NUM_CHUNKS * NUM_INSTANCES * sizeof(uint32_t);
    const size_t inst_bytes = NUM_INSTANCES * sizeof(uint32_t);

    uint32_t* d_accs       = nullptr;
    uint32_t* d_hist       = nullptr;
    uint32_t* d_prefix     = nullptr;
    uint32_t* d_first_addr = nullptr;
    uint32_t* d_last_addr  = nullptr;
    uint32_t* d_chunk_inst = nullptr;

    cudaMalloc(&d_accs,       accs_bytes);
    cudaMalloc(&d_hist,       hist_bytes);
    cudaMalloc(&d_prefix,     pfx_bytes);
    cudaMalloc(&d_first_addr, inst_bytes);
    cudaMalloc(&d_last_addr,  inst_bytes);
    cudaMalloc(&d_chunk_inst, ci_bytes);
    cudaMemset(d_hist, 0,       hist_bytes);
    cudaMemset(d_chunk_inst, 0, ci_bytes);

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    // to get temp_bytes, only!
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_hist, d_prefix, N_ADDR);
    cudaMalloc(&d_temp, temp_bytes);

    std::cout << std::fixed << std::setprecision(2)
              << "  GPU alloc (" << (accs_bytes + hist_bytes + pfx_bytes + ci_bytes) / 1e9
              << " GB + " << std::setprecision(1) << temp_bytes / 1e6
              << " MB CUB temp) in " << std::setprecision(2)
              << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // H2D + histogram pipelined by chunks
    t = omp_get_wtime();
    constexpr int N_STREAMS = 4;
    cudaStream_t streams[N_STREAMS];
    for (int s = 0; s < N_STREAMS; s++)
        cudaStreamCreate(&streams[s]);

    const size_t chunk_bytes = CHUNK_SIZE * sizeof(uint32_t);
    for (uint32_t c = 0; c < NUM_CHUNKS - 1; c++) {
        int s = c % N_STREAMS;
        size_t offset = (size_t)c * CHUNK_SIZE;
        cudaMemcpyAsync(d_accs + offset, h_accs + offset, chunk_bytes,cudaMemcpyHostToDevice, streams[s]);
        histogram_kernel<<<32, 256, 0, streams[s]>>>(d_accs + offset, d_hist, CHUNK_SIZE);
    }
    cudaDeviceSynchronize();  // all prior chunks done (simulates real-time generation)

    double t_last_chunk = omp_get_wtime();
    {
        uint32_t c = NUM_CHUNKS - 1;
        int s = c % N_STREAMS;
        size_t offset = (size_t)c * CHUNK_SIZE;
        cudaMemcpyAsync(d_accs + offset, h_accs + offset, chunk_bytes,
                        cudaMemcpyHostToDevice, streams[s]);
        histogram_kernel<<<32, 256, 0, streams[s]>>>(d_accs + offset, d_hist, CHUNK_SIZE);
    }
    cudaDeviceSynchronize();
    std::cout << std::fixed
              << "  H2D + histogram pipelined (" << std::setprecision(1)
              << accs_bytes / 1e9 << " GB, " << NUM_CHUNKS << " chunks, "
              << N_STREAMS << " streams) in " << std::setprecision(2)
              << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    for (int s = 0; s < N_STREAMS; s++)
        cudaStreamDestroy(streams[s]);

    // Prefix sum
    t = omp_get_wtime();
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_hist, d_prefix, N_ADDR);
    uint32_t n_accs_val = N_ACCS;
    cudaMemcpy(d_prefix + N_ADDR, &n_accs_val, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    std::cout << std::fixed << std::setprecision(2)
              << "  GPU prefix sum in " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // GPU: instance boundaries (all 128)
    t = omp_get_wtime();
    instance_boundaries_kernel<<<1, 256>>>(d_prefix, d_first_addr, d_last_addr);
    cudaDeviceSynchronize();
    std::cout << std::fixed << std::setprecision(2)
              << "  GPU instance boundaries in " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // GPU: chunk-instance counts
    t = omp_get_wtime();
    chunk_inst_count_kernel<<<4096, 256>>>(d_accs, d_first_addr, d_last_addr, d_chunk_inst, N_ACCS);
    cudaDeviceSynchronize();
    std::cout << std::fixed << std::setprecision(2)
              << "  GPU chunk-instance counts in " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    //  D2H: boundaries + chunk counts
    t = omp_get_wtime();
    std::vector<uint32_t> h_first_addr(NUM_INSTANCES);
    std::vector<uint32_t> h_last_addr(NUM_INSTANCES);
    cudaMemcpy(h_first_addr.data(), d_first_addr, inst_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_last_addr.data(),  d_last_addr,  inst_bytes, cudaMemcpyDeviceToHost);

    std::vector<uint32_t> chunk_inst_counts(NUM_CHUNKS * NUM_INSTANCES);
    cudaMemcpy(chunk_inst_counts.data(), d_chunk_inst, ci_bytes, cudaMemcpyDeviceToHost);
    std::cout << std::fixed << std::setprecision(1)
              << "  D2H boundaries + chunk counts (" << (inst_bytes * 2 + ci_bytes) / 1e3
              << " KB) in " << std::setprecision(2)
              << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // --- Build InstanceMeta for active instances (D2H + CPU) ---
    t = omp_get_wtime();
    metas.resize(active_instances.size());
    size_t total_prefix_bytes = 0;

    double t_first_full_meta = 0;
    for (size_t i = 0; i < active_instances.size(); i++) {
        uint32_t inst = active_instances[i];
        uint32_t fa = h_first_addr[inst];
        uint32_t la = h_last_addr[inst];
        uint32_t n_entries = la - fa + 1;

        metas[i].inst_id    = inst;
        metas[i].first_addr = fa;
        metas[i].last_addr  = la;

        // D2H: fetch prefix slice and compute instance offsets
        std::vector<uint32_t> tmp(n_entries);
        cudaMemcpy(tmp.data(), d_prefix + fa,
                   n_entries * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        total_prefix_bytes += n_entries * sizeof(uint32_t);

        uint32_t out_start = inst * INSTANCE_SIZE;
        metas[i].instance_offsets.resize(n_entries);
        for (uint32_t j = 0; j < n_entries; j++)
            metas[i].instance_offsets[j] = (int32_t)(tmp[j] - out_start);

        // CPU: build chunks list
        auto& chunks = metas[i].chunks;
        auto& accs_expected = metas[i].accs_expected;
        chunks.reserve(NUM_CHUNKS);
        accs_expected.reserve(NUM_CHUNKS);
        for (uint32_t c = 0; c < NUM_CHUNKS; c++) {
            uint32_t count = chunk_inst_counts[c * NUM_INSTANCES + inst];
            if (count > 0) {
                chunks.push_back(c);
                accs_expected.push_back(count);
            }
        }

        if (i == 0) t_first_full_meta = omp_get_wtime();
    }
    std::cout << std::fixed
              << "  Build InstanceMeta for " << NUM_ACTIVE << " instances ("
              << std::setprecision(1) << total_prefix_bytes / 1e6
              << " MB D2H) in " << std::setprecision(2)
              << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // --- Cleanup ---
    cudaFree(d_accs);
    cudaFree(d_hist);
    cudaFree(d_prefix);
    cudaFree(d_first_addr);
    cudaFree(d_last_addr);
    cudaFree(d_chunk_inst);
    cudaFree(d_temp);

    std::cout << std::fixed << std::setprecision(2)
              << "GPU METADATA TOTAL: " << (omp_get_wtime() - t_total) * 1e3 << " ms" << std::endl
              << "  ** Last chunk to first full instance metadata: "
              << (t_first_full_meta - t_last_chunk) * 1e3 << " ms **" << std::endl
              << "  ** Last chunk to all metadata ready: "
              << (omp_get_wtime() - t_last_chunk) * 1e3 << " ms **" << std::endl;

    /*for (const auto& m : metas) {
        std::cout << "  Instance " << std::setw(2) << m.inst_id
                  << ": addrs " << std::setw(8) << m.first_addr
                  << ".." << std::left << std::setw(8) << m.last_addr << std::right
                  << ", chunks=" << (uint32_t)m.chunks.size() << "/" << NUM_CHUNKS
                  << std::endl;
    }*/
}


void cpu_fill_instances(const uint32_t* h_accs, const uint32_t* h_vals,
                        uint32_t* out_accs, uint32_t* out_vals,
                        std::vector<InstanceMeta>& metas) {
    std::cout << std::endl << "=== CPU Fill Phase (" << (uint32_t)metas.size() << " instances) ===" << std::endl;
    double t_total = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t idx = 0; idx < metas.size(); idx++) {
        auto& m = metas[idx];
        uint32_t out_base = m.inst_id * INSTANCE_SIZE;

        uint32_t total_written = 0;
        for (size_t ci = 0; ci < m.chunks.size(); ci++) {
            size_t base = (size_t)m.chunks[ci] * CHUNK_SIZE;
            uint32_t expected = m.accs_expected[ci];
            uint32_t found = 0;
            for (uint32_t j = 0; j < CHUNK_SIZE; j++) {
                uint32_t addr = h_accs[base + j];
                if (addr < m.first_addr || addr > m.last_addr) continue;
                uint32_t ind = addr - m.first_addr;
                int32_t pos = m.instance_offsets[ind]++;
                if (pos >= 0 && pos < (int32_t)INSTANCE_SIZE) {
                    out_accs[out_base + pos] = addr;
                    out_vals[out_base + pos] = h_vals[base + j];
                    total_written++;
                }
                if (++found >= expected) break;
            }
            if (total_written >= INSTANCE_SIZE) break;
        }
    }

    std::cout << std::fixed << std::setprecision(2)
              << "CPU FILL TOTAL: " << (omp_get_wtime() - t_total) * 1e3 << " ms" << std::endl;
}

void reference_sort(const uint32_t* accs, const uint32_t* vals,
                    uint32_t* ref_accs, uint32_t* ref_vals) {
    std::cout << std::endl << "=== Reference sort" << std::endl;
    double t = omp_get_wtime();
    struct Triple { uint32_t key, pos, val; };
    std::vector<Triple> triples(N_ACCS);
    for (uint32_t i = 0; i < N_ACCS; i++)
        triples[i] = {accs[i], i, vals[i]};
    std::sort(triples.begin(), triples.end(), [](const Triple& a, const Triple& b) {
        return a.key < b.key || (a.key == b.key && a.pos < b.pos);
    });
    for (uint32_t i = 0; i < N_ACCS; i++) {
        ref_accs[i] = triples[i].key;
        ref_vals[i] = triples[i].val;
    }
    std::cout << std::fixed << std::setprecision(2)
              << "  Reference sorted in " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;
}

void verify_instances(const uint32_t* out_accs, const uint32_t* out_vals,
                      const uint32_t* ref_accs, const uint32_t* ref_vals,
                      const std::vector<InstanceMeta>& metas) {
    std::cout << std::endl << "Verifying " << metas.size() << " active instances..." << std::endl;
    double t = omp_get_wtime();
    uint32_t total_verified = 0;
    for (const auto& m : metas) {
        uint32_t gid = m.inst_id * INSTANCE_SIZE;
        for (uint32_t i = gid; i < gid + INSTANCE_SIZE; i++) {
            if (out_accs[i] != ref_accs[i] || out_vals[i] != ref_vals[i]) {
                std::cout << "MISMATCH at position " << i << " (instance " << m.inst_id
                          << "): got (" << out_accs[i] << "," << out_vals[i]
                          << ") expected (" << ref_accs[i] << "," << ref_vals[i] << ")" << std::endl;
                return;
            }
        }
        total_verified += INSTANCE_SIZE;
    }
    std::cout << std::fixed << std::setprecision(2)
              << "  Verified in " << (omp_get_wtime() - t) * 1e3
              << " ms -- " << total_verified << " entries across "
              << metas.size() << " instances match!" << std::endl;
}




// =====================================================================

int main() {
    // Pick random active instances
    std::vector<uint32_t> active = pick_random_instances();

    // Generate pairs
    uint32_t* accs = nullptr;
    cudaMallocHost(&accs, (size_t)N_ACCS * sizeof(uint32_t));  
    uint32_t* vals = new uint32_t[N_ACCS];
    generate_pairs(accs, vals);

    // GPU metadata
    std::vector<InstanceMeta> metas;
    gpu_metadata(accs, active, metas);

    // CPU fill active instances
    uint32_t* out_accs = new uint32_t[N_ACCS];
    uint32_t* out_vals = new uint32_t[N_ACCS];
    cpu_fill_instances(accs, vals, out_accs, out_vals, metas);

    // CPU reference (full sort — needed to verify)
    uint32_t* ref_accs = new uint32_t[N_ACCS];
    uint32_t* ref_vals = new uint32_t[N_ACCS];
    reference_sort(accs, vals, ref_accs, ref_vals);

    // Verify only active instance ranges
    verify_instances(out_accs, out_vals, ref_accs, ref_vals, metas);

    cudaFreeHost(accs);
    delete[] vals;
    delete[] out_accs;
    delete[] out_vals;
    delete[] ref_accs;
    delete[] ref_vals;
}
