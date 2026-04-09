#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <span>
#include <vector>

// =====================================================================
// Constants (same as main_real.cu)
// =====================================================================

constexpr uint32_t N_ADDR_ROM   = 1u << 24;   // 16M
constexpr uint32_t N_ADDR_INPUT = 1u << 24;   // 16M
constexpr uint32_t N_ADDR_RAM   = 1u << 26;   // 64M
constexpr uint32_t N_ADDR = N_ADDR_ROM + N_ADDR_INPUT + N_ADDR_RAM;  // 96M

constexpr uint32_t INSTANCE_SIZE = 1u << 22;  // 4M entries per instance

constexpr uint32_t MAX_INST_ROM   = 32;
constexpr uint32_t MAX_INST_INPUT = 32;
constexpr uint32_t MAX_INST_RAM   = 256;
constexpr uint32_t MAX_INSTANCES  = MAX_INST_ROM + MAX_INST_INPUT + MAX_INST_RAM;
constexpr uint32_t MAX_INST[3]    = {MAX_INST_ROM, MAX_INST_INPUT, MAX_INST_RAM};
constexpr uint32_t MASK_WORDS     = (MAX_INSTANCES + 31) / 32;

constexpr uint32_t MAX_OPS    = (uint32_t)MAX_INSTANCES * INSTANCE_SIZE;
constexpr uint32_t N_WORKERS  = 16;
constexpr uint32_t MAX_ACTIVE = (MAX_INSTANCES + N_WORKERS - 1) / N_WORKERS;
constexpr uint32_t MAX_CHUNKS = 4096;

// Region identifiers
constexpr uint8_t REGION_ROM            = 0;
constexpr uint8_t REGION_INPUT          = 1;
constexpr uint8_t REGION_RAM            = 2;
constexpr const char* REGION_NAME[3]    = {"ROM", "INPUT", "RAM"};
constexpr uint32_t REGION_ADDR_START[3] = {0, N_ADDR_ROM, N_ADDR_ROM + N_ADDR_INPUT};
constexpr uint32_t REGION_N_ADDR[3]     = {N_ADDR_ROM, N_ADDR_INPUT, N_ADDR_RAM};

// Coarse histogram for CPU metadata
constexpr int      COARSE_SHIFT = 14;
constexpr uint32_t COARSE_BINS  = (N_ADDR >> COARSE_SHIFT) + 1;

// =====================================================================
// Helper Functions (same as main_real.cu)
// =====================================================================

inline uint32_t compact_addr(uint32_t raw) {
    if (raw >= 0xA0000000u)
        return ((raw - 0xA0000000u) >> 3) + N_ADDR_ROM + N_ADDR_INPUT;
    if (raw >= 0x90000000u)
        return ((raw - 0x90000000u) >> 3) + N_ADDR_ROM;
    return (raw - 0x80000000u) >> 3;
}

inline uint32_t expand_addr(uint32_t compact) {
    if (compact >= N_ADDR_ROM + N_ADDR_INPUT)
        return ((compact - N_ADDR_ROM - N_ADDR_INPUT) << 3) + 0xA0000000u;
    if (compact >= N_ADDR_ROM)
        return ((compact - N_ADDR_ROM) << 3) + 0x90000000u;
    return (compact << 3) + 0x80000000u;
}

// =====================================================================
// InstanceMeta (same as GPU version)
// =====================================================================

struct InstanceMeta {
    uint32_t inst_id;            // local instance ID within region
    uint8_t  type;               // REGION_ROM / REGION_INPUT / REGION_RAM
    uint32_t first_addr;         // first raw hardware address covered
    uint32_t last_addr;          // last raw hardware address covered
    std::span<const uint32_t> nops_per_chunk;
    std::span<uint32_t>       addr_offsets;
    uint32_t first_addr_chunk;
    uint32_t first_addr_skip;
    uint32_t last_addr_chunk;
    uint32_t last_addr_include;
};

// =====================================================================
// PairSortCPU
// =====================================================================

class PairSortCPU {
    // --- Host memory ---
    uint32_t* ops;
    uint32_t* vals;
    uint32_t* compact_keys;

    // --- Per-instance metadata buffers ---
    uint32_t* result_nops;     // [MAX_ACTIVE * MAX_CHUNKS]
    uint32_t* offsets_buf;     // packed addr_offsets for all instances

    // Per-region output arrays
    uint32_t* out_ops_rom;    uint32_t* out_vals_rom;
    uint32_t* out_ops_input;  uint32_t* out_vals_input;
    uint32_t* out_ops_ram;    uint32_t* out_vals_ram;

    // Per-region reference arrays
    uint32_t* ref_ops_rom;    uint32_t* ref_vals_rom;
    uint32_t* ref_ops_input;  uint32_t* ref_vals_input;
    uint32_t* ref_ops_ram;    uint32_t* ref_vals_ram;

    // --- Active instance tracking ---
    uint32_t active_mask[MASK_WORDS];
    uint32_t active_local_ids[MAX_ACTIVE];
    std::vector<InstanceMeta> metas;

    // --- Runtime state ---
    uint32_t num_ops;
    uint32_t num_chunks;
    uint32_t num_instances;
    uint32_t num_active;
    std::vector<uint32_t> chunk_offsets;

    // --- Per-region state ---
    uint32_t num_inst[3];
    uint32_t num_active_per[3];
    uint32_t active_offset[3];
    uint32_t region_ops_start[3];
    uint32_t region_n_ops[3];

public:
    PairSortCPU();
    ~PairSortCPU();

    void generate(uint32_t block_number);
    void cpu_metadata();
    void cpu_fill();
    void reference_sort();
    void verify();

private:
    void create_active_mask();
    void pick_active_instances();
};

// =====================================================================
// PairSortCPU Implementation
// =====================================================================

PairSortCPU::PairSortCPU()
    : metas(MAX_ACTIVE),
      num_ops(0), num_chunks(0), num_instances(0), num_active(0),
      num_inst{}, num_active_per{}, active_offset{}, region_ops_start{}, region_n_ops{},
      vals(nullptr), compact_keys(nullptr),
      result_nops(nullptr), offsets_buf(nullptr)
{
    ops = new uint32_t[MAX_OPS];

    size_t rom_sz   = (size_t)MAX_INST_ROM   * INSTANCE_SIZE;
    size_t input_sz = (size_t)MAX_INST_INPUT * INSTANCE_SIZE;
    size_t ram_sz   = (size_t)MAX_INST_RAM   * INSTANCE_SIZE;

    out_ops_rom   = new uint32_t[rom_sz]();    out_vals_rom   = new uint32_t[rom_sz]();
    out_ops_input = new uint32_t[input_sz]();  out_vals_input = new uint32_t[input_sz]();
    out_ops_ram   = new uint32_t[ram_sz]();    out_vals_ram   = new uint32_t[ram_sz]();

    ref_ops_rom   = new uint32_t[rom_sz];      ref_vals_rom   = new uint32_t[rom_sz];
    ref_ops_input = new uint32_t[input_sz];    ref_vals_input = new uint32_t[input_sz];
    ref_ops_ram   = new uint32_t[ram_sz];      ref_vals_ram   = new uint32_t[ram_sz];

    std::cout << "=== Setup ===" << std::endl;
}

PairSortCPU::~PairSortCPU() {
    delete[] ops;
    delete[] vals;
    delete[] compact_keys;
    delete[] result_nops;
    delete[] offsets_buf;
    delete[] out_ops_rom;    delete[] out_vals_rom;
    delete[] out_ops_input;  delete[] out_vals_input;
    delete[] out_ops_ram;    delete[] out_vals_ram;
    delete[] ref_ops_rom;    delete[] ref_vals_rom;
    delete[] ref_ops_input;  delete[] ref_vals_input;
    delete[] ref_ops_ram;    delete[] ref_vals_ram;
}

void PairSortCPU::create_active_mask() {
    std::mt19937 rng(123);
    std::uniform_int_distribution<uint32_t> dist(0, N_WORKERS - 1);
    uint32_t worker_id = dist(rng);

    memset(active_mask, 0, sizeof(active_mask));
    for (uint32_t i = 0; i < MAX_INSTANCES; i++)
        if (i % N_WORKERS == worker_id)
            active_mask[i / 32] |= (1u << (i % 32));
}

void PairSortCPU::pick_active_instances() {
    uint32_t pos = 0;
    uint32_t gid_base = 0;
    for (uint8_t r = 0; r < 3; r++) {
        num_active_per[r] = 0;
        active_offset[r] = pos;
        for (uint32_t lid = 0; lid < num_inst[r]; lid++) {
            uint32_t gid = gid_base + lid;
            if (active_mask[gid / 32] & (1u << (gid % 32)))
                active_local_ids[pos + num_active_per[r]++] = lid;
        }
        pos += num_active_per[r];
        gid_base += num_inst[r];
    }
    num_active = num_active_per[0] + num_active_per[1] + num_active_per[2];

    std::cout << "  Instances: " << num_instances
              << " (ROM: " << num_inst[0]
              << ", INPUT: " << num_inst[1]
              << ", RAM: " << num_inst[2] << ")"
              << ", Active: " << num_active
              << " (ROM: " << num_active_per[0]
              << ", INPUT: " << num_active_per[1]
              << ", RAM: " << num_active_per[2] << ")"
              << std::endl;
    std::cout << "  Active global IDs:";
    for (uint32_t i = 0; i < num_instances; i++)
        if (active_mask[i / 32] & (1u << (i % 32)))
            std::cout << " " << i;
    std::cout << std::endl;
}

// =====================================================================
// Pipeline stages
// =====================================================================

void PairSortCPU::generate(uint32_t block_number) {
    std::cout << std::endl << "=== Generate (block " << block_number << ") ===" << std::endl;
    double t = omp_get_wtime();

    chunk_offsets.clear();
    chunk_offsets.push_back(0);
    uint32_t total_ops = 0;
    char path[512];

    for (uint32_t file_idx = 0; ; file_idx++) {
        snprintf(path, sizeof(path), "data/%u/mem_addr_%04u.bin", block_number, file_idx);
        FILE* f = fopen(path, "rb");
        if (!f) break;

        fseek(f, 0, SEEK_END);
        size_t file_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint32_t n_entries = file_size / sizeof(uint32_t);

        if (total_ops + n_entries > MAX_OPS) { fclose(f); break; }

        size_t read = fread(ops + total_ops, sizeof(uint32_t), n_entries, f);
        if (read != n_entries) { fclose(f); break; }
        fclose(f);

        if (n_entries > 0) {
            total_ops += n_entries;
            chunk_offsets.push_back(total_ops);
        }
        if (chunk_offsets.size() - 1 >= MAX_CHUNKS) break;
    }

    num_ops = total_ops;
    if (num_ops == 0) { std::cerr << "  ERROR: no ops found" << std::endl; exit(1); }
    num_chunks = chunk_offsets.size() - 1;

    vals         = new uint32_t[num_ops];
    compact_keys = new uint32_t[num_ops];
    for (uint32_t i = 0; i < num_ops; i++)
        vals[i] = i;

    create_active_mask();

    std::cout << std::fixed << std::setprecision(2)
              << "  " << num_ops << " ops in " << num_chunks << " chunks"
              << " (" << (omp_get_wtime() - t) * 1e3 << " ms)" << std::endl;
}

void PairSortCPU::cpu_metadata() {
    std::cout << std::endl << "=== CPU Metadata ===" << std::endl;
    double t_total = omp_get_wtime(), t;

    int T = 1;
    #pragma omp parallel
    #pragma omp single
    T = omp_get_num_threads();

    // =================================================================
    // Pass 1: Compact addresses + coarse histogram
    // =================================================================
    t = omp_get_wtime();
    std::vector<uint32_t> coarse_hists(T * COARSE_BINS, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t* h = coarse_hists.data() + tid * COARSE_BINS;
        
        // Process file chunks assigned to this thread (round-robin)
        for (uint32_t c = tid; c < num_chunks; c += T) {
            uint32_t lo = chunk_offsets[c];
            uint32_t hi = chunk_offsets[c + 1];
            for (uint32_t i = lo; i < hi; i++) {
                uint32_t ck = compact_addr(ops[i]);
                compact_keys[i] = ck;
                h[ck >> COARSE_SHIFT]++;
            }
        }
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  Compact + coarse: " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // Merge thread histograms
    t = omp_get_wtime();
    std::vector<uint32_t> coarse_hist(COARSE_BINS, 0);
    for (int t = 0; t < T; t++)
        for (uint32_t b = 0; b < COARSE_BINS; b++)
            coarse_hist[b] += coarse_hists[t * COARSE_BINS + b];

    std::cout << std::fixed << std::setprecision(2)
              << "  Merge histograms: " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // Coarse prefix sum
    t = omp_get_wtime();
    std::vector<uint32_t> coarse_prefix(COARSE_BINS + 1);
    coarse_prefix[0] = 0;
    for (uint32_t b = 0; b < COARSE_BINS; b++)
        coarse_prefix[b + 1] = coarse_prefix[b] + coarse_hist[b];

    std::cout << std::fixed << std::setprecision(2)
              << "  Coarse prefix:    " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // =================================================================
    // Post-1: Region counts, instances, active instances, relevant bins
    // =================================================================
    t = omp_get_wtime();

    // Region boundaries in coarse bins
    uint32_t rom_bin_end   = N_ADDR_ROM >> COARSE_SHIFT;
    uint32_t input_bin_end = (N_ADDR_ROM + N_ADDR_INPUT) >> COARSE_SHIFT;

    region_n_ops[REGION_ROM] = coarse_prefix[rom_bin_end];
    region_n_ops[REGION_INPUT] = coarse_prefix[input_bin_end] - coarse_prefix[rom_bin_end];
    region_n_ops[REGION_RAM] = coarse_prefix[COARSE_BINS] - coarse_prefix[input_bin_end];

    region_ops_start[REGION_ROM]   = 0;
    region_ops_start[REGION_INPUT] = region_n_ops[REGION_ROM];
    region_ops_start[REGION_RAM]   = region_n_ops[REGION_ROM] + region_n_ops[REGION_INPUT];

    num_instances = 0;
    for (uint8_t r = 0; r < 3; r++) {
        num_inst[r] = (region_n_ops[r] + INSTANCE_SIZE - 1) / INSTANCE_SIZE;
        if (num_inst[r] > MAX_INST[r]) {
            std::cerr << "ERROR: too many instances in region " << REGION_NAME[r]
                      << " (" << num_inst[r] << " > " << MAX_INST[r] << ")" << std::endl;
            exit(1);
        }
        num_instances += num_inst[r];
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  Region counts:"
              << " ROM=" << region_n_ops[0] << " (" << num_inst[0] << " inst),"
              << " INPUT=" << region_n_ops[1] << " (" << num_inst[1] << " inst),"
              << " RAM=" << region_n_ops[2] << " (" << num_inst[2] << " inst),"
              << " total=" << num_instances << " inst" << std::endl;

    pick_active_instances();

    // Per-instance: approximate address range (in compact address space)
    struct InstInfo {
        uint32_t pos_start, pos_end;        // global sorted position range
        uint32_t approx_first, approx_last; // expanded compact address range
        uint32_t hist_base;                 // base address for local histogram
        uint32_t hist_size;                 // number of addresses in local histogram
    };
    std::vector<InstInfo> inst_info(num_active);

    uint32_t ai = 0;
    for (uint8_t r = 0; r < 3; r++) {
        uint32_t region_bin_start = REGION_ADDR_START[r] >> COARSE_SHIFT;
        uint32_t region_bin_end   = (REGION_ADDR_START[r] + REGION_N_ADDR[r]) >> COARSE_SHIFT;

        for (uint32_t j = 0; j < num_active_per[r]; j++, ai++) {
            uint32_t lid = active_local_ids[active_offset[r] + j];
            uint32_t pos_start = region_ops_start[r] + lid * INSTANCE_SIZE;
            uint32_t pos_end   = pos_start + std::min(INSTANCE_SIZE,
                                                       region_n_ops[r] - lid * INSTANCE_SIZE);
            uint32_t search_start = (lid == 0) ? pos_start : pos_start - 1;

            // Linear search coarse prefix for the bin containing search_start
            uint32_t b_start = region_bin_start;
            while (b_start + 1 < region_bin_end &&
                   coarse_prefix[b_start + 1] <= search_start)
                b_start++;

            // Linear search for the bin containing pos_end
            uint32_t b_end = b_start;
            while (b_end + 1 < region_bin_end &&
                   coarse_prefix[b_end + 1] < pos_end)
                b_end++;

            // Expand by ±1 bin
            if (b_start > region_bin_start) b_start--;
            if (b_end + 1 < region_bin_end) b_end++;

            uint32_t approx_first = b_start << COARSE_SHIFT;
            uint32_t approx_last  = std::min(((b_end + 1) << COARSE_SHIFT) - 1, N_ADDR - 1);

            inst_info[ai] = {pos_start, pos_end, approx_first, approx_last,
                             approx_first, approx_last - approx_first + 1};
        }
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  Analysis:         " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // =================================================================
    // Pass 2: Build per-instance address histograms
    // =================================================================
    t = omp_get_wtime();

    std::vector<std::vector<uint32_t>> inst_hist(num_active);
    for (uint32_t i = 0; i < num_active; i++)
        inst_hist[i].assign(inst_info[i].hist_size, 0);

    // Coarse-bin quick-reject for approximate ranges
    std::vector<bool> approx_bin(COARSE_BINS, false);
    for (uint32_t i = 0; i < num_active; i++) {
        uint32_t b_lo = inst_info[i].approx_first >> COARSE_SHIFT;
        uint32_t b_hi = inst_info[i].approx_last >> COARSE_SHIFT;
        for (uint32_t b = b_lo; b <= b_hi; b++)
            approx_bin[b] = true;
    }

    #pragma omp parallel
    {
        std::vector<std::vector<uint32_t>> local_hist(num_active);
        for (uint32_t i = 0; i < num_active; i++)
            local_hist[i].assign(inst_info[i].hist_size, 0);

        uint32_t chunk_sz = (num_ops + T - 1) / T;
        uint32_t tid = omp_get_thread_num();
        uint32_t lo = std::min((uint32_t)tid * chunk_sz, num_ops);
        uint32_t hi = std::min(lo + chunk_sz, num_ops);

        for (uint32_t fi = lo; fi < hi; fi++) {
            uint32_t ck = compact_keys[fi];
            if (!approx_bin[ck >> COARSE_SHIFT]) continue;
            for (uint32_t i = 0; i < num_active; i++) {
                if (ck >= inst_info[i].approx_first && ck <= inst_info[i].approx_last)
                    local_hist[i][ck - inst_info[i].hist_base]++;
            }
        }

        #pragma omp critical
        {
            for (uint32_t i = 0; i < num_active; i++)
                for (uint32_t a = 0; a < inst_info[i].hist_size; a++)
                    inst_hist[i][a] += local_hist[i][a];
        }
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  Histograms:       " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // =================================================================
    // Post-3: Find exact boundaries + compute addr_offsets
    // =================================================================
    t = omp_get_wtime();

    // Allocate metadata buffers
    result_nops = new uint32_t[(size_t)num_active * num_chunks]();
    // Compute total addresses needed for offsets buffer
    uint32_t total_offset_addrs = 0;
    std::vector<uint32_t> offset_starts(num_active);

    ai = 0;
    for (uint8_t r = 0; r < 3; r++) {
        for (uint32_t j = 0; j < num_active_per[r]; j++, ai++) {
            auto& info = inst_info[ai];
            auto& hist = inst_hist[ai];

            uint32_t lid = active_local_ids[active_offset[r] + j];
            uint32_t region_start = region_ops_start[r];
            uint32_t base_pos  = region_start + lid * INSTANCE_SIZE;
            uint32_t inst_size = std::min(INSTANCE_SIZE, region_n_ops[r] - lid * INSTANCE_SIZE);
            uint32_t halo_base = (lid == 0) ? base_pos : base_pos - 1;

            // Local prefix sum over this instance's portion of the union histogram
            std::vector<uint32_t> local_prefix(info.hist_size + 1);
            local_prefix[0] = 0;
            for (uint32_t a = 0; a < info.hist_size; a++)
                local_prefix[a + 1] = local_prefix[a] + hist[a];

            uint32_t global_prefix_at_base = coarse_prefix[info.approx_first >> COARSE_SHIFT];

            uint32_t inst_start = halo_base;
            uint32_t target_prefix_start = inst_start - global_prefix_at_base;

            uint32_t first_addr_local_idx = 0;
            {
                uint32_t lo = 0, hi = info.hist_size;
                while (lo < hi) {
                    uint32_t mid = lo + (hi - lo + 1) / 2;
                    if (local_prefix[mid] <= target_prefix_start) lo = mid;
                    else hi = mid - 1;
                }
                first_addr_local_idx = lo;
            }
            uint32_t first_addr_compact = info.approx_first + first_addr_local_idx;

            // Binary search for last_addr: largest addr where
            // global_prefix_at_base + local_prefix[addr] < inst_end
            uint32_t inst_end = base_pos + inst_size;
            uint32_t target_end = inst_end - global_prefix_at_base;

            uint32_t last_addr_local_idx = first_addr_local_idx;
            {
                uint32_t lo = first_addr_local_idx, hi = info.hist_size;
                while (lo < hi) {
                    uint32_t mid = lo + (hi - lo + 1) / 2;
                    if (local_prefix[mid] < target_end) lo = mid;
                    else hi = mid - 1;
                }
                last_addr_local_idx = lo;
            }

            // Trim trailing empty addresses
            {
                uint32_t max_pref = local_prefix[last_addr_local_idx + 1];
                uint32_t tlo = first_addr_local_idx, thi = last_addr_local_idx;
                while (tlo < thi) {
                    uint32_t mid = tlo + (thi - tlo + 1) / 2;
                    if (local_prefix[mid] < max_pref) tlo = mid;
                    else thi = mid - 1;
                }
                last_addr_local_idx = tlo;
            }

            uint32_t last_addr_compact = info.approx_first + last_addr_local_idx;
            uint32_t num_addrs = last_addr_local_idx - first_addr_local_idx + 1;

            metas[ai].inst_id    = lid;
            metas[ai].type       = r;
            metas[ai].first_addr = expand_addr(first_addr_compact);
            metas[ai].last_addr  = expand_addr(last_addr_compact);

            // Store info for addr_offsets computation
            offset_starts[ai] = total_offset_addrs;
            total_offset_addrs += num_addrs;

            // Store the local offset data we need for addr_offsets
            // We'll compute addr_offsets after allocating the buffer
            // For now, update inst_info with exact boundaries
            inst_info[ai].approx_first = first_addr_compact;  // now exact
            inst_info[ai].approx_last  = last_addr_compact;
            inst_info[ai].hist_base    = info.hist_base;  // keep original for prefix lookup
        }
    }

    // Allocate and compute addr_offsets
    offsets_buf = new uint32_t[total_offset_addrs];

    ai = 0;
    for (uint8_t r = 0; r < 3; r++) {
        for (uint32_t j = 0; j < num_active_per[r]; j++, ai++) {
            auto& info = inst_info[ai];
            uint32_t lid = active_local_ids[active_offset[r] + j];
            uint32_t region_start = region_ops_start[r];
            uint32_t base_pos  = region_start + lid * INSTANCE_SIZE;
            uint32_t halo_base = (lid == 0) ? base_pos : base_pos - 1;

            uint32_t first_addr_compact = info.approx_first;
            uint32_t last_addr_compact = info.approx_last;
            uint32_t num_addrs  = last_addr_compact - first_addr_compact + 1;

            uint32_t global_prefix_at_base = coarse_prefix[inst_info[ai].hist_base >> COARSE_SHIFT];
            uint32_t first_addr_local_idx = first_addr_compact - inst_info[ai].hist_base;

            uint32_t* out = offsets_buf + offset_starts[ai];

            // Same formula as GPU compute_addr_offsets_kernel
            out[0] = (halo_base == base_pos) ? 1 : 0;

            auto& hist_v = inst_hist[ai];
            // Recompute prefix from first_addr_local_idx
            uint32_t prefix_at_fa = 0;
            for (uint32_t a = 0; a < first_addr_local_idx; a++)
                prefix_at_fa += hist_v[a];
            uint32_t global_prefix_at_fa = global_prefix_at_base + prefix_at_fa;

            uint32_t cumul = 0;
            for (uint32_t a = 1; a < num_addrs; a++) {
                cumul += hist_v[first_addr_local_idx + a - 1];
                out[a] = global_prefix_at_fa + cumul - (base_pos - 1);
            }

            metas[ai].addr_offsets = {out, num_addrs};
        }
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  Boundaries:       " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // =================================================================
    // Pass 3: FML counts using exact boundaries + coarse-bin quick-reject
    // =================================================================
    t = omp_get_wtime();

    // Coarse-bin lookup for exact ranges (much tighter than approximate)
    std::vector<bool> exact_bin(COARSE_BINS, false);
    for (uint32_t i = 0; i < num_active; i++) {
        uint32_t b_lo = inst_info[i].approx_first >> COARSE_SHIFT;  // now exact
        uint32_t b_hi = inst_info[i].approx_last >> COARSE_SHIFT;
        for (uint32_t b = b_lo; b <= b_hi; b++)
            exact_bin[b] = true;
    }

    std::vector<uint32_t> fml(num_active * num_chunks * 3, 0);

    #pragma omp parallel for schedule(dynamic, 4)
    for (uint32_t c = 0; c < num_chunks; c++) {
        uint32_t c_start = chunk_offsets[c];
        uint32_t c_end   = chunk_offsets[c + 1];

        for (uint32_t j = c_start; j < c_end; j++) {
            uint32_t ck = compact_keys[j];
            if (!exact_bin[ck >> COARSE_SHIFT]) continue;
            for (uint32_t i = 0; i < num_active; i++) {
                uint32_t fa = inst_info[i].approx_first;
                uint32_t la = inst_info[i].approx_last;
                if (ck < fa || ck > la) continue;

                uint32_t addr_category;
                if (ck == fa)      addr_category = 0;
                else if (ck == la) addr_category = 2;
                else               addr_category = 1;

                fml[(i * num_chunks + c) * 3 + addr_category]++;
                break;
            }
        }
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  FML counts:       " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;

    // =================================================================
    // Post-4: Build metas (skip/include + nops_per_chunk)
    // =================================================================
    t = omp_get_wtime();

    for (uint32_t i = 0; i < num_active; i++) {
        uint32_t* nops = result_nops + (size_t)i * num_chunks;
        uint32_t first_addr_compact = inst_info[i].approx_first;
        uint32_t last_addr_compact = inst_info[i].approx_last;
        bool single_addr = (first_addr_compact == last_addr_compact);

        uint32_t lid = metas[i].inst_id;
        uint8_t  r   = metas[i].type;
        uint32_t region_start = region_ops_start[r];
        uint32_t base_pos  = region_start + lid * INSTANCE_SIZE;
        uint32_t inst_size = std::min(INSTANCE_SIZE, region_n_ops[r] - lid * INSTANCE_SIZE);
        uint32_t halo_base = (lid == 0) ? base_pos : base_pos - 1;

        uint32_t global_prefix_at_base = coarse_prefix[inst_info[i].hist_base >> COARSE_SHIFT];
        uint32_t first_addr_local_idx = first_addr_compact - inst_info[i].hist_base;
        auto& hist_v = inst_hist[i];
        uint32_t prefix_at_fa = 0;
        for (uint32_t a = 0; a < first_addr_local_idx; a++)
            prefix_at_fa += hist_v[a];
        uint32_t global_prefix_at_fa = global_prefix_at_base + prefix_at_fa;

        // first_addr_total_skip = halo_base - global_prefix_at_fa
        uint32_t first_addr_total_skip = halo_base - global_prefix_at_fa;

        // last_addr_total_include
        uint32_t last_addr_total_include;
        if (!single_addr) {
            // Count ops between first_addr and last_addr (exclusive)
            uint32_t last_addr_local_idx = last_addr_compact - inst_info[i].hist_base;
            uint32_t filled_before_last = 0;
            for (uint32_t a = first_addr_local_idx; a < last_addr_local_idx; a++)
                filled_before_last += hist_v[a];
            filled_before_last = (global_prefix_at_fa + filled_before_last) - base_pos;
            last_addr_total_include = inst_size - filled_before_last;
        } else {
            last_addr_total_include = inst_size;
            if (halo_base != base_pos) last_addr_total_include++;
        }

        // Find first_addr_chunk and first_addr_skip
        uint32_t first_addr_chunk = 0, first_addr_skip = 0;
        {
            uint32_t cum = 0;
            for (uint32_t c = 0; c < num_chunks; c++) {
                uint32_t first_count = fml[(i * num_chunks + c) * 3 + 0];
                if (cum + first_count > first_addr_total_skip) {
                    first_addr_chunk = c;
                    first_addr_skip  = first_addr_total_skip - cum;
                    break;
                }
                cum += first_count;
            }
        }

        // Find last_addr_chunk and last_addr_include
        uint32_t last_addr_chunk = 0, last_addr_include = 0;
        {
            uint32_t last_addr_category = single_addr ? 0 : 2;
            uint32_t la_threshold = single_addr
                ? (first_addr_total_skip + last_addr_total_include)
                : last_addr_total_include;

            uint32_t cum = 0;
            for (uint32_t c = 0; c < num_chunks; c++) {
                uint32_t last_count = fml[(i * num_chunks + c) * 3 + last_addr_category];
                if (cum + last_count >= la_threshold) {
                    last_addr_chunk   = c;
                    last_addr_include = la_threshold - cum;
                    break;
                }
                cum += last_count;
            }
        }

        metas[i].first_addr_chunk  = first_addr_chunk;
        metas[i].first_addr_skip   = first_addr_skip;
        metas[i].last_addr_chunk   = last_addr_chunk;
        metas[i].last_addr_include = last_addr_include;

        // Write nops_per_chunk with chunk elimination
        for (uint32_t c = 0; c < num_chunks; c++) {
            uint32_t first_count = fml[(i * num_chunks + c) * 3 + 0];
            uint32_t middle_count = fml[(i * num_chunks + c) * 3 + 1];
            uint32_t last_count = fml[(i * num_chunks + c) * 3 + 2];
            if (first_count + middle_count + last_count == 0) continue;

            bool needed = (middle_count > 0);
            if (first_count > 0) {
                if (single_addr) {
                    if (c >= first_addr_chunk && c <= last_addr_chunk) needed = true;
                } else {
                    if (c >= first_addr_chunk) needed = true;
                }
            }
            if (last_count > 0 && c <= last_addr_chunk)
                needed = true;
            if (needed)
                nops[c] = first_count + middle_count + last_count;
        }

        metas[i].nops_per_chunk = {nops + 0, num_chunks};
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  Build metas:      " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl
              << "  TOTAL:            " << (omp_get_wtime() - t_total) * 1e3 << " ms" << std::endl;
}

// Same as GPU version's cpu_fill
void PairSortCPU::cpu_fill() {
    std::cout << std::endl << "=== CPU Fill ===" << std::endl;
    double t = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t idx = 0; idx < num_active; idx++) {
        auto& m = metas[idx];
        bool single_addr = (m.first_addr == m.last_addr);

        uint32_t* o_ops;
        uint32_t* o_vals;
        if (m.type == REGION_ROM)        { o_ops = out_ops_rom;   o_vals = out_vals_rom;   }
        else if (m.type == REGION_INPUT) { o_ops = out_ops_input; o_vals = out_vals_input; }
        else                             { o_ops = out_ops_ram;   o_vals = out_vals_ram;   }

        uint32_t inst_size     = std::min(INSTANCE_SIZE, region_n_ops[m.type] - m.inst_id * INSTANCE_SIZE);
        uint32_t out_base      = m.inst_id * INSTANCE_SIZE;
        uint32_t total_written = 0;

        for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
            uint32_t expected = m.nops_per_chunk[chunk];
            if (expected == 0) continue;

            uint32_t chunk_start = chunk_offsets[chunk];
            uint32_t chunk_size  = chunk_offsets[chunk + 1] - chunk_start;
            uint32_t found       = 0;
            uint32_t first_found = 0;
            uint32_t last_found  = 0;

            for (uint32_t j = 0; j < chunk_size && found < expected; j++) {
                uint32_t raw  = ops[chunk_start + j];
                if (raw < m.first_addr || raw > m.last_addr) continue;
                uint32_t ind  = (raw - m.first_addr) >> 3;
                found++;

                bool skip = false;
                if (raw == m.first_addr) {
                    first_found++;
                    if (chunk < m.first_addr_chunk) skip = true;
                    else if (chunk == m.first_addr_chunk && first_found <= m.first_addr_skip) skip = true;
                    else if (single_addr) {
                        if (chunk > m.last_addr_chunk) skip = true;
                        else if (chunk == m.last_addr_chunk && first_found > m.last_addr_include) skip = true;
                    }
                } else if (raw == m.last_addr) {
                    last_found++;
                    if (chunk > m.last_addr_chunk) skip = true;
                    else if (chunk == m.last_addr_chunk && last_found > m.last_addr_include) skip = true;
                }
                if (skip) continue;

                uint32_t pos = m.addr_offsets[ind]++;
                if (pos == 0) continue;  // halo entry
                uint32_t out_pos = out_base + pos - 1;
                o_ops[out_pos]  = raw;
                o_vals[out_pos] = vals[chunk_start + j];
                total_written++;
                if (total_written >= inst_size) break;
            }
            if (total_written >= inst_size) break;
        }
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  " << num_active << " instances in "
              << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;
}

void PairSortCPU::reference_sort() {
    std::cout << std::endl << "=== Verify ===" << std::endl;
    double t = omp_get_wtime();

    struct Triple { uint32_t compact, pos, val, raw; };
    std::vector<Triple> triples(num_ops);
    for (uint32_t i = 0; i < num_ops; i++)
        triples[i] = {compact_addr(ops[i]), i, vals[i], ops[i]};
    std::sort(triples.begin(), triples.end(), [](const Triple& a, const Triple& b) {
        return a.compact < b.compact || (a.compact == b.compact && a.pos < b.pos);
    });

    memset(ref_ops_rom,    0, (size_t)MAX_INST_ROM   * INSTANCE_SIZE * sizeof(uint32_t));
    memset(ref_vals_rom,   0, (size_t)MAX_INST_ROM   * INSTANCE_SIZE * sizeof(uint32_t));
    memset(ref_ops_input,  0, (size_t)MAX_INST_INPUT * INSTANCE_SIZE * sizeof(uint32_t));
    memset(ref_vals_input, 0, (size_t)MAX_INST_INPUT * INSTANCE_SIZE * sizeof(uint32_t));
    memset(ref_ops_ram,    0, (size_t)MAX_INST_RAM   * INSTANCE_SIZE * sizeof(uint32_t));
    memset(ref_vals_ram,   0, (size_t)MAX_INST_RAM   * INSTANCE_SIZE * sizeof(uint32_t));

    for (uint32_t p = 0; p < num_ops; p++) {
        uint32_t ca = triples[p].compact;
        uint32_t* r_ops;
        uint32_t* r_vals;
        uint32_t local_p;

        if (ca < N_ADDR_ROM) {
            r_ops   = ref_ops_rom;   r_vals = ref_vals_rom;
            local_p = p - region_ops_start[REGION_ROM];
        } else if (ca < N_ADDR_ROM + N_ADDR_INPUT) {
            r_ops   = ref_ops_input; r_vals = ref_vals_input;
            local_p = p - region_ops_start[REGION_INPUT];
        } else {
            r_ops   = ref_ops_ram;   r_vals = ref_vals_ram;
            local_p = p - region_ops_start[REGION_RAM];
        }

        r_ops[local_p]  = triples[p].raw;
        r_vals[local_p] = triples[p].val;
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  Reference sort:   " << (omp_get_wtime() - t) * 1e3 << " ms" << std::endl;
}

void PairSortCPU::verify() {
    double t = omp_get_wtime();
    uint32_t total_verified = 0;

    for (uint32_t idx = 0; idx < num_active; idx++) {
        auto& m = metas[idx];

        uint32_t* o_ops;  uint32_t* o_vals;
        uint32_t* r_ops;  uint32_t* r_vals;
        if (m.type == REGION_ROM) {
            o_ops = out_ops_rom;   o_vals = out_vals_rom;
            r_ops = ref_ops_rom;   r_vals = ref_vals_rom;
        } else if (m.type == REGION_INPUT) {
            o_ops = out_ops_input; o_vals = out_vals_input;
            r_ops = ref_ops_input; r_vals = ref_vals_input;
        } else {
            o_ops = out_ops_ram;   o_vals = out_vals_ram;
            r_ops = ref_ops_ram;   r_vals = ref_vals_ram;
        }

        uint32_t inst_size = std::min(INSTANCE_SIZE,
                                      region_n_ops[m.type] - m.inst_id * INSTANCE_SIZE);
        uint32_t start = m.inst_id * INSTANCE_SIZE;

        for (uint32_t j = 0; j < inst_size; j++) {
            uint32_t ind = start + j;
            if (o_ops[ind] != r_ops[ind] || o_vals[ind] != r_vals[ind]) {
                std::cout << "MISMATCH " << REGION_NAME[m.type] << " inst " << m.inst_id
                          << " local " << j
                          << ": got (" << o_ops[ind] << "," << o_vals[ind] << ")"
                          << " expected (" << r_ops[ind] << "," << r_vals[ind] << ")" << std::endl;
                return;
            }
        }
        total_verified += inst_size;
    }

    std::cout << std::fixed << std::setprecision(2)
              << "  Verify:           " << (omp_get_wtime() - t) * 1e3
              << " ms -- " << total_verified << " entries, " << num_active << " instances OK"
              << std::endl;
}

// =====================================================================
// Main
// =====================================================================

int main(int argc, char** argv) {
    bool do_verify = false;
    uint32_t block_number = 0;
    bool have_block = false;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-v")
            do_verify = true;
        else {
            block_number = std::strtoul(argv[i], nullptr, 10);
            have_block = true;
        }
    }
    if (!have_block) {
        std::cerr << "Usage: " << argv[0] << " <block_number> [-v]" << std::endl;
        return 1;
    }

    PairSortCPU app;
    app.generate(block_number);
    app.cpu_metadata();
    app.cpu_fill();

    if (do_verify) {
        app.reference_sort();
        app.verify();
    }
}
