GPU_ARCH := sm_120

pair_sort_real_gpu: src/main_real.cu
	nvcc -O3 -arch=$(GPU_ARCH) -std=c++17 -Xcompiler -fopenmp -lgomp -o $@ $<

clean:
	rm -f pair_sort_real_gpu

.PHONY: clean
