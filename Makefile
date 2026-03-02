GPU_ARCH := sm_120

pair_sort_gpu: src/main.cu
	nvcc -O3 -arch=$(GPU_ARCH) -std=c++17 -Xcompiler -fopenmp -lgomp -o $@ $<

clean:
	rm -f pair_sort_gpu

.PHONY: clean
