GPU_ARCH := sm_120

pair_sort_real_gpu: src/main_real.cu
	nvcc -O3 -arch=$(GPU_ARCH) -std=c++20 -Xcompiler -fopenmp -lgomp -o $@ $<

pair_sort_real_cpu: src/main_real_cpu.cpp
	g++ -O3 -std=c++20 -fopenmp -lgomp -o $@ $<

clean:
	rm -f pair_sort_real_gpu pair_sort_real_cpu

.PHONY: clean
