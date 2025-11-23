ifndef CUDA_HOME
CUDA_HOME := /usr/local/cuda
endif


CC=g++
UTIL_DIR = ../../utilities
BENCHMARK = jacobi-1d-imper


DATASET ?= -DSTANDARD_DATASET
$(foreach size,mini small standard large extralarge,\
  $(if $(filter test_$(size),$(MAKECMDGOALS)),$(eval DATASET=-D$(shell echo $(size) | tr a-z A-Z)_DATASET)))

## Ensure CUDA headers are available when compiling polybench.c with g++
CCFLAGS = -O2 -I. -I$(UTIL_DIR) -I$(CUDA_HOME)/include -DPOLYBENCH_TIME $(DATASET)
# Use explicit gencode for Tegra X1/Xavier style Maxwell GPUs (compute 5.3)
# and ensure nvcc uses the system g++ as host compiler (ccbin).
NVCCFLAGS = -Xcompiler -fopenmp -lineinfo --ptxas-options=-v --use_fast_math \
	-gencode arch=compute_53,code=sm_53 -ccbin $(CC) $(CCFLAGS)


$(BENCHMARK)_cuda: $(BENCHMARK).o polybench.o
	# Link the final host executable with nvcc (don't use -dlink here)
	$(CUDA_HOME)/bin/nvcc $(NVCCFLAGS) $^ -o $@ -lm -lcudart -lgomp

%.o: %.cu $(BENCHMARK).h $(UTIL_DIR)/polybench.h
	$(CUDA_HOME)/bin/nvcc $(NVCCFLAGS) -c $< -o $@

polybench.o: $(UTIL_DIR)/polybench.c $(UTIL_DIR)/polybench.h
	# Compile polybench.c with the C++ compiler so symbols match C++ linkage
	$(CUDA_HOME)/bin/nvcc $(NVCCFLAGS) -c $< -o $@

.PHONY: all clean run test_mini test_small test_standard test_large test_extralarge

all: $(BENCHMARK)_cuda

clean:
	rm -f $(BENCHMARK)_cuda *.o *~

run: $(BENCHMARK)_cuda
	@echo "Esecuzione del codice CUDA..."
	chmod +x $<
	perf stat -r 10 ./$<

test_mini test_small test_standard test_large test_extralarge: $(BENCHMARK)_cuda
	@echo "==================================================="
	@echo "Esecuzione $(patsubst test_%,%,$@)_DATASET"
	chmod +x $<
	perf stat -r 10 ./$<
	@echo "...................................................."

optimized: NVCCFLAGS += -O3 -ffast-math
optimized: $(BENCHMARK)_cuda

debug: NVCCFLAGS = -g -O0 -I. -I$(UTIL_DIR) $(DATASET) -DPOLYBENCH_TIME
debug: $(BENCHMARK)_cuda