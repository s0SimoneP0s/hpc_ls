ifndef CUDA_HOME
CUDA_HOME := /usr/local/cuda
endif

UTIL_DIR = ../../utilities
BENCHMARK = jacobi-1d-imper

# Dataset selection
DATASET ?= -DSTANDARD_DATASET
$(foreach size,mini small standard large extralarge,\
  $(if $(filter test_$(size),$(MAKECMDGOALS)),$(eval DATASET=-D$(shell echo $(size) | tr a-z A-Z)_DATASET)))

# Compiler flags
CFLAGS = -O2 -I$(UTIL_DIR) -DPOLYBENCH_TIME $(DATASET)
NVCCFLAGS = -Xcompiler -fopenmp -lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math $(CFLAGS)

# Build
$(BENCHMARK)_cuda: $(BENCHMARK).o polybench.o
	$(CUDA_HOME)/bin/nvcc -dlink $^ -o $@ $(CFLAGS) -lm -lcudart -lgomp

%.o: %.cu $(BENCHMARK).h $(UTIL_DIR)/polybench.h
	$(CUDA_HOME)/bin/nvcc $(NVCCFLAGS) -c $< -o $@

polybench.o: $(UTIL_DIR)/polybench.c $(UTIL_DIR)/polybench.h
	gcc $(CFLAGS) -c $< -o $@

.PHONY: all clean run test_mini test_small test_standard test_large test_extralarge

all: $(BENCHMARK)_cuda

clean:
	rm -f $(BENCHMARK)_cuda *.o *~

run: $(BENCHMARK)_cuda
	@echo "Esecuzione del codice CUDA..."
	perf stat -r 10 ./$<

test_mini test_small test_standard test_large test_extralarge: $(BENCHMARK)_cuda
	@echo "==================================================="
	@echo "Esecuzione $(patsubst test_%,%,$@)_DATASET"
	perf stat -r 10 ./$<
	@echo "...................................................."

optimized: NVCCFLAGS += -O3 -ffast-math
optimized: $(BENCHMARK)_cuda

debug: NVCCFLAGS = -g -O0 -I. -I$(UTIL_DIR) $(DATASET) -DPOLYBENCH_TIME
debug: $(BENCHMARK)_cuda