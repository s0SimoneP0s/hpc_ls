ifndef CUDA_HOME
CUDA_HOME:= /usr/local/cuda
endif

TIME_UTIL_DIR = ../../../utils
UTIL_DIR = ../../utilities
BUILD_DIR ?= ./build
MKDIR_P ?= mkdir -p

CC=g++
NVCC := $(CUDA_HOME)/bin/nvcc

BENCHMARK := jacobi-1d-imper.cu
EXECUTABLE := $(BENCHMARK:.cu=_cuda)
OBJ_FILES := $(BUILD_DIR)/$(BENCHMARK:.cu=.cu.o) $(BUILD_DIR)/polybench.c.o $(BUILD_DIR)/elapsed.c.o

# Determina il dataset in base al target
ifeq ($(MAKECMDGOALS),test_mini_CU_qa)
    DATASET := -DMINI_DATASET
else ifeq ($(MAKECMDGOALS),test_small_CU_qa)
    DATASET := -DSMALL_DATASET
else ifeq ($(MAKECMDGOALS),test_standard_CU_qa)
    DATASET := -DSTANDARD_DATASET
else ifeq ($(MAKECMDGOALS),test_large_CU_qa)
    DATASET := -DLARGE_DATASET
else ifeq ($(MAKECMDGOALS),test_extralarge_CU_qa)
    DATASET := -DEXTRALARGE_DATASET
else
    # Default se non specificato
    DATASET := -DSTANDARD_DATASET
endif

$(info Dataset flag: $(DATASET))

TIMING = -DPOLYBENCH_TIME

# C flags
CCFLAGS := -O2 -g -I. -I$(UTIL_DIR) -I$(TIME_UTIL_DIR) $(DATASET) $(TIMING) $(EXT_CCFLAGS)
NVCCFLAGS := -Xcompiler -fopenmp -lineinfo --ptxas-options=-v --use_fast_math \
        -arch=sm_53 $(CCFLAGS)

# linker flags
CLFLAGS := -lm -lcudart $(EXT_CLFLAGS)
NVCLFLAGS := $(CLFLAGS) -lgomp


$(EXECUTABLE): $(OBJ_FILES)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(OBJ_FILES) -o $@ $(NVCLFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/polybench.c.o: $(UTIL_DIR)/polybench.c
	$(MKDIR_P) $(dir $@)
	$(NVCC) -x cu $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/elapsed.c.o: $(TIME_UTIL_DIR)/elapsed.c
	$(MKDIR_P) $(dir $@)
	$(NVCC) -x cu $(NVCCFLAGS) -c $< -o $@

.PHONY: all clean run test_mini_CU_qa test_small_CU_qa test_standard_CU_qa test_large_CU_qa test_extralarge_CU_qa

all: $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE) *.o *~ $(BUILD_DIR)/*.o

run: $(EXECUTABLE)
	@echo "Esecuzione del codice CUDA..."
	chmod +x $
	perf stat -r 10 ./$

profile: $(EXECUTABLE)
	chmod +x $
	$(CUDA_HOME)/bin/nvprof ./$(EXECUTABLE)

# Target di test - rebuild per ogni test per applicare il dataset corretto
test_mini_CU_qa: clean
	@echo "==================================================="
	@echo "Building and running MINI dataset"
	@echo "==================================================="
	$(MAKE) $(EXECUTABLE) DATASET=-DMINI_DATASET
	chmod +x $(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=32 ---"
	NUM_THREADS=128 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=128 ---"
	NUM_THREADS=128 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=256 ---"
	NUM_THREADS=128 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=512 ---"
	NUM_THREADS=128 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=32 ---"
	NUM_THREADS=256 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=128 ---"
	NUM_THREADS=256 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=256 ---"
	NUM_THREADS=256 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=512 ---"
	NUM_THREADS=256 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=32 ---"
	NUM_THREADS=512 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=128 ---"
	NUM_THREADS=512 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=256 ---"
	NUM_THREADS=512 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=512 ---"
	NUM_THREADS=512 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "==================================================="

test_small_CU_qa: clean
	@echo "==================================================="
	@echo "Building and running SMALL dataset"
	@echo "==================================================="
	$(MAKE) $(EXECUTABLE) DATASET=-DSMALL_DATASET
	chmod +x $(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=32 ---"
	NUM_THREADS=128 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=128 ---"
	NUM_THREADS=128 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=256 ---"
	NUM_THREADS=128 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=512 ---"
	NUM_THREADS=128 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=32 ---"
	NUM_THREADS=256 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=128 ---"
	NUM_THREADS=256 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=256 ---"
	NUM_THREADS=256 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=512 ---"
	NUM_THREADS=256 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=32 ---"
	NUM_THREADS=512 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=128 ---"
	NUM_THREADS=512 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=256 ---"
	NUM_THREADS=512 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=512 ---"
	NUM_THREADS=512 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "==================================================="

test_standard_CU_qa: clean
	@echo "==================================================="
	@echo "Building and running STANDARD dataset"
	@echo "==================================================="
	$(MAKE) $(EXECUTABLE) DATASET=-DSTANDARD_DATASET
	chmod +x $(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=32 ---"
	NUM_THREADS=128 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=128 ---"
	NUM_THREADS=128 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=256 ---"
	NUM_THREADS=128 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=512 ---"
	NUM_THREADS=128 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=32 ---"
	NUM_THREADS=256 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=128 ---"
	NUM_THREADS=256 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=256 ---"
	NUM_THREADS=256 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=512 ---"
	NUM_THREADS=256 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=32 ---"
	NUM_THREADS=512 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=128 ---"
	NUM_THREADS=512 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=256 ---"
	NUM_THREADS=512 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=512 ---"
	NUM_THREADS=512 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "==================================================="

test_large_CU_qa: clean
	@echo "==================================================="
	@echo "Building and running LARGE dataset"
	@echo "==================================================="
	$(MAKE) $(EXECUTABLE) DATASET=-DLARGE_DATASET
	chmod +x $(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=32 ---"
	NUM_THREADS=128 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=128 ---"
	NUM_THREADS=128 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=256 ---"
	NUM_THREADS=128 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=512 ---"
	NUM_THREADS=128 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=32 ---"
	NUM_THREADS=256 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=128 ---"
	NUM_THREADS=256 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=256 ---"
	NUM_THREADS=256 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=512 ---"
	NUM_THREADS=256 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=32 ---"
	NUM_THREADS=512 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=128 ---"
	NUM_THREADS=512 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=256 ---"
	NUM_THREADS=512 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=512 ---"
	NUM_THREADS=512 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "==================================================="

test_extralarge_CU_qa: clean
	@echo "==================================================="
	@echo "Building and running EXTRALARGE dataset"
	@echo "==================================================="
	$(MAKE) $(EXECUTABLE) DATASET=-DEXTRALARGE_DATASET
	chmod +x $(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=32 ---"
	NUM_THREADS=128 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=128 ---"
	NUM_THREADS=128 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=256 ---"
	NUM_THREADS=128 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=128 BLOCK_SIZE=512 ---"
	NUM_THREADS=128 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=32 ---"
	NUM_THREADS=256 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=128 ---"
	NUM_THREADS=256 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=256 ---"
	NUM_THREADS=256 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=256 BLOCK_SIZE=512 ---"
	NUM_THREADS=256 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=32 ---"
	NUM_THREADS=512 BLOCK_SIZE=32  perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=128 ---"
	NUM_THREADS=512 BLOCK_SIZE=128 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=256 ---"
	NUM_THREADS=512 BLOCK_SIZE=256 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "--- NUM_THREADS=512 BLOCK_SIZE=512 ---"
	NUM_THREADS=512 BLOCK_SIZE=512 perf stat -r 3 ./$(EXECUTABLE)
	@echo ""
	@echo "==================================================="

optimized: NVCCFLAGS += -O3 -ffast-math
optimized: $(EXECUTABLE)

debug: NVCCFLAGS = -g -O0 -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXECUTABLE)
