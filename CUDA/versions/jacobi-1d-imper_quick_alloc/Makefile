ifndef CUDA_HOME
CUDA_HOME:= /usr/local/cuda
endif

LD_LIBRARY_PATH := /usr/local/cuda/lib64:$$LD_LIBRARY_PATH
TIME_UTIL_DIR = ../../../utils
UTIL_DIR = ../../utilities
BUILD_DIR ?= ./build
MKDIR_P ?= mkdir -p

CC=g++
NVCC := $(CUDA_HOME)/bin/nvcc

BENCHMARK := jacobi-1d-imper.cu
EXECUTABLE := $(BENCHMARK:.cu=_cuda)
OBJ_FILES := $(BUILD_DIR)/$(BENCHMARK:.cu=.cu.o) $(BUILD_DIR)/polybench.c.o $(BUILD_DIR)/elapsed.c.o


$(foreach size,mini_CU_qa small_CU_qa standard_CU_qa large_CU_qa extralarge_CU_qa,\
  $(if $(filter test_$(size)_CU_qa,$(MAKECMDGOALS)),$(eval DATASET=-D$(shell echo $(size) | tr a-z A-Z)_DATASET)))


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
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(CUDA_HOME)/bin/nvprof --print-gpu-trace -o trace ./$(EXECUTABLE)


STAT_TRYS=4
test_mini_CU_qa test_small_CU_qa test_standard_CU_qa test_large_CU_qa test_extralarge_CU_qa: $(EXECUTABLE)
	@echo "==================================================="
	@echo "Esecuzione $(patsubst %,%,$@)"
	chmod +x $<
	@for nt in 128 256 512 1024; do \
		for bs in 32 256 512; do \
			make clean; \
			echo "=== NUM_THREADS=$$nt BLOCK_SIZE=$$bs ==="; \
			$(MAKE) -B NVCCFLAGS="$(NVCCFLAGS) -DNUM_THREADS=$$nt -DBLOCK_SIZE=$$bs" $(EXECUTABLE); \
			perf stat -r $(STAT_TRYS) ./$(EXECUTABLE); \
		done; \
	done
	@echo "...................................................."


optimized: NVCCFLAGS += -O3 -ffast-math
optimized: $(EXECUTABLE)

debug: NVCCFLAGS = -g -O0 -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXECUTABLE)
