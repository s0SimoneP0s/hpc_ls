ifndef CUDA_HOME
CUDA_HOME := /usr/local/cuda
endif

TIME_UTIL_DIR = ../../elapsed_utils
UTIL_DIR = ../../utilities
BUILD_DIR ?= ./build
MKDIR_P ?= mkdir -p

CC=g++
NVCC := $(CUDA_HOME)/bin/nvcc

BENCHMARK := jacobi-1d-imper.cu
EXECUTABLE := $(BENCHMARK:.cu=_cuda)
OBJ_FILES := $(BUILD_DIR)/$(BENCHMARK:.cu=.cu.o) $(BUILD_DIR)/polybench.c.o $(BUILD_DIR)/elapsed.c.o

DATASET ?= -DSTANDARD_DATASET
$(foreach size,mini small standard large extralarge,\
  $(if $(filter test_$(size),$(MAKECMDGOALS)),$(eval DATASET=-D$(shell echo $(size) | tr a-z A-Z)_DATASET)))

# C flags
CCFLAGS := -O2 -g -I. -I$(UTIL_DIR) -I$(TIME_UTIL_DIR) $(EXT_CCFLAGS)
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
	$(NVCC) -x cu $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(NVCC) -x cu $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/polybench.c.o: $(UTIL_DIR)/polybench.c
	$(MKDIR_P) $(dir $@)
	$(NVCC) -x cu $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/elapsed.c.o: $(TIME_UTIL_DIR)/elapsed.c
	$(MKDIR_P) $(dir $@)
	$(NVCC) -x cu $(NVCCFLAGS) -c $< -o $@

.PHONY: all clean run test_mini_CU_es test_small_CU_es test_standard_CU_es test_large_CU_es test_extralarge_CU_es

all: $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE) *.o *~ $(BUILD_DIR)/*.o

run: $(EXECUTABLE)
	@echo "Esecuzione del codice CUDA..."
	chmod +x $<
	perf stat -r 10 ./$<

profile: $(EXECUTABLE)
	chmod +x $<
	$(CUDA_HOME)/bin/nvprof ./$(EXECUTABLE)

test_mini_CU_es test_small_CU_es test_standard_CU_es test_large_CU_es test_extralarge_CU_es: $(EXECUTABLE)
	@echo "==================================================="
	@echo "Esecuzione $(patsubst test_%,%,$@)_DATASET"
	chmod +x $<
	NUM_THREADS=128 BLOCK_SIZE=32  perf stat -r 5 ./$<
	NUM_THREADS=128 BLOCK_SIZE=128 perf stat -r 5 ./$<
	NUM_THREADS=128 BLOCK_SIZE=256 perf stat -r 5 ./$<
	NUM_THREADS=128 BLOCK_SIZE=512 perf stat -r 5 ./$<
	NUM_THREADS=256 BLOCK_SIZE=32  perf stat -r 5 ./$<
	NUM_THREADS=256 BLOCK_SIZE=128 perf stat -r 5 ./$<
	NUM_THREADS=256 BLOCK_SIZE=256 perf stat -r 5 ./$<
	NUM_THREADS=256 BLOCK_SIZE=512 perf stat -r 5 ./$<
	NUM_THREADS=512 BLOCK_SIZE=32  perf stat -r 5 ./$<
	NUM_THREADS=512 BLOCK_SIZE=128 perf stat -r 5 ./$<
	NUM_THREADS=512 BLOCK_SIZE=256 perf stat -r 5 ./$<
	NUM_THREADS=512 BLOCK_SIZE=512 perf stat -r 5 ./$<
	@echo "...................................................."

optimized: NVCCFLAGS += -O3 -ffast-math
optimized: $(EXECUTABLE)

debug: NVCCFLAGS = -g -O0 -I. -I$(UTIL_DIR) $(DATASET) -DPOLYBENCH_TIME
debug: $(EXECUTABLE)