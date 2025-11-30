# Makefile per Jacobi-1D con OpenMP

TIME_UTIL_DIR = ../../../utils
UTIL_DIR = ../../utilities
BUILD_DIR ?= ./build
MKDIR_P ?= mkdir -p

CC = gcc



BENCHMARK := jacobi-1d-imper.c
EXECUTABLE := $(BENCHMARK:.c=_omp)
OBJ_FILES := $(BUILD_DIR)/$(BENCHMARK:.c=.c.o) $(BUILD_DIR)/polybench.c.o $(BUILD_DIR)/elapsed.c.o


$(foreach size,mini small standard large extralarge,\
  $(if $(filter test_$(size),$(MAKECMDGOALS)),$(eval DATASET=-D$(shell echo $(size) | tr a-z A-Z)_DATASET)))

$(info Dataset : $(DATASET))

TIMING = -DPOLYBENCH_TIME

# C flags
CCFLAGS = -O2 -fopenmp -march=native -I$(UTIL_DIR) -I$(TIME_UTIL_DIR) $(DATASET) 

# linker flags
CLFLAGS = -lm -fopenmp 


$(EXECUTABLE): $(OBJ_FILES)
	$(MKDIR_P) $(dir $@)
	$(CC) $(CCFLAGS) $(OBJ_FILES) -o $@ $(CLFLAGS)


$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CC) $(CCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CCFLAGS) -c $< -o $@

$(BUILD_DIR)/polybench.c.o: $(UTIL_DIR)/polybench.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CCFLAGS) -c $< -o $@

$(BUILD_DIR)/elapsed.c.o: $(TIME_UTIL_DIR)/elapsed.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CCFLAGS) -c $< -o $@


.PHONY: all clean run test_mini test_small test_standard test_large test_extralarge

all: $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE) *.o *~ $(BUILD_DIR)/*.o

THREADS ?= 1
run: $(EXECUTABLE)
	@echo "Esecuzione con $(THREADS) thread..."
	OMP_NUM_THREADS=$(THREADS) perf stat -r 10 ./$(EXECUTABLE)


test_mini test_small test_standard test_large test_extralarge: $(EXECUTABLE)
	@echo ===================================================
	@echo "Esecuzione $(patsubst %,%,$@)"
	@echo "=== Test con 1 thread ==="
	OMP_NUM_THREADS=1 perf stat -r 10 ./$<
	@echo ....................................................


optimized: CCFLAGS += -O3 -march=native -mtune=native -ffast-math
optimized: $(EXECUTABLE)

debug: CCFLAGS = -g -O0 -fopenmp -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXECUTABLE)
