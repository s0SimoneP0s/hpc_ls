# Makefile per Jacobi-1D con OpenMP
TIME_UTIL_DIR = ../../../utils
UTIL_DIR = ../../utilities
BUILD_DIR ?= ./build
MKDIR_P ?= mkdir -p


CC = gcc

BENCHMARK := jacobi-1d-imper.c
EXECUTABLE := $(BENCHMARK:.c=_omp)
OBJ_FILES := $(BUILD_DIR)/$(BENCHMARK:.c=.c.o) $(BUILD_DIR)/polybench.c.o $(BUILD_DIR)/elapsed.c.o

$(foreach size,mini_G small_G standard_G large_G extralarge_G,\
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




.PHONY: all clean run test_mini_G test_small_G test_standard_G test_large_G test_extralarge_G

all: $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE) *.o *~ $(BUILD_DIR)/*.o


THREADS ?= 4
run: $(EXECUTABLE)
	@echo "Esecuzione con $(THREADS) thread..."
	OMP_NUM_THREADS=$(THREADS) perf stat -r 5 ./$(EXECUTABLE)


test_mini_G test_small_G test_standard_G test_large_G test_extralarge_G: $(EXECUTABLE)
	@echo ===================================================
	@echo "Esecuzione MINI_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAMS_THREAD_LIMIT=16 perf stat -r 5 ./$(EXECUTABLE)
	OMP_NUM_TEAMS=2 OMP_TEAMS_THREAD_LIMIT=128 perf stat -r 5 ./$(EXECUTABLE)
	OMP_NUM_TEAMS=4 OMP_TEAMS_THREAD_LIMIT=16 perf stat -r 5 ./$(EXECUTABLE)
	OMP_NUM_TEAMS=4 OMP_TEAMS_THREAD_LIMIT=128 perf stat -r 5 ./$(EXECUTABLE)
	OMP_NUM_TEAMS=32 OMP_TEAMS_THREAD_LIMIT=16 perf stat -r 5 ./$(EXECUTABLE)
	OMP_NUM_TEAMS=32 OMP_TEAMS_THREAD_LIMIT=128 perf stat -r 5 ./$(EXECUTABLE)
	OMP_NUM_TEAMS=128 OMP_TEAMS_THREAD_LIMIT=16 perf stat -r 5 ./$(EXECUTABLE)
	OMP_NUM_TEAMS=128 OMP_TEAMS_THREAD_LIMIT=128 perf stat -r 5 ./$(EXECUTABLE)
	@echo ....................................................


# OMP_TARGET_OFFLOAD=MANDATORY

optimized: CCFLAGS += -O3 -march=native -mtune=native -ffast-math
optimized: $(EXECUTABLE)
debug: CCFLAGS = -g -O0 -fopenmp -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXECUTABLE)
