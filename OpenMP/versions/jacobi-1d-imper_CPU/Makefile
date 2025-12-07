# Makefile per Jacobi-1D con OpenMP

TIME_UTIL_DIR = ../../../utils
UTIL_DIR = ../../utilities
BUILD_DIR ?= ./build
MKDIR_P ?= mkdir -p

CC = gcc


BENCHMARK := jacobi-1d-imper.c
EXECUTABLE := $(BENCHMARK:.c=_omp)
OBJ_FILES := $(BUILD_DIR)/$(BENCHMARK:.c=.c.o) $(BUILD_DIR)/polybench.c.o $(BUILD_DIR)/elapsed.c.o




$(foreach size,mini_C small_C standard_C large_C extralarge_C,\
  $(if $(filter test_$(size)_C,$(MAKECMDGOALS)),$(eval DATASET=-D$(shell echo $(size) | tr a-z A-Z)_DATASET)))

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


.PHONY: all clean run test_mini_C test_small_C test_standard_C test_large_C test_extralarge_C

all: $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE) *.o *~ $(BUILD_DIR)/*.o

THREADS ?= 4
run: $(EXECUTABLE)
	@echo "Esecuzione con $(THREADS) thread..."
	OMP_NUM_THREADS=$(THREADS) perf stat -r 10 ./$(EXECUTABLE)


STAT_TRYS=10
test_mini_C test_small_C test_standard_C test_large_C test_extralarge_C: $(EXECUTABLE)
	@echo "==================================================="
	@echo "Esecuzione $(patsubst %,%,$@)"
	chmod +x $<
	@for nt in 1 2 4 8; do \
		make clean; \
		echo "=== OMP_NUM_THREADS=$$nt  ==="; \
		echo "=== Test con $$nt thread ===" \
		$(MAKE) -B CCFLAGS="$(CCFLAGS) -DOMP_NUM_THREADS=$$nt" $(EXECUTABLE); \
		perf stat -r $(STAT_TRYS) ./$(EXECUTABLE); \
	done
	@echo "...................................................."



optimized: CCFLAGS += -O3 -march=native -mtune=native -ffast-math
optimized: $(EXECUTABLE)

debug: CCFLAGS = -g -O0 -fopenmp -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXECUTABLE)
