# Makefile per Jacobi-1D con OpenMP

UTIL_DIR = ../../../utilities

# Configurazione compilatore
CC = gcc
CFLAGS = -O3 -fopenmp -march=native -I$(UTIL_DIR)
LDFLAGS = -lm -fopenmp

# File sorgente
BENCHMARK = jacobi-1d-imper
EXE = $(BENCHMARK)_omp
SRC = $(BENCHMARK).c $(UTIL_DIR)/polybench.c
HEADERS = ../$(BENCHMARK).h $(UTIL_DIR)/polybench.h

# Opzione per il timing
TIMING = -DPOLYBENCH_TIME

# Imposta il dataset in base al target
ifeq ($(MAKECMDGOALS),test_mini)
	DATASET = -DMINI_DATASET
endif
ifeq ($(MAKECMDGOALS),test_small)
	DATASET = -DSMALL_DATASET
endif
ifeq ($(MAKECMDGOALS),test_standard)
	DATASET = -DSTANDARD_DATASET
endif
ifeq ($(MAKECMDGOALS),test_large)
	DATASET = -DLARGE_DATASET
endif
ifeq ($(MAKECMDGOALS),test_extralarge)
	DATASET = -DEXTRALARGE_DATASET
endif

# Default se non specificato
DATASET ?= -DSTANDARD_DATASET

# Combina tutte le flags
CFLAGS += $(DATASET) $(TIMING)

.PHONY: all clean run test_mini test_small test_standard test_large test_extralarge

all: $(EXE)

$(EXE): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(EXE) *.o *~

THREADS ?= 4
run: $(EXE)
	@echo "Esecuzione con $(THREADS) thread..."
	OMP_NUM_THREADS=$(THREADS) perf stat -r 10 ./$(EXE)

# Esegui test per dataset specifici
test_mini: $(EXE)
	@echo ===================================================
	@echo "Esecuzione MINI_DATASET "
	@echo "=== Test con 1 thread ==="
	OMP_NUM_THREADS=1 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_THREADS=2 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_THREADS=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_THREADS=8 perf stat -r 10 ./$(EXE)
	@echo ....................................................

test_small: $(EXE)
	@echo ===================================================
	@echo "Esecuzione SMALL_DATASET "
	@echo "=== Test con 1 thread ==="
	OMP_NUM_THREADS=1 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_THREADS=2 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_THREADS=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_THREADS=8 perf stat -r 10 ./$(EXE)
	@echo ....................................................

test_standard: $(EXE)
	@echo ===================================================
	@echo "Esecuzione STANDARD_DATASET "
	@echo "=== Test con 1 thread ==="
	OMP_NUM_THREADS=1 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_THREADS=2 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_THREADS=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_THREADS=8 perf stat -r 10 ./$(EXE)
	@echo ....................................................

test_large: $(EXE)
	@echo ===================================================
	@echo "Esecuzione LARGE_DATASET "
	@echo "=== Test con 1 thread ==="
	OMP_NUM_THREADS=1 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_THREADS=2 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_THREADS=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_THREADS=8 perf stat -r 10 ./$(EXE)
	@echo ....................................................

test_extralarge: $(EXE)
	@echo ===================================================
	@echo "Esecuzione EXTRALARGE_DATASET "
	@echo "=== Test con 1 thread ==="
	OMP_NUM_THREADS=1 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_THREADS=2 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_THREADS=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_THREADS=8 perf stat -r 10 ./$(EXE)
	@echo ....................................................

optimized: CFLAGS += -O3 -march=native -mtune=native -ffast-math
optimized: $(EXE)

debug: CFLAGS = -g -O0 -fopenmp -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXE)
