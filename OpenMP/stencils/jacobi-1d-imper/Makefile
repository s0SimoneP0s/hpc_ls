# Makefile per Jacobi-1D con OpenMP

UTIL_DIR = ../../utilities

# Configurazione compilatore
CC = gcc
# IMPORTANTE: -I utilities deve essere presente per includere polybench.h
# e polybench.c deve essere compilato insieme al codice
CFLAGS = -O3 -fopenmp -march=native -I$(UTIL_DIR)
LDFLAGS = -lm -fopenmp

# File sorgente
BENCHMARK = jacobi-1d-imper
EXE = $(BENCHMARK)_omp
SRC = $(BENCHMARK).c $(UTIL_DIR)/polybench.c
HEADERS = $(BENCHMARK).h $(UTIL_DIR)/polybench.h

# Opzioni di dataset (scegli una)
DATASET = -DMINI_DATASET      # TSTEPS=2, N=500
# DATASET = -DSMALL_DATASET     # TSTEPS=10, N=1000
# DATASET = -DSTANDARD_DATASET    # TSTEPS=100, N=10000 (default)
# DATASET = -DLARGE_DATASET     # TSTEPS=1000, N=100000
# DATASET = -DEXTRALARGE_DATASET # TSTEPS=1000, N=1000000

# Opzione per il timing
TIMING = -DPOLYBENCH_TIME

# Combina tutte le flags
CFLAGS += $(DATASET) $(TIMING)

.PHONY: all clean run test

all: $(EXE)

$(EXE): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(EXE) *.o *~

# Esegui con numero di thread specificato
# Esempio: make run THREADS=4
THREADS ?= 4
run: $(EXE)
	@echo "Esecuzione con $(THREADS) thread..."
	OMP_NUM_THREADS=$(THREADS) perf stat -r 10 ./$(EXE)

# Test con diversi numeri di thread
test: $(EXE)
	@echo "=== Test con 1 thread ==="
	OMP_NUM_THREADS=1 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_THREADS=2 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_THREADS=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_THREADS=8 perf stat -r 10 ./$(EXE)

# Versione ottimizzata per performance
optimized: CFLAGS += -O3 -march=native -mtune=native -ffast-math
optimized: $(EXE)

# Versione per debugging
debug: CFLAGS = -g -O0 -fopenmp -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXE)
