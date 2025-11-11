# Makefile per Jacobi-1D con OpenMP

UTIL_DIR = ../../utilities

# Configurazione compilatore
CC = gcc
CFLAGS = -O3 -fopenmp -march=native -I$(UTIL_DIR)
LDFLAGS = -lm -fopenmp

# File sorgente
BENCHMARK = jacobi-1d-imper
EXE = $(BENCHMARK)_omp
SRC = $(BENCHMARK).c $(UTIL_DIR)/polybench.c
HEADERS = $(BENCHMARK).h $(UTIL_DIR)/polybench.h

# Opzione per il timing
TIMING = -DPOLYBENCH_TIME

# Imposta il dataset in base al target
ifeq ($(MAKECMDGOALS),test2)
	DATASET = -DMINI_DATASET
endif
ifeq ($(MAKECMDGOALS),test10)
	DATASET = -DSMALL_DATASET
endif
ifeq ($(MAKECMDGOALS),test100)
	DATASET = -DSTANDARD_DATASET
endif
ifeq ($(MAKECMDGOALS),test1000)
	DATASET = -DLARGE_DATASET
endif
ifeq ($(MAKECMDGOALS),test10000)
	DATASET = -DEXTRALARGE_DATASET
endif

# Default se non specificato
DATASET ?= -DSTANDARD_DATASET

# Combina tutte le flags
CFLAGS += $(DATASET) $(TIMING)

.PHONY: all clean run test2 test10 test100 test1000 test10000

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
test2: $(EXE)
	@echo "Esecuzione MINI_DATASET con $(THREADS) thread..."
	OMP_NUM_THREADS=$(THREADS) perf stat -r 10 ./$(EXE)

test10: $(EXE)
	@echo "Esecuzione SMALL_DATASET con $(THREADS) thread..."
	OMP_NUM_THREADS=$(THREADS) perf stat -r 10 ./$(EXE)

test100: $(EXE)
	@echo "Esecuzione STANDARD_DATASET con $(THREADS) thread..."
	OMP_NUM_THREADS=$(THREADS) perf stat -r 10 ./$(EXE)

test1000: $(EXE)
	@echo "Esecuzione LARGE_DATASET con $(THREADS) thread..."
	OMP_NUM_THREADS=$(THREADS) perf stat -r 10 ./$(EXE)

test10000: $(EXE)
	@echo "Esecuzione EXTRALARGE_DATASET con $(THREADS) thread..."
	OMP_NUM_THREADS=$(THREADS) perf stat -r 10 ./$(EXE)

optimized: CFLAGS += -O3 -march=native -mtune=native -ffast-math
optimized: $(EXE)

debug: CFLAGS = -g -O0 -fopenmp -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXE)
