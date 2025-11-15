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
ifeq ($(MAKECMDGOALS),test_mini)
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


ifeq ($(MAKECMDGOALS),test_minig)
	DATASET = -DMINI_DATASET
endif
ifeq ($(MAKECMDGOALS),test10g)
	DATASET = -DSMALL_DATASET
endif
ifeq ($(MAKECMDGOALS),test100g)
	DATASET = -DSTANDARD_DATASET
endif
ifeq ($(MAKECMDGOALS),test1000g)
	DATASET = -DLARGE_DATASET
endif
ifeq ($(MAKECMDGOALS),test10000g)
	DATASET = -DEXTRALARGE_DATASET
endif


# Default se non specificato
DATASET ?= -DSTANDARD_DATASET

# Combina tutte le flags
CFLAGS += $(DATASET) $(TIMING)

.PHONY: all clean run test_mini test10 test100 test1000 test10000

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

test10: $(EXE)
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

test100: $(EXE)
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

test1000: $(EXE)
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

test10000: $(EXE)
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

#-------------------GPU-----------------------------------------

test_minig: $(EXE)
	@echo ===================================================
	@echo "Esecuzione MINI_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo ....................................................

test10g: $(EXE)
	@echo ===================================================
	@echo "Esecuzione SMALL_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo ....................................................

test100g: $(EXE)
	@echo ===================================================
	@echo "Esecuzione STANDARD_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo ....................................................

test1000g: $(EXE)
	@echo ===================================================
	@echo "Esecuzione LARGE_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo ....................................................

test10000g: $(EXE)
	@echo ===================================================
	@echo "Esecuzione EXTRALARGE_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo "=== Test con limit2 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=4 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 2 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=16 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 4 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=64 perf stat -r 10 ./$(EXE)
	@echo "\n=== Test con 8 thread ==="
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=128 perf stat -r 10 ./$(EXE)
	@echo ....................................................



optimized: CFLAGS += -O3 -march=native -mtune=native -ffast-math
optimized: $(EXE)

debug: CFLAGS = -g -O0 -fopenmp -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXE)
