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


ifeq ($(MAKECMDGOALS),test_mini_G)
	DATASET = -DMINI_DATASET
endif
ifeq ($(MAKECMDGOALS),test_small_G)
	DATASET = -DSMALL_DATASET
endif
ifeq ($(MAKECMDGOALS),test_standard_G)
	DATASET = -DSTANDARD_DATASET
endif
ifeq ($(MAKECMDGOALS),test_large_G)
	DATASET = -DLARGE_DATASET
endif
ifeq ($(MAKECMDGOALS),test_extralarge_G)
	DATASET = -DEXTRALARGE_DATASET
endif

# Default se non specificato
DATASET ?= -DSTANDARD_DATASET
# Combina tutte le flags


CFLAGS += $(DATASET) $(TIMING)
.PHONY: all clean run test_mini_G test_small_G test_standard_G test_large_G test_extralarge_G



all: $(EXE)

$(EXE): $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(EXE) *.o *~



THREADS ?= 4
run: $(EXE)
	OMP_NUM_THREADS=$(THREADS) perf stat -r 5 ./$(EXE)


# Esegui test per dataset specifici
#-------------------GPU-----------------------------------------
test_mini_G: $(EXE)
	@echo ===================================================
	@echo "Esecuzione MINI_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	@echo ....................................................

test_small_G: $(EXE)
	@echo ===================================================
	@echo "Esecuzione SMALL_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	@echo ....................................................

test_standard_G: $(EXE)
	@echo ===================================================
	@echo "Esecuzione STANDARD_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	@echo ....................................................

test_large_G: $(EXE)
	@echo ===================================================
	@echo "Esecuzione LARGE_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	@echo ....................................................

test_extralarge_G: $(EXE)
	@echo ===================================================
	@echo "Esecuzione EXTRALARGE_DATASET "
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=2 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=4 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=32 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=4 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=16 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=64 perf stat -r 5 ./$(EXE)
	OMP_NUM_TEAMS=128 OMP_TEAM_LIMIT=128 perf stat -r 5 ./$(EXE)
	@echo ....................................................


optimized: CFLAGS += -O3 -march=native -mtune=native -ffast-math
optimized: $(EXE)
debug: CFLAGS = -g -O0 -fopenmp -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXE)
