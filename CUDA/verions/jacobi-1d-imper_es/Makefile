ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif

# nvcc -arch=sm_53 -O2 jacobi-1d.c -o jacobi-1d -lpolybench
UTIL_DIR = ../../utilities

CXX = g++
#CXXFLAGS = -O2 -march=native -I$(UTIL_DIR)
CXXFLAGS = -O2 -I$(UTIL_DIR)


NVCC=$(CUDA_HOME)/bin/nvcc
NVOPT:=-Xcompiler -fopenmp -lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math
NVCFLAGS = $(CXXFLAGS) $(NVOPT)
LDFLAGS = -lm -lcudart
NVLDFLAGS:=$(LDFLAGS) -lgomp

# choose dataset
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
# default
DATASET ?= -DSTANDARD_DATASET


TIMING = -DPOLYBENCH_TIME
NVCFLAGS += $(DATASET) $(TIMING)


BENCHMARK = jacobi-1d-imper
EXE = $(BENCHMARK)_cuda
SRC = $(BENCHMARK).cu $(UTIL_DIR)/polybench.c
HEADERS = $(BENCHMARK).h $(UTIL_DIR)/polybench.h
OBJS = $(SRC:.cu=.o) $(SRC:.c=.o)


$(EXE):	$(OBJS)
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

%.o: %.c $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@



.PHONY: all clean run test_mini test_small test_standard test_large test_extralarge

all: $(EXE)


clean:
	rm -f $(EXE) *.o *~

run: $(EXE)
	@echo "Esecuzione del codice CUDA..."
	perf stat -r 10 ./$(EXE)


test_mini: $(EXE)
	@echo ===================================================
	@echo "Esecuzione MINI_DATASET "
	perf stat -r 10 ./$(EXE) $(DATASET)
	@echo ....................................................

test_small: $(EXE)
	@echo ===================================================
	@echo "Esecuzione SMALL_DATASET "
	perf stat -r 10 ./$(EXE) $(DATASET)
	@echo ....................................................

test_standard: $(EXE)
	@echo ===================================================
	@echo "Esecuzione STANDARD_DATASET "
	perf stat -r 10 ./$(EXE) $(DATASET)
	@echo ....................................................

test_large: $(EXE)
	@echo ===================================================
	@echo "Esecuzione LARGE_DATASET "
	perf stat -r 10 ./$(EXE) $(DATASET)
	@echo ....................................................

test_extralarge: $(EXE)
	@echo ===================================================
	@echo "Esecuzione EXTRALARGE_DATASET "
	perf stat -r 10 ./$(EXE) $(DATASET)
	@echo ....................................................


optimized: NVCFLAGS += -O3 -ffast-math
optimized: $(EXE)

debug: NVCFLAGS = -g -O0 -I. -I$(UTIL_DIR) $(DATASET) $(TIMING)
debug: $(EXE)
