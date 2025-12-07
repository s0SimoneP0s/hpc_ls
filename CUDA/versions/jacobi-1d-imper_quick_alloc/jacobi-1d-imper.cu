#ifndef __CUDACC__
  #define __CUDACC__
#endif

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <polybench.h>
#include <elapsed.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "jacobi-1d-imper.h"

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        printf("Error code: %d\n", err); \
        exit(1); \
    } \
} while(0)


int NUM_THREADS = atoi(getenv("NUM_THREADS"));
int BLOCK_SIZE = atoi(getenv("BLOCK_SIZE"));


static void init_array(int n,
                       DATA_TYPE POLYBENCH_1D(A, N, n),
                       DATA_TYPE POLYBENCH_1D(B, N, n) )
{
  int i;
  for (i = 0; i < n; i++)
  {
    A[i] = ((DATA_TYPE)i + 2) / n;
    B[i] = ((DATA_TYPE)i + 3) / n;
  }
}

static void print_array(int n, DATA_TYPE POLYBENCH_1D(A, N, n))
{
  int i;
  for (i = 0; i < n; i++)
  {
    fprintf(stderr, DATA_PRINTF_MODIFIER, A[i]);
    if (i % 20 == 0)
      fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

__global__ void jacobi_1d_kernel(DATA_TYPE *A, DATA_TYPE *B, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i > 0 && i < n - 1)
  {
    DATA_TYPE tmp = A[i - 1] + A[i] + A[i + 1];
    B[i] = 0.33333 * tmp;
  }
}

__global__ void myCudaMemcpy(DATA_TYPE *A, DATA_TYPE *B, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = B[i];
}

void kernel_jacobi_1d_imper(int tsteps, int n,
                           DATA_TYPE *A,
                           DATA_TYPE *B)
{
  dim3 numThreads(BLOCK_SIZE);
  int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  

  for (int t = 0; t < tsteps; t++) {

    start_timer();
    jacobi_1d_kernel<<<numBlocks, numThreads>>>(A, B, n);
    stop_timer();
    print_elapsed_ms("SAXPY execution time");

    myCudaMemcpy<<<numBlocks, numThreads>>>(A, B, n);
    
  }
}

int main(int argc, char **argv)
{
  
  int n = (int)N;
  int tsteps = (int)TSTEPS;
  printf("n = %d\ntsteps = %d\n", n, tsteps);
  printf("Threads: %d\nBlock Size: %d\n",NUM_THREADS,BLOCK_SIZE);

  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
  
  DATA_TYPE *A_uvm = NULL;
  DATA_TYPE *B_uvm = NULL;
  
  cudaMallocManaged(&A_uvm, sizeof(POLYBENCH_ARRAY(A)));
  cudaMallocManaged(&B_uvm, sizeof(POLYBENCH_ARRAY(B)));
  
  cudaMemcpy(A_uvm, POLYBENCH_ARRAY(A), sizeof(POLYBENCH_ARRAY(A)), cudaMemcpyHostToDevice);
  cudaMemcpy(B_uvm, POLYBENCH_ARRAY(B), sizeof(POLYBENCH_ARRAY(B)), cudaMemcpyHostToDevice);
  
  start_timer();
  CUDA_CHECK(kernel_jacobi_1d_imper(tsteps, n, A_uvm, B_uvm));
  stop_timer();
  print_elapsed_ms("Kernel execution time");  

  cudaFree(A_uvm);
  cudaFree(B_uvm);

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  return 0;
}
