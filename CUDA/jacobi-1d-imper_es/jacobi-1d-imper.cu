#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>


/* Cuda libraries*/
#include <cuda.h>
#include <cuda_runtime.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 100x10000. */
#include "jacobi-1d-imper.h"


/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE POLYBENCH_1D(A, N, n),
                       DATA_TYPE POLYBENCH_1D(B, N, n))
{
  int i;

  for (i = 0; i < n; i++)
  {
    A[i] = ((DATA_TYPE)i + 2) / n;
    B[i] = ((DATA_TYPE)i + 3) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_1D(A, N, n))

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


#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256 
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH (BLOCK_SIZE + 2)
#endif

__global__ void jacobi_1d_kernel(DATA_TYPE *A, DATA_TYPE *B, int n)
{
  __shared__ DATA_TYPE A_sh[TILE_WIDTH];
  
  int tx = threadIdx.x;
  int global_i = blockIdx.x * blockDim.x + tx;
  
  // target element
  if (global_i < n) {
    A_sh[tx + 1] = A[global_i];
  }
  
  // -1 halo
  if (tx == 0 && blockIdx.x > 0) {
    A_sh[0] = A[global_i - 1];
  }
  
  // +1 halo
  if (tx == blockDim.x - 1 && global_i < n - 1) {
    A_sh[tx + 2] = A[global_i + 1];
  }
  
  __syncthreads();
  
  // cumpute
  if (global_i > 0 && global_i < n - 1 && tx < blockDim.x) {
    DATA_TYPE neighbor_sum = A_sh[tx] + A_sh[tx + 1] + A_sh[tx + 2];
    B[global_i] = 0.33333 * neighbor_sum;
  }
}

#ifndef NUM_THREADS
#define NUM_THREADS 256 
#endif

void kernel_jacobi_1d_imper(int tsteps, int n, 
                           DATA_TYPE POLYBENCH_1D(A, N, n),
                           DATA_TYPE POLYBENCH_1D(B, N, n))
{
  dim3 numThreads(NUM_THREADS);
  int numBlocks = (n + NUM_THREADS - 1) / NUM_THREADS;
  
  // run
  for (int t = 0; t < tsteps; t++) {
    jacobi_1d_kernel<<<numBlocks, numThreads>>>(A, B, n); // UVM only
    cudaDeviceSynchronize();
    // swap
    DATA_TYPE *tmp = A;
    A = B;
    B = tmp;
  }
  cudaDeviceSynchronize();
}


int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;
  printf("n = %d\ntsteps = %d\n", n, tsteps);

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);

  /* Initialize array(s) UVM for cuda. */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));


  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_1d_imper(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean.  UVM for cuda*/
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
