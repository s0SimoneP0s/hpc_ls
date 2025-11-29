#ifndef __CUDACC__
  #define __CUDACC__
#endif

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <polybench.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "jacobi-1d-imper.h"


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

/*CPU linear print*/
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



__global__ void jacobi_1d_kernel(DATA_TYPE *A, DATA_TYPE *B, int n )
{

  int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i > 0 && i < n - 1) 
    {
        B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
    }
  
  __syncthreads();
  
}

__global__ void myCudaMemcpy(DATA_TYPE *A, DATA_TYPE *B, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = B[i];
}

void kernel_jacobi_1d_imper(int tsteps, int n, 
                           DATA_TYPE POLYBENCH_1D(A, N, n),
                           DATA_TYPE POLYBENCH_1D(B, N, n)
                          )
{
  dim3 numThreads(NUM_THREADS);
  int numBlocks = (n + NUM_THREADS - 1) / NUM_THREADS;
  
  // run
  for (int t = 0; t < tsteps; t++) {
    jacobi_1d_kernel<<<numBlocks, numThreads>>>(A, B, n); // UVM only
    cudaDeviceSynchronize();
    cudaMemcpy(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), n * sizeof(DATA_TYPE), cudaMemcpyDeviceToDevice);
    //myCudaMemcpy<<<numBlocks, numThreads>>>(A, B, n); 
    cudaDeviceSynchronize();
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

  //DATA_TYPE *d_A = cudaMalloc((void**)&d_A, n * sizeof(DATA_TYPE));
  //DATA_TYPE *d_B = cudaMalloc((void**)&d_A, n * sizeof(DATA_TYPE));


  //cudaMemcpy(d_A, POLYBENCH_ARRAY(A), n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_B, POLYBENCH_ARRAY(B), n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);


  /* Run kernel. */
  kernel_jacobi_1d_imper(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Be clean.  UVM for cuda*/
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
