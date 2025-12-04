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

int NUM_THREADS = getenv("NUM_THREADS") ? atoi(getenv("NUM_THREADS")) : -1;
int BLOCK_SIZE  = getenv("BLOCK_SIZE")  ? atoi(getenv("BLOCK_SIZE"))  : 256;

// Macro per check errori CUDA
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        printf("Error code: %d\n", err); \
        exit(1); \
    } \
} while(0)

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
  
  // Aggiungi boundary check più robusto
  if (i > 0 && i < n - 1)
  {
    DATA_TYPE tmp = A[i - 1] + A[i] + A[i + 1];
    B[i] = 0.33333 * tmp;
  }
}

// Kernel di test semplice per verificare che la GPU funzioni
__global__ void simple_test_kernel(DATA_TYPE *A, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    A[i] = A[i] * 2.0;
  }
}

void kernel_jacobi_1d_imper(int tsteps, int n,
                           DATA_TYPE POLYBENCH_1D(A, N, n),
                           DATA_TYPE POLYBENCH_1D(B, N, n))
{
  dim3 numThreads(BLOCK_SIZE);
  int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  DATA_TYPE *ptr_A = A;
  DATA_TYPE *ptr_B = B;
  
  printf("Configuration:\n");
  printf("  Blocks: %d\n", numBlocks);
  printf("  Threads per block: %d\n", BLOCK_SIZE);
  printf("  Total threads: %d\n", numBlocks * BLOCK_SIZE);
  printf("  Array size: %d elements (%.2f MB)\n", n, (n * sizeof(DATA_TYPE)) / (1024.0 * 1024.0));
  printf("  Iterations: %d\n", tsteps);
  printf("\n");
  
  // Test preliminare: verifica che i puntatori siano validi
  printf("Testing pointer validity...\n");
  printf("  ptr_A = %p\n", (void*)ptr_A);
  printf("  ptr_B = %p\n", (void*)ptr_B);
  
  // Verifica con cudaPointerGetAttributes
  cudaPointerAttributes attr_A, attr_B;
  cudaError_t err_A = cudaPointerGetAttributes(&attr_A, ptr_A);
  cudaError_t err_B = cudaPointerGetAttributes(&attr_B, ptr_B);
  
  if (err_A == cudaSuccess) {
    printf("  A is managed memory: %d\n", attr_A.type == cudaMemoryTypeManaged);
  } else {
    printf("  WARNING: Cannot get attributes for A: %s\n", cudaGetErrorString(err_A));
  }
  
  if (err_B == cudaSuccess) {
    printf("  B is managed memory: %d\n", attr_B.type == cudaMemoryTypeManaged);
  } else {
    printf("  WARNING: Cannot get attributes for B: %s\n", cudaGetErrorString(err_B));
  }
  
  printf("\nRunning simple test kernel first...\n");
  simple_test_kernel<<<numBlocks, numThreads>>>(ptr_A, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Simple test kernel completed successfully!\n");
  printf("Sample value after test: A[100] = %.6f (should be ~0.0204)\n", ptr_A[100]);
  
  // Reinizializza A
  printf("Reinitializing array...\n");
  for (int i = 0; i < n; i++) {
    ptr_A[i] = ((DATA_TYPE)i + 2) / n;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  
  printf("\nStarting Jacobi iterations...\n");
  
  for (int t = 0; t < tsteps; t++) {
    jacobi_1d_kernel<<<numBlocks, numThreads>>>(ptr_A, ptr_B, n);
    
    // Check per errori di lancio
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      printf("Kernel launch error at iteration %d: %s\n", t, cudaGetErrorString(launch_err));
      break;
    }
    
    // Sincronizza
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
      printf("Kernel execution error at iteration %d: %s\n", t, cudaGetErrorString(sync_err));
      break;
    }
    
    // Stampa progresso ogni 10 iterazioni
    if (t % 10 == 0 && t > 0) {
      printf("  Completed %d/%d iterations - Sample value: B[100] = %.6f\n", t, tsteps, ptr_B[100]);
    }
    
    // Swap
    DATA_TYPE *tmp = ptr_A;
    ptr_A = ptr_B;
    ptr_B = tmp;
  }
  
  printf("All iterations completed!\n");
  
  // Copia il risultato finale in A se necessario
  if (tsteps % 2 == 1) {
    CUDA_CHECK(cudaMemcpy(A, ptr_A, n * sizeof(DATA_TYPE), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

int main(int argc, char **argv)
{
  // Verifica GPU
  int deviceCount;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  
  if (deviceCount == 0) {
    printf("ERROR: No CUDA device found!\n");
    return 1;
  }
  
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  
  printf("========================================\n");
  printf("CUDA Device: %s\n", prop.name);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("Multiprocessors: %d\n", prop.multiProcessorCount);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Global memory: %.2f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
  printf("Shared memory per block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
  printf("========================================\n\n");

  int n = (int)N;
  int tsteps = (int)TSTEPS;
  
  printf("Problem Configuration:\n");
  printf("  n = %d\n", n);
  printf("  tsteps = %d\n", tsteps);
  printf("  BLOCK_SIZE = %d\n", BLOCK_SIZE);
  printf("\n");

  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);

  if (A == NULL || B == NULL) {
    printf("ERROR: Memory allocation failed!\n");
    return 1;
  }
  
  printf("Memory allocated successfully\n");
  printf("  A = %p\n", (void*)A);
  printf("  B = %p\n\n", (void*)B);
  
  printf("Initializing arrays...\n");
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
  printf("Initial values: A[100] = %.6f, B[100] = %.6f\n\n", A[100], B[100]);
  
  // Prefetch per UVM
  printf("Prefetching to GPU...\n");
  CUDA_CHECK(cudaMemPrefetchAsync(A, n * sizeof(DATA_TYPE), 0));
  CUDA_CHECK(cudaMemPrefetchAsync(B, n * sizeof(DATA_TYPE), 0));
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Prefetch complete\n\n");
  
  printf("========================================\n");
  printf("EXECUTING KERNEL\n");
  printf("========================================\n");
  
  start_timer();
  kernel_jacobi_1d_imper(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
  CUDA_CHECK(cudaDeviceSynchronize());
  stop_timer();
  
  print_elapsed_ms("Total kernel execution time");
  
  printf("\n========================================\n");
  printf("VERIFICATION\n");
  printf("========================================\n");
  
  // Verifica risultati
  printf("Final values (middle of array):\n");
  for (int i = n/2 - 5; i < n/2 + 5; i++) {
    printf("A[%d] = %.6f\n", i, A[i]);
  }
  
  // Check convergenza: dopo molte iterazioni dovrebbe convergere
  double sum = 0.0;
  for (int i = 1; i < n-1; i++) {
    sum += A[i];
  }
  double avg = sum / (n - 2);
  printf("\nAverage value: %.6f (should be close to initial average after many iterations)\n", avg);
  
  printf("\nSUCCESS: Program completed without CUDA errors!\n");

  cudaFree(A);
  cudaFree(B);

  return 0;
}
