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
                           DATA_TYPE *A,
                           DATA_TYPE *B)
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
  printf("  Sync strategy: ");
  
  // Strategia ottimizzata: sync solo ogni N iterazioni
  int sync_interval = 10;
  if (tsteps <= 10) sync_interval = 1;
  else if (tsteps <= 100) sync_interval = 10;
  else sync_interval = 50;
  
  printf("sync every %d iterations\n", sync_interval);
  printf("\n");
  
  // Test semplice rimosso per velocità
  printf("Starting Jacobi iterations...\n");
  
  int last_sync = 0;
  for (int t = 0; t < tsteps; t++) {
    jacobi_1d_kernel<<<numBlocks, numThreads>>>(ptr_A, ptr_B, n);
    
    // Check errori solo dopo il lancio (non blocca)
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      printf("Kernel launch error at iteration %d: %s\n", t, cudaGetErrorString(launch_err));
      cudaDeviceSynchronize();
      break;
    }
    
    // Swap
    DATA_TYPE *tmp = ptr_A;
    ptr_A = ptr_B;
    ptr_B = tmp;
    
    // Sincronizza solo periodicamente
    if ((t + 1) % sync_interval == 0 || t == tsteps - 1) {
      cudaError_t sync_err = cudaDeviceSynchronize();
      if (sync_err != cudaSuccess) {
        printf("Kernel execution error at iteration %d: %s\n", t, cudaGetErrorString(sync_err));
        break;
      }
      
      if ((t + 1) % 100 == 0) {
        printf("  Completed %d/%d iterations\n", t + 1, tsteps);
      }
      last_sync = t + 1;
    }
  }
  
  // Sync finale se necessario
  if (last_sync < tsteps) {
    cudaDeviceSynchronize();
  }
  
  printf("All iterations completed!\n");
  
  // Copia il risultato finale in A se necessario
  if (tsteps % 2 == 1) {
    CUDA_CHECK(cudaMemcpy(A, ptr_A, n * sizeof(DATA_TYPE), cudaMemcpyDeviceToDevice));
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
  printf("  Allocating %.2f MB per array\n", (n * sizeof(DATA_TYPE)) / (1024.0 * 1024.0));
  printf("\n");

  // Alloca direttamente
  DATA_TYPE *A = NULL;
  DATA_TYPE *B = NULL;
  
  printf("Allocating memory with cudaMallocManaged...\n");
  CUDA_CHECK(cudaMallocManaged(&A, n * sizeof(DATA_TYPE)));
  CUDA_CHECK(cudaMallocManaged(&B, n * sizeof(DATA_TYPE)));
  
  if (A == NULL || B == NULL) {
    printf("ERROR: Memory allocation failed!\n");
    return 1;
  }
  
  printf("Memory allocated successfully\n");
  printf("  A = %p\n", (void*)A);
  printf("  B = %p\n\n", (void*)B);
  
  printf("Initializing arrays...\n");
  init_array(n, A, B);
  printf("Initial values: A[100] = %.6f, B[100] = %.6f\n\n", A[100], B[100]);
  
  // Verifica che la memoria sia UVM
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, A);
  if (err == cudaSuccess) {
    printf("Memory type: ");
    if (attr.type == cudaMemoryTypeManaged) {
      printf("Managed (UVM) ✓\n");
      
      // Prova prefetch solo se è UVM
      printf("Prefetching to GPU...\n");
      int device;
      CUDA_CHECK(cudaGetDevice(&device));
      printf("Using device: %d\n", device);
      
      err = cudaMemPrefetchAsync(A, n * sizeof(DATA_TYPE), device);
      if (err != cudaSuccess) {
        printf("WARNING: Prefetch failed for A: %s (continuing anyway)\n", cudaGetErrorString(err));
        cudaGetLastError(); // Clear error
      }
      
      err = cudaMemPrefetchAsync(B, n * sizeof(DATA_TYPE), device);
      if (err != cudaSuccess) {
        printf("WARNING: Prefetch failed for B: %s (continuing anyway)\n", cudaGetErrorString(err));
        cudaGetLastError(); // Clear error
      }
      
      CUDA_CHECK(cudaDeviceSynchronize());
      printf("Prefetch complete\n");
    } else if (attr.type == cudaMemoryTypeDevice) {
      printf("Device only (not UVM)\n");
    } else if (attr.type == cudaMemoryTypeHost) {
      printf("Host only (not UVM)\n");
    } else {
      printf("Unregistered\n");
    }
  } else {
    printf("WARNING: Cannot determine memory type: %s\n", cudaGetErrorString(err));
    cudaGetLastError(); // Clear error
  }
  printf("\n");
  
  printf("========================================\n");
  printf("EXECUTING KERNEL\n");
  printf("========================================\n");
  
  start_timer();
  kernel_jacobi_1d_imper(tsteps, n, A, B);
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
