#include <stdio.h>
#include <cuda_runtime.h>

// macro per error checking
#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("cuda error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    }} while(0)

// kernel gpu
__global__ void stress_kernel(float *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        for (int k = 0; k < 2000; ++k)
            v = v * 1.00001f + 0.00001f;
        a[i] = v;
    }
}

int main()
{
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        printf("error: no gpu cuda found\n");
        return 1;
    }

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));

    printf("============================================\n");
    printf("gpu found:\n");
    printf("  name: %s\n", p.name);
    printf("  compute capability: %d.%d\n", p.major, p.minor);
    printf("  multiprocessor: %d\n", p.multiProcessorCount);
    printf("  globale memory: %.2f MB\n", p.totalGlobalMem / (1024.0 * 1024.0));
    printf("============================================\n\n");

    // test size
    int n = 50 * 1024 * 1024;   // 50M ele
    size_t bytes = n * sizeof(float);

    printf("allocate %.2f MB...\n", bytes / (1024.0 * 1024.0));

    float *a;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));

    // inizializzazione
    for (int i = 0; i < 1000; i++) a[i] = 1.0f;

    CUDA_CHECK(cudaDeviceSynchronize());

    int block = 256;
    int grid = (n + block - 1) / block;

    printf("grid = %d  block = %d\n", grid, block);

    // tile elaps
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    stress_kernel<<<grid, block>>>(a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("\nkernel time elapsed: %.2f ms\n", ms);
    printf("verifica valori...\n");

    if (isnan(a[10]) || isinf(a[10])) {
        printf("error: not valid error, gpu problem should exist\n");
        return 1;
    }

    printf("test works correctly, cuda gpu correctly intsalled\n");

    cudaFree(a);
    return 0;
}
