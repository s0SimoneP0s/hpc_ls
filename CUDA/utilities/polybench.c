
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sched.h>
#include <math.h>
#if 0

#endif
#ifndef __CUDACC__
  #define __CUDACC__
  #include <cuda.h>
  #include <cuda_runtime.h>
#endif

void* polybench_alloc_data(unsigned long long int n, int elt_size)
{
  size_t val = n;
  val *= elt_size;
  void* ret = NULL;
  //cudaMalloc((void**)&ret, val);
  //cudaMallocManaged(&ret, val, cudaMemAttachGlobal);
  cudaMallocManaged((void **)&ret, val);
  return ret;
}
