/**
 * polybench.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
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
/* OpenMP support removed for this CUDA-only example. */
#endif
#ifndef __CUDACC__
  #define __CUDACC__
  #include <cuda.h>
  #include <cuda_runtime.h>
#endif

/* By default, collect PAPI counters on thread 0. */
#ifndef POLYBENCH_THREAD_MONITOR
# define POLYBENCH_THREAD_MONITOR 0
#endif

/* Total LLC cache size. By default 32+MB.. */
#ifndef POLYBENCH_CACHE_SIZE_KB
# define POLYBENCH_CACHE_SIZE_KB 32770
#endif


int polybench_papi_counters_threadid = POLYBENCH_THREAD_MONITOR;
double polybench_program_total_flops = 0;

/* PAPI support removed for this simplified build. */

/* Timer code (gettimeofday). */
double polybench_t_start, polybench_t_end;

static
double rtclock()
{
#ifdef POLYBENCH_TIME
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
#else
    return 0;
#endif
}


#if 0
/* Cycle-accurate timer removed for simplicity. */
#endif

void polybench_flush_cache()
{
  int cs = POLYBENCH_CACHE_SIZE_KB * 1024 / sizeof(double);
  double* flush = (double*) calloc (cs, sizeof(double));
  int i;
  double tmp = 0.0;
/* OpenMP parallel flush removed for this example. */
  for (i = 0; i < cs; i++)
    tmp += flush[i];
  assert (tmp <= 10.0);
  free (flush);
}


#if 0
/* Linux FIFO scheduler support removed. */
#endif

/* PAPI support removed for this simplified build. */

void polybench_prepare_instruments()
{
#ifndef POLYBENCH_NO_FLUSH_CACHE
  polybench_flush_cache ();
#endif
}


void polybench_timer_start()
{
  polybench_prepare_instruments ();
  /* Use gettimeofday-based timer */
  polybench_t_start = rtclock ();
}


void polybench_timer_stop()
{
  /* Use gettimeofday-based timer */
  polybench_t_end = rtclock ();
}


void polybench_timer_print()
{
#ifdef POLYBENCH_GFLOPS
      if  (polybench_program_total_flops == 0)
    {
      printf ("[PolyBench][WARNING] Program flops not defined, use polybench_set_program_flops(value)\n");
      printf ("%0.6lf\n", polybench_t_end - polybench_t_start);
    }
      else
    printf ("%0.2lf\n",
        (polybench_program_total_flops /
         (double)(polybench_t_end - polybench_t_start)) / 1000000000);
#else
      printf ("%0.6f\n", polybench_t_end - polybench_t_start);
#endif
}

static
void *
xmalloc (size_t num)
{
  void* tmp = NULL;
  int ret = posix_memalign (&tmp, 32, num);
  if (! tmp || ret)
    {
      fprintf (stderr, "[PolyBench] posix_memalign: cannot allocate memory");
      exit (1);
    }
  return tmp;
}

void* polybench_alloc_data(unsigned long long int n, int elt_size)
{
  size_t val = n;
  val *= elt_size;
  void* ret = NULL;
  
  /* cudaMallocManaged in C requires the flags parameter; use global attachment. */
  cudaMallocManaged(&ret, val, cudaMemAttachGlobal);
  
  return ret;
}
