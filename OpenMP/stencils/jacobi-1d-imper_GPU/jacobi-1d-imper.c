#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 100x10000. */
#include "jacobi-1d-imper.h"


in tomp_get_num_devices(void);

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

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_jacobi_1d_imper(int tsteps,
                                   int n,
                                   DATA_TYPE POLYBENCH_1D(A, N, n),
                                   DATA_TYPE POLYBENCH_1D(B, N, n)
                                   )
{

  char* num_teams_env = getenv("OMP_NUM_TEAMS");
  char* thread_limit_env = getenv("OMP_TEAMS_THREAD_LIMIT");
  printf("Teams: %s\nThread limit: %s\n",num_teams_env,thread_limit_env);

  int THREADS_CPU = atoi(num_teams_env);
  int THREADS_GPU = atoi(thread_limit_env);
  int t, i, j;
  #pragma omp target data map(tofrom: A[0:n]) map(alloc: B[0:n])
  {
    for (t = 0; t < _PB_TSTEPS; t++)
    {
      #pragma omp target teams distribute parallel for simd \
                  num_teams(THREADS_CPU) thread_limit(THREADS_GPU) \
                  schedule(static) 
      for (i = 1; i < _PB_N - 1; i++)
      {
        B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
      }
      #pragma omp target teams distribute parallel for simd \
                  num_teams(THREADS_CPU) thread_limit(THREADS_GPU) \
                  schedule(static) 
      for (j = 1; j < _PB_N - 1; j++)
      {
        A[i] = B[i];
      }
    }
  } 
}


int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;
  printf("n = %d\ntsteps = %d\n",n,tsteps);




  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);




  /* Initialize array(s). */
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

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

    printf("Numero dispositivi: %d\n", omp_get_num_devices());
    printf("Dispositivo default: %d\n", omp_get_default_device());
    


  return 0;
}
