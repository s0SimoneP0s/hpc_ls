#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 100x10000. */
#include "jacobi-1d-imper.h"


// #define THREADS_GPU 64
// #define THREADS_CPU 4


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
  int t, i, j;
  #pragma omp target data map(tofrom: A[0:n], B[0:n])
  for (t = 0; t < _PB_TSTEPS; t++)
  {
    #pragma omp target teams distribute parallel for simd 
    for (i = 1; i < _PB_N - 1; i++)
      B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
    #pragma omp target teams distribute parallel for simd
    for (j = 1; j < _PB_N - 1; j++) 
      A[j] = B[j];
  }
}

int omp_get_team_num(void);
int omp_get_thread_limit(void);
int omp_get_num_teams(void);

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;
  printf("n = %d\ntsteps = %d\n",n,tsteps);

  #pragma omp target teams num_teams(2) thread_limit(6)
  {

      int nteams  = omp_get_num_teams();

      #pragma omp parallel
      {
          #pragma omp single
          {
              printf("Teams: %d\nThread limit: %d\n",
                     nteams, omp_get_thread_limit());
          }
      }
  }

#pragma omp target teams distribute num_teams(2) thread_limit(6)
for(int i = 0; i < 2; i++) 
{
    int num_teams = omp_get_num_teams();
    int team_num = omp_get_team_num();
    int thread_limit = omp_get_thread_limit();
    
    printf("Iterazione %d - Team %d di %d, Thread limit: %d\n", 
           i, team_num, num_teams, thread_limit);
}

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

  return 0;
}
