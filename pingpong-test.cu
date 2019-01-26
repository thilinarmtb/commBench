#include "comm-bench.h"
#include "comm-bench-cuda.h"

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // > 0 verbose output
  int verbose = 0;

  // get device id based on rank
  int device_id = getDeviceId(MPI_COMM_WORLD, verbose);
  cudaSetDevice(device_id);
  if(verbose) printf("device_id at rank=%d is %d\n", rank, device_id);

  int N = 1024;
  int M = N;
  int NITER = 100000;
  int size = N*sizeof(float);
  int cuda_aware_mpi = 0; 

#if defined(CUDA_AWARE_MPI)
  cuda_aware_mpi = 1;
#endif

  float *ha, *hb, *result;
  float *da, *db, *dresult;

  // Allocate Memory on host
  ha = (float *)malloc(size);
  hb = (float *)malloc(size);
  result = (float *) calloc(N, sizeof(float));

  // Allocate memory on device
  cudaMalloc(&da, size);
  cudaMalloc(&db, size);
  cudaMalloc(&dresult, size); cudaMemset(dresult, 0, size); 

  // Fill a with bogus values
  for(int i = 0; i < N; i++) {
    ha[i] = 1.0*i + 10.0;
  } 

  if(verbose) printf("ha[0] at rank=%d is %lf\n", rank, ha[0]);

  // Copy values to device arrays
  cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
  cudaMemcpy(db, ha, size, cudaMemcpyHostToDevice);

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
#if defined(CUDA_AWARE_MPI)
  for(int i = 0; i < NITER; i++) {
    MPI_Allreduce(da, result, M, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  }
  cudaMemcpy(result, dresult, M*sizeof(float), cudaMemcpyDeviceToHost);
#else
  for(int i = 0; i < NITER; i++) {
    cudaMemcpy(hb, db, M*sizeof(float), cudaMemcpyDeviceToHost);
    MPI_Allreduce(hb, result, M, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  double diff = MPI_Wtime() - start;

  if(rank == 0) {
    printf("Test = pingpong, Time = %lf (s), result = %f, CUDA_AWARE_MPI = %d\n",
      diff/NITER, result[0], cuda_aware_mpi);
  }

  cudaFree(da);
  cudaFree(db);
  cudaFree(dresult);
  free(ha);
  free(hb);
  free(result);

  MPI_Finalize(); 

  return 0;
}
