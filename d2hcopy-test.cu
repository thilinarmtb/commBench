#include "comm-bench.h"
#include "comm-bench-cuda.h"

inline cudaError_t cudaCheck(cudaError_t result) {
#if defined(COMMBENCH_DEBUG)
  if(result != cudaSuccess) {
    fprintf(stderr,"CUDA runtime error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void timeCopy(float *h_a, float *d_a, int bytes, int NITER, int rank) {
  cudaEvent_t start, stop;
  float time0;
  float time1;

  cudaCheck(cudaEventCreate(&start)); 
  cudaCheck(cudaEventCreate(&stop)); 

  cudaCheck(cudaEventRecord(start, 0));
  cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaEventRecord(stop, 0));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&time0, start, stop));

  cudaCheck(cudaEventRecord(start, 0));
  for(int i=0; i<NITER; i++) {
    cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice)); 
  }
  cudaCheck(cudaEventRecord(stop, 0));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&time1, start, stop));
  time1 /= NITER;

  if(rank == 0) {
    printf("Test=h2d,size=%d,time0=%f,time1=%f\n",bytes,time0, time1);
  }

  cudaCheck(cudaEventRecord(start, 0));
  cudaCheck(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
  cudaCheck(cudaEventRecord(stop, 0));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&time0, start, stop));

  cudaCheck(cudaEventRecord(start, 0));
  for(int i=0; i<NITER; i++) {
    cudaCheck(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost)); 
  }
  cudaCheck(cudaEventRecord(stop, 0));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&time1, start, stop));
  time1 /= NITER;

  if(rank == 0) {
    printf("Test=d2h,size=%d,time0=%f,time1=%f\n",bytes,time0, time1);
  }

  cudaCheck(cudaEventDestroy(start));
  cudaCheck(cudaEventDestroy(stop));

  return;
}

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  int rank, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  // > 0 verbose output
  int verbose = 0;

  // get device id based on rank
  int device_id = getDeviceId(MPI_COMM_WORLD, verbose);
  cudaSetDevice(device_id);
  if(verbose) printf("device_id at rank=%d is %d\n", rank, device_id);

  int N = atoi(argv[1]);
  int NITER = 100000;
  int size = N*sizeof(float);

  float *ha;
  float *da;

  // Allocate Memory on host
  ha = (float *)malloc(size);

  // Allocate memory on device
  cudaMalloc(&da, size);

  // Fill a with bogus values
  for(int i = 0; i < N; i++) {
    ha[i] = 1.0*i + 10.0;
  } 

  if(rank ==0) printf("np = %d\n", np);
  timeCopy(ha, da, size, NITER, rank);

  cudaFree(da);
  free(ha);

  MPI_Finalize(); 

  return 0;
}
