#if !defined(comm_bench_)
#define comm_bench_

#include <stdio.h>
#include <mpi.h>
#include <assert.h>

int getDeviceId(MPI_Comm c, int verbose);

#endif
