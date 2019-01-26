#include <unistd.h>
#include <stdlib.h>

#include "comm-bench.h"

int getDeviceId(MPI_Comm c, int verbose) {
  int rank, size;
  
  MPI_Comm_rank(c, &rank);
  MPI_Comm_size(c, &size);
    	     
  long int hostId = gethostid();

  long int* hostIds = (long int*) calloc(size,sizeof(long int));
  MPI_Allgather(&hostId,1,MPI_LONG,hostIds,1,MPI_LONG,MPI_COMM_WORLD);

  int device_id = 0;
  int totalDevices = 0;
  int r;
  for (r=0;r<rank;r++) {
    if (hostIds[r]==hostId) device_id++;
  }
  for (r=0;r<size;r++) {
    if (hostIds[r]==hostId) totalDevices++;
  }

  if(verbose)
    printf("rank %d with hostid %ld gets device %d out of %d\n",
      rank, hostId, device_id, totalDevices);

  return device_id;
}
