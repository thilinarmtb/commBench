CUDA_AWARE_MPI?=0
DEBUG?=0
CXXFLAGS?=-O0
NVCCFLAGS?=-O0

CXX=mpic++
NVCC=nvcc
LDFLAGS=-L${CUDA_DIR}/lib64 -lcudart

ifneq ($(CUDA_AWARE_MPI),0)
  CXXFLAGS+=-DCOMMBENCH_CUDA_AWARE_MPI
  NVCCFLAGS+=-DCOMMBENCH_CUDA_AWARE_MPI
endif

ifneq ($(DEBUG),0)
  CXXFLAGS+=-g -DCOMMBENCH_DEBUG
  NVCCFLAGS+=-g -DCOMMBENCH_DEBUG
endif

.PHONY: all
all: allreduce pingpong d2hcopy

.PHONY: allreduce
allreduce: allreduce-test.o comm-bench.o comm-bench-cuda.o
	$(CXX) -o allreduce.o $^ $(LDFLAGS)

.PHONY: pingpong
pingpong: pingpong-test.o comm-bench.o comm-bench-cuda.o
	$(CXX) -o pingpong.o $^ $(LDFLAGS)

.PHONY: d2hcopy
d2hcopy: d2hcopy-test.o comm-bench.o comm-bench-cuda.o
	$(CXX) -o d2hcopy.o $^ $(LDFLAGS)

### Object files
.PHONY: d2hcopy.o
d2hcopy-test.o:
	$(NVCC) $(NVCCFLAGS) -c d2hcopy-test.cu -o d2hcopy-test.o

.PHONY: allreduce-test.o
allreduce-test.o:
	$(NVCC) $(NVCCFLAGS) -c allreduce-test.cu -o allreduce-test.o

.PHONY: pingpong-test.o
pingpong-test.o:
	$(NVCC) $(NVCCFLAGS) -c pingpong-test.cu -o pingpong-test.o

.PHONY: comm-bench.o
comm-bench.o:
	$(CXX) $(CXXFLAGS) -c comm-bench.c -o comm-bench.o

.PHONY: comm-bench-cuda.o
comm-bench-cuda.o:
	$(NVCC) $(NVCCFLAGS) -c comm-bench-cuda.cu -o comm-bench-cuda.o

.PHONY: clean
clean:
	@rm *.o

submit-%:
	bsub -W 1:00 -P CSC262 -nnodes 2 -J $* -o $* -e $* ./submit.sh $*
