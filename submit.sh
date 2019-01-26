#!/bin/bash

### Begin LSF Directives ###
#BSUB -P CSC262
#BSUB -W 1:00
#BSUB -nnodes 2
 
cd /ccs/home/thilina/testMPI
jsrun --smpiargs="-gpu" -r1 -a6 -c6 -g6 ./$1_test.out   256
jsrun --smpiargs="-gpu" -r1 -a6 -c6 -g6 ./$1_test.out   512
jsrun --smpiargs="-gpu" -r1 -a6 -c6 -g6 ./$1_test.out  1024
jsrun --smpiargs="-gpu" -r1 -a6 -c6 -g6 ./$1_test.out  2048
jsrun --smpiargs="-gpu" -r1 -a6 -c6 -g6 ./$1_test.out  4096
jsrun --smpiargs="-gpu" -r1 -a6 -c6 -g6 ./$1_test.out  8192
jsrun --smpiargs="-gpu" -r1 -a6 -c6 -g6 ./$1_test.out 16384
jsrun --smpiargs="-gpu" -r1 -a6 -c6 -g6 ./$1_test.out 32768 
