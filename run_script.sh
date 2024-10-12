#!/bin/bash

#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -p <partition_name>
#SBATCH -J my_job

N=10  # number of times to run the program
PROGRAM=./fluid_sim  # path to your program

# run the program N times and measure the execution time with perf
for i in $(seq 1 $N); do
  perf stat -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores $PROGRAM
done

# calculate the average execution time
average_time=$(perf stat -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores -r $N $PROGRAM | awk '/seconds time elapsed/ {print $1}' | awk '{sum+=$1} END {print sum/NR}')
echo "Average execution time: $average_time seconds"
