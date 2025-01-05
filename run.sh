#!/bin/bash

#SBATCH --partition=cpar           # Partition to use
#SBATCH --constraint=k20
#SBATCH --exclusive

time ./fluid_sim

