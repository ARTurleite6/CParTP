#!/bin/bash

#SBATCH --partition=cpar           # Partition to use
#SBATCH --constraint=k20

nvprof ./fluid_sim

