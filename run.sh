#!/bin/bash

#SBATCH --partition=cpar           # Partition to use

nvprof ./fluid_sim

