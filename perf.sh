#!/bin/bash

module load gcc/11.2.0

perf record ./fluid_sim

perf report -n > perfreport
