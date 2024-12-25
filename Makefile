CPP = nvcc
CXXFLAGS = --std=c++17 -O3 
SRCS = main.cu fluid_solver.cu EventManager.cpp resource_manager.cu

all:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim

run:
	sbatch ./run.sh

profile: all
	/usr/lib/linux-tools-5.15.0-122/perf stat -r 3 -e instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,branch,branch-misses ./fluid_sim

runseq:
	OMP_NUM_THREADS=1 ./fluid_sim

runpar:
	OMP_NUM_THREADS=16 ./fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@echo Done.
