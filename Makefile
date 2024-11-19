CPP = g++ -Wall --std=c++20 -Ofast -ffast-math -march=native -ftree-vectorize -funroll-loops -fopenmp
SRCS = main.cpp fluid_solver.cpp EventManager.cpp

all:
	$(CPP) $(SRCS) -o fluid_sim

debug:
	$(CPP) $(SRCS) -pg -fno-omit-frame-pointer -o fluid_sim

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
