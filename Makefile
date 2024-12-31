CPP = nvcc
CXXFLAGS = --std=c++17 -O3  -Wno-deprecated-gpu-targets
SRCS = main.cu fluid_solver.cu EventManager.cpp 

all:
	$(CPP) $(CXXFLAGS) $(SRCS) -o fluid_sim

run:
	sbatch ./run.sh

copy-to-search:
	scp -p run.sh $(SRCS) fluid_solver.h EventManager.h events.txt Makefile a97027@s7edu.di.uminho.pt:~/3dfluid

runseq:
	OMP_NUM_THREADS=1 ./fluid_sim

runpar:
	OMP_NUM_THREADS=16 ./fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@echo Done.
