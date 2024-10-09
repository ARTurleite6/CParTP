CPP = g++ -Wall --std=c++20 -O2 -funroll-loops -ftree-vectorize -msse4
SRCS = main.cpp fluid_solver.cpp EventManager.cpp

all:
	$(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid
	@echo Done.
