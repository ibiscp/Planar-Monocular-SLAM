CXX=g++
CXXOPTS=-std=gnu++14 -Wall -Ofast -I /usr/include/eigen3
LDOPTS=-lpthread

.phony:	clean all

all:	runba_sim

runba_sim:	runba_sim.cpp
	$(CXX) -o $@ $(CXXOPTS) $^ $(LDOPTS)

clean:
	rm *.o *~ *.dat runba_sim
