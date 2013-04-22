FLAGS=-W -Wall -Werror -Wno-reorder

all: test-xor test-and

test-%: test/%.cpp include/Neuron.hpp include/DataSet.hpp include/MersenneTwister.h
	g++ $< $(FLAGS) -o bin/$@

clean:
	rm -f bin/*