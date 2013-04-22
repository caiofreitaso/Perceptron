#include <cstdlib>
#include <ctime>
#include "../include/Neuron.hpp"

array<unsigned,3> layers;
DataSet<2,1> truthTable;

int main(int argc, char const *argv[])
{
	layers[0] = 2;
	layers[1] = 2;
	layers[2] = 1;
	Network mlp(layers,sigmoid);

	unsigned epochs = 20000;
	double learning = 0.8;
	double momentum = 0.3;

	if (argc > 1) {
		epochs = strtoul(argv[1], 0, 0);
		if (argc > 2)
		learning = atof(argv[2]);
		if (argc == 4)
			momentum = atof(argv[3]);
	}

	array<double,2> input;
	input[0] = 0;
	input[1] = 0;
	truthTable.add(input,0);
	input[1] = 1;
	truthTable.add(input,1);
	input[0] = 1;
	input[1] = 0;
	truthTable.add(input,1);
	input[1] = 1;
	truthTable.add(input,0);
	
	time_t start,end;
	time(&start);
	mlp.learn(truthTable,learning,momentum,epochs);
	time(&end);
	std::cout << difftime(end,start) << "\n";

	for (unsigned i = 1; i < 3; i++)
		for (unsigned j = 0; j < layers[i]; j++)
			for (unsigned k = 0; k < layers[i-1]; k++)
				std::cout << "w"<<i<<"("<<j<<","<<k<<"): "<< mlp[i][j][k].weight << "\n";
	std::cout << "\n";
	for (unsigned i = 0; i < truthTable.instances(); i++) {
		mlp = truthTable.input(i).toPointer();
		mlp.update();
		std::cout << "\t" << truthTable.honestOutput(i) << " -- " << mlp[2]()[0];
		std::cout << " {" << (truthTable.honestOutput(i) - mlp[2]()[0]) * (truthTable.honestOutput(i) - mlp[2]()[0]) << "}\n";
	}
	return 0;
}
