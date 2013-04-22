#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <iostream>
#include <cmath>
#include "MersenneTwister.h"
#include "DataSet.hpp"

namespace Perceptron {
	inline double identity(double x) { return x; }
	inline double sigmoid(double x) { return 1/(1 + exp(-x)); }
	inline double hiperbolicTangent(double x) { double e = exp(x+x); return (e-1)/(e+1); }

	struct Layer {
		unsigned neurons;
		double* output;
		double* bias;
		double** connections;
		double** history;
		double* error;

		Layer(unsigned neurons);
		Layer(unsigned neurons, unsigned connections);
		~Layer();
	};

	struct Network {
		unsigned count;
		Layer** layers;

		template<unsigned N>
		Network(array<unsigned,N> layers);

		Network& operator=(double const* i);
		double const* toArray() const;
		
		~Network();
	};

	void update(Layer& target, Layer const& previous, double (*function)(double))
	{
		for(unsigned i = 0; i < target.neurons; i++) {
			target.output[i] = 0;
			for(unsigned j = 0; j < previous.neurons; j++)
				target.output[i] += previous.output[j] * target.connections[i][j];
			target.output[i] = function(target.output[i] - target.bias[i]);
		}
	}
	void update(Network& target, double(*function)(double))
	{
		for (unsigned i = 1; i < target.count; i++)
			update(*target.layers[i], *target.layers[i-1], function);
	}

	inline void finalWeight(Layer& target, unsigned i, double const* previous, unsigned p_neurons, double error, double learningRate, double momentum)
	{
		double history;

		target.error[i] = target.output[i] * (1 - target.output[i]) * error;
		
		double t = learningRate * target.error[i];
		target.bias[i] -= t;
		
		for(unsigned j = 0; j < p_neurons; j++) {
			history = target.connections[i][j] - target.history[i][j];
			target.history[i][j] = target.connections[i][j];
			target.connections[i][j] += t * previous[j] + momentum * history;
		}
	}
	inline double error(unsigned i, Layer const& next)
	{
		double err = 0;
		for (unsigned j = 0; j < next.neurons; j++)
			err += next.connections[j][i] * next.error[j];
		return err;
	}
	void updateWeights(Layer& target, Layer const& previous, Layer const& next, double learningRate, double momentum)
	{
		for(unsigned i = 0; i < target.neurons; i++)
			finalWeight(target,i,previous.output,previous.neurons,error(i,next),learningRate,momentum);
	}
	void updateWeights(Layer& target, Layer const& previous, double const* next, double learningRate, double momentum)
	{
		for(unsigned i = 0; i < target.neurons; i++)
			finalWeight(target,i,previous.output,previous.neurons,next[i] - target.output[i],learningRate,momentum);
	}

	template<unsigned I, unsigned O>
	void learn(Network& target, unsigned epochs, double learningRate, double momentum, DataSet<I,O> const& data, double (*function)(double))
	{
		if (I == target.layers[0]->neurons && O == target.layers[target.count-1]->neurons) {
			array<double,O> real;
			array<double,O> expected;
			for(unsigned e = 0, i = 0; e < epochs; e++) {
//				std::cout << e << " [" << i << "]: ";
				target = data.input(i).toPointer();
				
				update(target, function);
				real = target.toArray();
				expected = data.output(i);
//				for (unsigned k = 0; k < O; k++)
//					std::cout << "\t" << expected[k] << " -- " << real[k] << " {" << (expected[k] - real[k]) * (expected[k] - real[k]) << "}\n";

				updateWeights(*target.layers[target.count-1], *target.layers[target.count-2],
							  expected.toPointer(), learningRate, momentum);
				for (unsigned k = target.count-2; k > 0; k--)
					updateWeights(*target.layers[k],*target.layers[k-1],*target.layers[k+1],learningRate,momentum);

				if (i == data.instances() - 1)
					i = 0;
				else
					i++;
			}
		}
	}
}

Perceptron::Layer::Layer(unsigned neurons)
: neurons(neurons),output(new double[neurons]), bias(0), connections(0),
  history(0), error(0)
{ }
Perceptron::Layer::Layer(unsigned neurons, unsigned connections)
: neurons(neurons), output(new double[neurons]), bias(new double[neurons]),
  connections(new double*[neurons]), history(new double*[neurons]),
  error(new double[neurons])
{
	static MTRand random;
	for(unsigned i = 0; i < neurons; i++) {
		this->connections[i] = new double[connections];
		history[i] = new double[connections];
		bias[i] = -1;
		for (unsigned j = 0; j < connections; j++) {
			this->connections[i][j] = (random.rand(4.8) - 2.4)/connections;
			history[i][j] = 0;
		}
	}
}
Perceptron::Layer::~Layer()
{
	delete[] output;
	delete[] bias;
	delete[] error;
	if (connections)
		for(unsigned i = 0; i < neurons; i++)
			delete[] connections[i];
	delete[] connections;
}

template<unsigned N>
Perceptron::Network::Network(array<unsigned,N> layers)
: count(N), layers(new Layer*[N])
{
	this->layers[0] = new Layer(layers[0]);
	for (unsigned i = 1; i < N; i++)
		this->layers[i] = new Layer(layers[i],layers[i-1]);
}
Perceptron::Network::~Network()
{
	for(unsigned i = 0; i < count; i++)
		delete layers[i];
	delete[] layers;
}
double const* Perceptron::Network::toArray() const { return layers[count-1]->output; }
Perceptron::Network& Perceptron::Network::operator=(double const* i)
{
	for(unsigned k = 0; k < layers[0]->neurons; k++)
		layers[0]->output[k] = i[k];
	return *this;
}

#endif
