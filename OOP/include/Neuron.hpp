#ifndef NEURON_HPP
#define NEURON_HPP

#include "MersenneTwister.h"
#include "DataSet.hpp"
#include <cmath>

class Neuron;
class Layer;

inline double identity(double x) { return x; }
inline double sigmoid(double x) { return 1/(1 + exp(-x)); }
inline double hiperbolicTangent(double x) { double e = exp(x+x); return (e-1)/(e+1); }

struct Connection
{
	Neuron* target;
	double weight;
	double history;
};

class Neuron
{
	double output;
	double (*activate)(double);
	double bias;
	unsigned n_connections;
	Connection* connections;
	double error;

	void finalWeight(double err, double learningRate, double momentum)
	{
		error = output * (1 - output) * err;

		double history, t;
		t = learningRate * error;
		bias -= t;

		for(unsigned i = 0; i < n_connections; i++) {
			history = connections[i].weight - connections[i].history;
			connections[i].history = connections[i].weight;
			connections[i].weight += t*connections[i].target->output + momentum*history;
		}
	}

	public:
	Neuron();
	Neuron(unsigned connections, double (*function)(double));
	~Neuron() { delete[] connections; }

	double operator()() const { return output; }
	void operator=(double i) { output = i; }
	Connection& operator[](unsigned i) { return connections[i]; }
	Connection const& operator[](unsigned i) const { return connections[i]; }

	double sum() const;
	void update() { output = activate(sum() - bias); }

	void connect(Layer const& lastLayer);
	void updateWeight(double expected, double learningRate, double momentum);
	void updateWeight(unsigned j, Layer const& lastLayer, double learningRate, double momentum);
};

class Layer
{
	unsigned n_neurons;
	Neuron** neurons;

	friend class Neuron;
	friend class Network;

	public:
	Layer(unsigned neurons);
	Layer(unsigned neurons, Layer const& lastLayer, double (*function)(double));

	double* operator()() const;
	void operator=(double const* i);
	Neuron& operator[](unsigned i) { return *neurons[i]; }
	Neuron const& operator[](unsigned i) const { return *neurons[i]; }

	void update();
	void updateWeight(double const* expected, double learningRate, double momentum);
	void updateWeight(Layer const& lastLayer, double learningRate, double momentum);
};

class Network
{
	unsigned n_layers;
	Layer** layers;
	
	public:
	template<unsigned N>
	Network(array<unsigned,N> layers, double (*function)(double));

	void operator=(double const* input) { *layers[0] = input; }
	Layer const& operator[](unsigned i) { return *layers[i]; }

	void update();

	template<unsigned I, unsigned O>
	void learn(DataSet<I,O> data, double learningRate, double momentum, unsigned epochs);
};

Neuron::Neuron()
: output(0), activate(identity),bias(0),n_connections(0),connections(0),error(0)
{ }
Neuron::Neuron(unsigned connections, double (*function)(double))
: output(0), activate(function),bias(-1),n_connections(connections),
  connections(new Connection[connections]),error(0)
{
	static MTRand random;
	for(unsigned i = 0; i < n_connections; i++) {
		this->connections[i].weight = (random.rand(4.8) - 2.4)/n_connections;
		this->connections[i].history = 0;
	}
}
double Neuron::sum() const
{
	double ret = 0;
	for (unsigned i = 0; i < n_connections; i++)
		ret += connections[i].target->output * connections[i].weight;
	return ret;
}
void Neuron::connect(Layer const& lastLayer)
{
	for (unsigned i = 0; i < n_connections; i++)
		connections[i].target = lastLayer.neurons[i];
}
void Neuron::updateWeight(double expected, double learningRate, double momentum)
{
	finalWeight(expected - output, learningRate, momentum);
}
void Neuron::updateWeight(unsigned j, Layer const& lastLayer, double learningRate, double momentum)
{
	double err = 0;
	for(unsigned i = 0; i < lastLayer.n_neurons; i++)
		err += lastLayer.neurons[i]->connections[j].weight * lastLayer.neurons[i]->error;
	finalWeight(err, learningRate, momentum);
}

Layer::Layer(unsigned neurons)
: n_neurons(neurons), neurons(new Neuron*[neurons])
{
	for (unsigned i = 0; i < n_neurons; i++)
		this->neurons[i] = new Neuron();
}
Layer::Layer(unsigned neurons, Layer const& lastLayer, double (*function)(double))
: n_neurons(neurons), neurons(new Neuron*[neurons])
{
	for (unsigned i = 0; i < n_neurons; i++) {
		this->neurons[i] = new Neuron(lastLayer.n_neurons,function);
		this->neurons[i]->connect(lastLayer);
	}
}
double* Layer::operator()() const
{
	double* ret = new double[n_neurons];
	for (unsigned i = 0; i < n_neurons; i++)
		ret[i] = (*neurons[i])();
	return ret;
}
void Layer::operator=(double const* input)
{
	for(unsigned i = 0; i < n_neurons; i++)
		*neurons[i] = input[i];
}
void Layer::update()
{
	for(unsigned i = 0; i < n_neurons; i++)
		neurons[i]->update();
}
void Layer::updateWeight(double const* expected, double learningRate, double momentum)
{
	for(unsigned i = 0; i < n_neurons; i++)
		neurons[i]->updateWeight(expected[i],learningRate,momentum);
}
void Layer::updateWeight(Layer const& lastLayer, double learningRate, double momentum)
{
	for(unsigned i = 0; i < n_neurons; i++)
		neurons[i]->updateWeight(i,lastLayer,learningRate,momentum);
}

template<unsigned N>
Network::Network(array<unsigned,N> layers, double (*function)(double))
: n_layers(N), layers(new Layer*[N])
{
	if (N > 1) {
		this->layers[0] = new Layer(layers[0]);
		for(unsigned i = 1; i < N; i++)
			this->layers[i] = new Layer(layers[i],*this->layers[i-1],function);
	}
}
void Network::update()
{
	for(unsigned i = 1; i < n_layers; i++)
		layers[i]->update();
}
template<unsigned I, unsigned O>
void Network::learn(DataSet<I,O> data, double learningRate, double momentum, unsigned epochs)
{
	if (I == layers[0]->n_neurons && O == layers[n_layers-1]->n_neurons) {
		array<double,O> real;
		array<double,O> expected;
		for(unsigned e = 0, i = 0; e < epochs; e++) {
//			std::cout << e << " [" << i << "]: ";
			
			(*layers[0]) = data.input(i).toPointer();
			update();
			real = (*layers[n_layers-1])();
			expected = data.output(i);
//			for (unsigned k = 0; k < O; k++)
//				std::cout << "\t" << expected[k] << " -- " << real[k] << " {" << (expected[k] - real[k]) * (expected[k] - real[k]) << "}\n";

			layers[n_layers-1]->updateWeight(expected.toPointer(),learningRate,momentum);
			for(unsigned k = n_layers-2; k > 0; k--)
				layers[k]->updateWeight(*layers[k+1],learningRate,momentum);

			if (i == data.instances() - 1)
				i = 0;
			else
				i++;
		}
	}	
}
#endif