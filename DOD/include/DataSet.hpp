#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>

template<typename T, unsigned N>
class array {
	T _a[N];

	public:
	array() { }
	array(array const& c);
	array(T const (&a)[N]);
	array(T (&a)[N]):_a(a) { }

	array& operator=(array const& c);
	array& operator=(T const* c);

	T& operator[](unsigned i) { return _a[i]; }
	T const& operator[](unsigned i) const { return _a[i]; }
	T const* toPointer() const { return _a; }
};

template<unsigned I, unsigned O>
class DataSet {
	std::vector< array<double,I> > _input;
	std::vector< array<double,O> > _output;

	public:
	void add(array<double,I> const& a, array<double,O> const& b);

	array<double,I>& input(unsigned i) { return _input[i]; }
	array<double,O>& output(unsigned i) { return _output[i]; }

	array<double,I> const& input(unsigned i) const { return _input[i]; }
	array<double,O> const& output(unsigned i) const { return _output[i]; }

	unsigned instances() const { return _input.size(); }
};

template<unsigned I>
class DataSet<I,1> {
	std::vector< array<double,I> > _input;
	std::vector< double > _output;

	public:
	void add(array<double,I> const& a, double const& b);

	array<double,I>& input(unsigned i) { return _input[i]; }
	array<double,1> output(unsigned i) const;
	double& honestOutput(unsigned i) { return _output[i]; }

	array<double,I> const& input(unsigned i) const { return _input[i]; }
	double const& honestOutput(unsigned i) const { return _output[i]; }

	unsigned instances() const { return _input.size(); }
};

template<typename T, unsigned N>
array<T,N>::array(array const& c)
{
	for (unsigned i = 0; i < N; i++)
		_a[i] = c._a[i];
}
template<typename T, unsigned N>
array<T,N>::array(T const (&a)[N])
{
	for (unsigned i = 0; i < N; i++)
		_a[i] = a[i];
}
template<typename T, unsigned N>
array<T,N>& array<T,N>::operator=(array<T,N> const& c)
{
	for (unsigned i = 0; i < N; i++)
		_a[i] = c._a[i];
	return *this;
}
template<typename T, unsigned N>
array<T,N>& array<T,N>::operator=(T const* c)
{
	for (unsigned i = 0; i < N; i++)
		_a[i] = c[i];
	return *this;
}

template<unsigned I, unsigned O>
void DataSet<I,O>::add(array<double,I> const& a, array<double,O> const& b)
{
	_input.push_back(a);
	_output.push_back(b);
}

template<unsigned I>
void DataSet<I,1>::add(array<double,I> const& a, double const& b)
{
	_input.push_back(a);
	_output.push_back(b);
}
template<unsigned I>
array<double,1> DataSet<I,1>::output(unsigned i) const
{
	array<double,1> ret;
	ret[0] = _output[i];
	return ret;
}

#endif