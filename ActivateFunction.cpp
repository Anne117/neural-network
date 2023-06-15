#include "ActivateFunction.h"

// vector<double>&
void SigmoidActivation::use(vector<double> &value, int n)
{
	for (int i = 0; i < n; i++)
		value[i] = 1 / (1 + exp(-value[i]));
}

void SigmoidActivation::useDer(vector<double> &value, int n)
{
	for (int i = 0; i < n; i++)
		value[i] = value[i] * (1 - value[i]);
}

double SigmoidActivation::useDer(double value)
{
	return 1 / (1 + exp(-value));
}

// ReLU
void ReLUActivation::use(vector<double> &value, int n)
{
	for (int i = 0; i < n; i++)
		value[i] = std::max(0.01 * value[i], 1. + 0.01 * (value[i] - 1.));
}

void ReLUActivation::useDer(vector<double> &value, int n)
{
	for (int i = 0; i < n; i++)
		value[i] = (value[i] < 0 || value[i] > 1) ? 0.01 : 1;
}

double ReLUActivation::useDer(double value)
{
	if (value != value || value == std::numeric_limits<double>::infinity() || value == -std::numeric_limits<double>::infinity())
		throw std::runtime_error("ReLUActivation::useDer: value is nan or infinity");
	return (value < 0 || value > 1) ? 0.01 : 1;
}

// Thx
void ThxActivation::use(vector<double> &value, int n)
{
	for (int i = 0; i < n; i++)
	{
		if (value[i] < 0)
			value[i] = 0.01 * (exp(value[i]) - exp(-value[i])) / (exp(value[i]) + exp(-value[i]));
		else
			value[i] = (exp(value[i]) - exp(-value[i])) / (exp(value[i]) + exp(-value[i]));
	}
}

void ThxActivation::useDer(vector<double> &value, int n)
{
	for (int i = 0; i < n; i++)
	{
		if (value[i] < 0)
			value[i] = 0.01 * (1 - value[i] * value[i]);
		else
			value[i] = 1 - value[i] * value[i];
	}
}

double ThxActivation::useDer(double value)
{
	if (value < 0)
		return 0.01 * (exp(value) - exp(-value)) / (exp(value) + exp(-value));
	else
		return (exp(value) - exp(-value)) / (exp(value) + exp(-value));
}