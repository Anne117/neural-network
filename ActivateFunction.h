#pragma once
#include <iostream>
#include <vector>
using namespace std;

class ActivateFunction
{
public:
	virtual ~ActivateFunction() {}
	virtual void use(vector<double> &value, int n) = 0;
	virtual void useDer(vector<double> &value, int n) = 0;
	virtual double useDer(double value) = 0;
};

class SigmoidActivation : public ActivateFunction
{
public:
	void use(vector<double> &value, int n) override;
	void useDer(vector<double> &value, int n) override;
	double useDer(double value) override;
};

class ReLUActivation : public ActivateFunction
{
public:
	void use(vector<double> &value, int n) override;
	void useDer(vector<double> &value, int n) override;
	double useDer(double value) override;
};

class ThxActivation : public ActivateFunction
{
public:
	void use(vector<double> &value, int n) override;
	void useDer(vector<double> &value, int n) override;
	double useDer(double value) override;
};
