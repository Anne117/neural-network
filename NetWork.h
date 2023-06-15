#pragma once
#include "ActivateFunction.h"
#include "Matrix.h"
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;

struct data_NetWork
{					  
	int L;			  
	vector<int> size; 
};

class NetWork
{
	int L;
	vector<int> size;
	unique_ptr<ActivateFunction>actFunc;
				 
	vector<Matrix> weights;							
	vector<vector<double>> bios;					
	vector<vector<double>> neurons_val, neurons_err; 
	vector<double> neurons_bios_val;				 

public:
	void Init(data_NetWork data);
	void PrintConfig();
	void SetInput(vector<double> &values);

	double ForwardFeed();
	int SearchMaxIndex(vector<double> &value);
	void PrintValues(int L);

	void BackPropogation(int expect);
	void WeightsUpdater(double lr);

	void SaveWeights();
	void ReadWeights();

	void Softmax(vector<double> &values);

	~NetWork();
};
