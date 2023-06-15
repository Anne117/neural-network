#pragma once
#include <iostream>
#include <vector>
using namespace std;

class Matrix
{
	vector<vector<double>> matrix;
	int row, col;

public:
	void Init(int row, int col);														 
	void Rand();																		  
	static void Multi(const Matrix &m, vector<double> &neuron, int n, vector<double> &c); 
	static void Multi_T(const Matrix &m, vector<double> &b, int n, vector<double> &c);	 
	static void SumVector(vector<double> &a, vector<double> &b, int n);					 
	double &operator()(int i, int j);
	friend std::ostream &operator<<(std::ostream &os, const Matrix &m); 
	friend std::istream &operator>>(std::istream &is, Matrix &m);		
};
