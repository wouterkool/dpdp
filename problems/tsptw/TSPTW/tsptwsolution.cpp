#include "tsptwsolution.h"
//------------------------------------------------------------------------------
TSPTWSolution::TSPTWSolution(TSPTW *tspAux)
{
    this->tsp = tspAux;
    this->solution = new int[this->tsp->numNodes];
}
//------------------------------------------------------------------------------
TSPTWSolution::~TSPTWSolution()
{
    delete[] this->solution;
}
//------------------------------------------------------------------------------
TSPTWSolution *TSPTWSolution::clone()
{
    TSPTWSolution *s = new TSPTWSolution(this->tsp);
    int numNodes = this->tsp->numNodes;
    for (int i = 0; i < numNodes; i++)
    {
        s->solution[i] = this->solution[i];
    }
    return s;
}
//------------------------------------------------------------------------------
int TSPTWSolution::getPathDistance()
{
	int numNodes = this->tsp->numNodes;
	int sum = this->tsp->matrix[this->solution[numNodes-1]][this->solution[0]];
    for (int i = 0; i < numNodes-1; i++)
    {
    	sum = sum + this->tsp->matrix[this->solution[i]][this->solution[i+1]];
    }
    return sum;
}
//------------------------------------------------------------------------------
int TSPTWSolution::penalty()
{
	int sum = 0;
	int psum = 0;
	int ni = 0;
	int next = 0;
	int d = 0;
	int numNodes = this->tsp->numNodes;
    for (int i = 0; i < numNodes-1; i++)
    {
        ni = this->solution[i];
        next = this->solution[i+1];
        d = this->tsp->matrix[ni][next];
        sum = max(this->tsp->readytime[ni], sum) + d;
        psum += max(0, sum - this->tsp->duedate[next]);
    }
    return psum;
}
//------------------------------------------------------------------------------
void TSPTWSolution::print()
{
	int numNodes = this->tsp->numNodes;
	cout << endl;
    for (int i = 0; i < numNodes; i++)
    {
        cout << solution[i] << " - "; 
    }
    cout << endl;
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
