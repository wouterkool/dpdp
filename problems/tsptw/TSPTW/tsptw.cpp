#include "tsptw.h"
//------------------------------------------------------------------------------
TSPTW::TSPTW(int numNodes)
{
    this->numNodes = numNodes;
    this->matrix = new int*[numNodes];
    this->readytime = new int[numNodes];
    this->duedate = new int[numNodes];
    this->compatible = new bool*[numNodes];
    for (int i = 0; i < numNodes; i++)
    {
        this->matrix[i] = new int[numNodes];
        for (int j = 0; j < numNodes; j++)
        {
            this->matrix[i][j] = 0;
        }
        this->compatible[i] = new bool[numNodes];
        for (int j = 0; j < numNodes; j++)
        {
            this->compatible[i][j] = true;
        }
        this->readytime[i] = 0;
        this->duedate[i] = 0;
    }
}
//------------------------------------------------------------------------------
TSPTW::~TSPTW()
{
    for (int i = 0; i < this->numNodes; i++)
    {
    	delete[] this->matrix[i];
    	delete[] this->compatible[i];
    }
    delete[] this->matrix;
    delete[] this->readytime;
    delete[] this->duedate;
    delete[] this->compatible;
}
//------------------------------------------------------------------------------
void TSPTW::setDistance(int n1, int n2, int d)
{
    this->matrix[n1][n2] = d;
    this->matrix[n2][n1] = d;
}
//------------------------------------------------------------------------------
int TSPTW::getWindowWidth(int n)
{
    return this->duedate[n] - this->readytime[n];
}
//------------------------------------------------------------------------------
void TSPTW::print()
{
    for (int i = 0; i < this->numNodes; i++)
    {
       for (int j = 0; j < this->numNodes; j++)
       {
            cout << "(" << i << "," << j << ")$" << this->matrix[i][j] << ";";
       }
       cout << endl;
    }
    
    for (int i = 0; i < this->numNodes; i++)
    {
        cout << i << "#[" << this->readytime[i] << "," << this->duedate[i] << "];";
    }
}
//------------------------------------------------------------------------------
void TSPTW::processCompatible()
{
	int n = 0;
	
	int ai = 0;
	int cij = 0;
	int bj = 0;
	
    for (int i = 0; i < this->numNodes; i++)
    {
       ai = this->readytime[i];
       for (int j = 0; j < this->numNodes; j++)
       {
            cij = this->matrix[i][j];
            bj = this->duedate[j];
            if (ai + cij > bj)
            {
            	this->compatible[i][j] = false;
                n++;
            }
       }
    }
}
//------------------------------------------------------------------------------
void TSPTW::triangleInequality()
{
    for (int i = 0; i < this->numNodes; i++)
    {
       for (int j = 0; j < this->numNodes; j++)
       {
           for (int k = 0; k < this->numNodes; k++)
           {
                if (this->matrix[i][j] > this->matrix[i][k] + this->matrix[k][j])
                {
                    this->matrix[i][j] = this->matrix[i][k] + this->matrix[k][j];
                }
           }
       }
    }
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
