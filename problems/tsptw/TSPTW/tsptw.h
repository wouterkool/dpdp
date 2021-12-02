#ifndef TSPTW_
#define TSPTW_
//------------------------------------------------------------------------------
#include "includes.h"
#include "tsptw.h"
//------------------------------------------------------------------------------
class TSPTW
{
	public:
		int numNodes;
		
		int **matrix;
		int *readytime;
		int *duedate;
		
		bool **compatible;
		
		TSPTW(int numNodes);
		~TSPTW();
		
		TSPTW *clone();
		
		void setDistance(int n1, int n2, int d);
		
		int getWindowWidth(int n);
		
		void processCompatible(); //Gendreu
		void triangleInequality(); //to obey trigle inequality
		
		void print();
};
//------------------------------------------------------------------------------   

#endif
