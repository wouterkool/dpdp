#ifndef TSPTWSOLUTION_
#define TSPTWSOLUTION_
//------------------------------------------------------------------------------
#include "includes.h"
#include "tsptw.h"
//------------------------------------------------------------------------------
class TSPTWSolution
{
	private:
		
	public:
		
		TSPTW *tsp;
		
		int *solution;
		
		TSPTWSolution(TSPTW *tsp);
		~TSPTWSolution();
		
		TSPTWSolution *clone();
		
		int getPathDistance();
		int penalty();
		
		void print();
};
//------------------------------------------------------------------------------   

#endif
