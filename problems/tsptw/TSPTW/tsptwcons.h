#ifndef TSPTWCONS_
#define TSPTWCONS_
//------------------------------------------------------------------------------
#include "includes.h"
#include "tsptw.h"
#include "tsptwsolution.h"
//------------------------------------------------------------------------------
class TSPTWCons
{
	private:
		TSPTW *tsp;
		
		int* inc;
		int* pen;
		
		int *feasible;
		int feasibleSize;
		
		int *unfeasible;
		int unfeasibleSize;
		
		int lastExchangeCons;
		
		TSPTWSolution *random();
		void local(TSPTWSolution *solution);
		void calcSets(TSPTWSolution *s);
		void exchangeCons(TSPTWSolution *s, int p, int paux);
		
		int penalty(TSPTWSolution *s);
		int penalty(TSPTWSolution *s, int p, int paux);
		
	public:
		TSPTWCons(TSPTW *tsp);
		~TSPTWCons();
		
		TSPTWSolution *process();
		
};
//------------------------------------------------------------------------------   

#endif
