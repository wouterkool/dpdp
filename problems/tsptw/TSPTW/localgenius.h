#ifndef LOCALGENIUS_
#define LOCALGENIUS_
//------------------------------------------------------------------------------
#include "includes.h"
#include "localsearch.h"
#include "tsptw.h"
#include "tsptwsolution.h"
//------------------------------------------------------------------------------
class LocalGenius: public LocalSearch
{
	private:
		
		TSPTW *tsp;
		int* inc;
		
		int movement(TSPTWSolution *s, int npos);
		
		bool isFeasible(TSPTWSolution *s, int p, int paux);
		void exchange(TSPTWSolution *s, int p, int paux);
		void compInc(TSPTWSolution *s);
		
	public:
		LocalGenius();
		virtual ~LocalGenius();

		virtual void process(TSPTWSolution *solution);
};
//------------------------------------------------------------------------------   

#endif
