#ifndef LOCAL2OPT_
#define LOCAL2OPT_
//------------------------------------------------------------------------------
#include "includes.h"
#include "localsearch.h"
#include "tsptw.h"
#include "tsptwsolution.h"
//------------------------------------------------------------------------------
class Local2Opt: public LocalSearch
{
	private:
		TSPTW *tsp;
		int* inc;
		
		int isFeasible(TSPTWSolution *s, int iaux, int jaux);
		void exchange(TSPTWSolution *s, int iaux, int jaux);
		
		void compInc(TSPTWSolution *s);
		
	public:
		Local2Opt();
		virtual ~Local2Opt();

		virtual void process(TSPTWSolution *solution);
};
//------------------------------------------------------------------------------   

#endif
