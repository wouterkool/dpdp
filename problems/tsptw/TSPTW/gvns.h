#ifndef GVNS_
#define GVNS_
//------------------------------------------------------------------------------
#include "includes.h"
#include "metaheuristic.h"
#include "tsptwcons.h"
#include "localvnd.h"
#include "vns.h"
//------------------------------------------------------------------------------
class GVNS: public MetaHeuristic
{
	private:
		TSPTW *tsp;
		int max;
		
	public:
		GVNS(TSPTW *tsp, int max);
		virtual ~GVNS();
		
		virtual TSPTWSolution *optimize();
};
//------------------------------------------------------------------------------   

#endif
