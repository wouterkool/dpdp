#ifndef METAHEURISTIC_
#define METAHEURISTIC_
//------------------------------------------------------------------------------
#include "includes.h"
#include "tsptwsolution.h"
//------------------------------------------------------------------------------
class MetaHeuristic
{		
	public:
		
		virtual TSPTWSolution *optimize() = 0;
};
//------------------------------------------------------------------------------   

#endif
