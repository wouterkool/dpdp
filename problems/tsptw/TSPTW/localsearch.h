#ifndef LOCALSEARCH_
#define LOCALSEARCH_
//------------------------------------------------------------------------------
#include "includes.h"
#include "tsptw.h"
#include "tsptwsolution.h"
//------------------------------------------------------------------------------
class LocalSearch
{		
    public:
		
		virtual void process(TSPTWSolution *s) = 0;
};
//------------------------------------------------------------------------------   

#endif
