#ifndef VNS_
#define VNS_
//------------------------------------------------------------------------------
#include "includes.h"
#include "tsptwsolution.h"
#include "metaheuristic.h"
#include "localsearch.h"
//------------------------------------------------------------------------------
class VNS: public MetaHeuristic
{
	private:
		LocalSearch *local;
		TSPTWSolution *solution;
		TSPTW *tsp;
		int itermax;
		int levelmax;
		int maxStop;
		
		int *inc;
		void disturb(TSPTWSolution *s, int level);
		bool isFeasible(TSPTWSolution *s, int p, int paux);
		void exchange(TSPTWSolution *s, int p, int paux);
		void compInc(TSPTWSolution *s);
		
	public:
		VNS(TSPTWSolution *s, LocalSearch *ls, int levelmax, int itermax);
		virtual ~VNS();
		
		virtual TSPTWSolution *optimize();
};
//------------------------------------------------------------------------------   

#endif
