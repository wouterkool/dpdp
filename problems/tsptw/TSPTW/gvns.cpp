#include "gvns.h"
//------------------------------------------------------------------------------
GVNS::GVNS(TSPTW *tsp, int max)
{
    this->tsp = tsp;
    this->max = max;
}
//------------------------------------------------------------------------------
GVNS::~GVNS()
{
    
}
//------------------------------------------------------------------------------
TSPTWSolution *GVNS::optimize()
{    
	TSPTWCons *cons = new TSPTWCons(tsp);
    LocalSearch *ls = new LocalVND();
	
    int levelmax = 8;//8//(this->tsp->numNodes/40)+5;
    int itermax = 30;//30//(this->tsp->numNodes/20)+10;
    
    TSPTWSolution *best = cons->process();
    int bestfo = best->getPathDistance();
    
    int iter = 0;
    while (iter < this->max)
    {
    	iter++;
    	
    	TSPTWSolution *s = cons->process();
    	
        VNS *ils = new VNS(s, ls, levelmax, itermax);
        TSPTWSolution *saux = ils->optimize();
        
        delete ils;
        delete s;
        
        int sauxfo = saux->getPathDistance();
        
        if (sauxfo < bestfo)
        {
            delete best;
            best = saux;
            bestfo = sauxfo;
        }
        else
        {
            delete saux;
        }
    }   

    delete cons;
    delete ls;
    
    return best;
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
