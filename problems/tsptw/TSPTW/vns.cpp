#include "vns.h"
//------------------------------------------------------------------------------
//Sort
struct dist3 {int i,c;};
bool operator<(const dist3& a, const dist3& b)
{
    return a.c < b.c;
}
//------------------------------------------------------------------------------
VNS::VNS(TSPTWSolution *s, LocalSearch *ls, int levelmax, int itermax)
{
    this->solution = s;
    this->tsp = s->tsp;
    this->local = ls;
    this->levelmax = levelmax;
    this->itermax = itermax;
    
    this->inc = new int[this->tsp->numNodes];
}
//------------------------------------------------------------------------------
VNS::~VNS()
{
	delete[] inc;
}
//------------------------------------------------------------------------------
TSPTWSolution *VNS::optimize()
{    
    TSPTWSolution *s = this->solution->clone();
    
    int i = 0;
    int level = 1;
    int iterlevel = 0;
    int sFo = s->getPathDistance();
    
    int iter = 0;
    while (level < this->levelmax)
    {
    	iter++;
        TSPTWSolution *saux = s->clone();
        
        this->disturb(saux, level);
        
        this->local->process(saux);
        
        int sauxFo = saux->getPathDistance();
        
        if (sauxFo < sFo)
        {
            if (s != NULL)
            {  
                delete s;
            }
            s = saux;
            sFo = sauxFo;
            iterlevel = 0;
            level = 1;
        }
        else 
        {
        	if (saux != NULL)
            {  
                delete saux;
            }
            if (iterlevel > this->itermax)
            {
                level++;
                iterlevel = 0;
            }
        }
        iterlevel++;
        i++;
    }
    
    return s;
}
//------------------------------------------------------------------------------
void VNS::disturb(TSPTWSolution *s, int level)
{
	this->compInc(s);
	int numNodes = this->tsp->numNodes;
    int j = 0;
    int levelMax = level*2;
    int *poss = new int[numNodes-1];
    int possSize = 0;
    
    while (j < levelMax)
    {
    	possSize = 0;
    	
    	int n1pos = (genrand_int32()%(numNodes-1))+1;

        int naux = 0;
        
        for (int i = n1pos-1; i > 0; i--)
        {
        	naux = s->solution[i];
        	if (!this->tsp->compatible[n1pos][naux])
        	{
        		break;
        	}
        	poss[possSize] = i;
        	possSize++;
        }
        
        for (int i = n1pos+1; i < numNodes-1; i++)
        {
        	naux = s->solution[i];
        	if (!this->tsp->compatible[naux][n1pos])
        	{
        		break;
        	}
        	poss[possSize] = i;
        	possSize++;
        }
        
        int r = 0;
        int nauxpos = 0;
        while (possSize > 0)
        {
        	r = genrand_int32()%possSize;
        	nauxpos = poss[r];
        	if (this->isFeasible(s, n1pos, nauxpos))
	        {
        		this->exchange(s, n1pos, nauxpos);
        		break;
	        }
	        poss[r] = poss[possSize-1];
	        possSize--;
        }
        j++;
    }
    delete[] poss;
}
//------------------------------------------------------------------------------
bool VNS::isFeasible(TSPTWSolution *s, int p, int paux)
{
	int numNodes = this->tsp->numNodes;
	
	int sum = 0;
	int ni = 0;
	int next = 0;
	
	//1
	int min = p;
	if (min > paux)
	{
		min = paux;
	}
	
	if (min == 1)
	{
		sum = 0;
	}
	else
	{
		sum = this->inc[min-2];
	}
	//fim - 1
	
	//2
	if (paux < p)
	{
		ni = s->solution[paux-1];
		next = s->solution[p];
	    if (this->tsp->readytime[ni] > sum)
	    {
	    	sum = this->tsp->readytime[ni];
	    }
	    sum += this->tsp->matrix[ni][next];
	    if (sum > this->tsp->duedate[next])
	    {
	    	return false;
	    }
	    //
	    ni = s->solution[p];
		next = s->solution[paux];
	    if (this->tsp->readytime[ni] > sum)
	    {
	    	sum = this->tsp->readytime[ni];
	    }
	    sum += this->tsp->matrix[ni][next];
	    if (sum > this->tsp->duedate[next])
	    {
	    	return false;
	    }
	    //
	    for (int i = paux; i < p-1; i++)
        {
            ni = s->solution[i];
            next = s->solution[i+1];
            if (this->tsp->readytime[ni] > sum)
            {
            	sum = this->tsp->readytime[ni];
            }
            sum += this->tsp->matrix[ni][next];
            if (sum > this->tsp->duedate[next])
            {
            	return false;
            }
    	}
	    //
	    if (p + 1 < numNodes)
	    {
		    ni = s->solution[p-1];
			next = s->solution[p+1];
		    if (this->tsp->readytime[ni] > sum)
		    {
		    	sum = this->tsp->readytime[ni];
		    }
		    sum += this->tsp->matrix[ni][next];
		    if (sum > this->tsp->duedate[next])
		    {
		    	return false;
		    }
	    }
	    //
	    for (int i = p+1; i < numNodes-1; i++)
        {
            ni = s->solution[i];
            next = s->solution[i+1];
            if (this->tsp->readytime[ni] > sum)
            {
            	sum = this->tsp->readytime[ni];
            }
            sum += this->tsp->matrix[ni][next];
            if (sum > this->tsp->duedate[next])
            {
            	return false;
            }
            if (sum <= this->inc[i])
            {
            	break;
            }
    	}
	}
	else //p < paux
	{
		ni = s->solution[p-1];
		next = s->solution[p+1];
	    if (this->tsp->readytime[ni] > sum)
	    {
	    	sum = this->tsp->readytime[ni];
	    }
	    sum += this->tsp->matrix[ni][next];
	    if (sum > this->tsp->duedate[next])
	    {
	    	return false;
	    }

	    //
	    for (int i = p+1; i < paux; i++)
        {
            ni = s->solution[i];
            next = s->solution[i+1];
            if (this->tsp->readytime[ni] > sum)
            {
            	sum = this->tsp->readytime[ni];
            }
            sum += this->tsp->matrix[ni][next];
            if (sum > this->tsp->duedate[next])
            {
            	return false;
            }
    	}
	    
	    //
	    ni = s->solution[paux];
		next = s->solution[p];
	    if (this->tsp->readytime[ni] > sum)
	    {
	    	sum = this->tsp->readytime[ni];
	    }
	    sum += this->tsp->matrix[ni][next];
	    if (sum > this->tsp->duedate[next])
	    {
	    	return false;
	    }
	    
	    //
	    if (paux+1 < numNodes)
	    {
		    ni = s->solution[p];
			next = s->solution[paux+1];
		    if (this->tsp->readytime[ni] > sum)
		    {
		    	sum = this->tsp->readytime[ni];
		    }
		    sum += this->tsp->matrix[ni][next];
		    if (sum > this->tsp->duedate[next])
		    {
		    	return false;
		    }
	    }
	    
	    //
	    for (int i = paux+1; i < numNodes-1; i++)
        {
            ni = s->solution[i];
            next = s->solution[i+1];
            if (this->tsp->readytime[ni] > sum)
            {
            	sum = this->tsp->readytime[ni];
            }
            sum += this->tsp->matrix[ni][next];
            if (sum > this->tsp->duedate[next])
            {
            	return false;
            }
            if (sum <= this->inc[i])
            {
            	break;
            }
    	}
	}
    //fim - 2
	
	return true;
}
//------------------------------------------------------------------------------
void VNS::exchange(TSPTWSolution *s, int p, int paux)
{
	int n = s->solution[p];
	if (paux > p)
    {
		for (int i = p; i < paux; i++)
		{
			s->solution[i] = s->solution[i+1];
		}
    }
	else
	{
		for (int i = p; i > paux; i--)
		{
			s->solution[i] = s->solution[i-1];
		}
	}
	s->solution[paux] = n;
	
	//compInc
	int numNodes = this->tsp->numNodes;
	
	int min = p;
	int max = paux;
	if (paux < p)
	{
		min = paux;
		max = p;
	}
	
	int sum = 0;
	if (min > 1)
	{
		sum = this->inc[min-2];
	}
	int ni = 0;
	int next = 0;
	
    for (int i = min-1; i < max; i++)
    {
        ni = s->solution[i];
        next = s->solution[i+1];
        if (this->tsp->readytime[ni] > sum)
        {
        	sum = this->tsp->readytime[ni];
        }
        sum += this->tsp->matrix[ni][next];
        this->inc[i] = sum;
	}
    
    for (int i = max; i < numNodes-1; i++)
    {
        ni = s->solution[i];
        next = s->solution[i+1];
        if (this->tsp->readytime[ni] > sum)
        {
        	sum = this->tsp->readytime[ni];
        	break;
        }
        sum += this->tsp->matrix[ni][next];
        this->inc[i] = sum;
	}
    //
}
//------------------------------------------------------------------------------
void VNS::compInc(TSPTWSolution *s)
{
	int numNodes = this->tsp->numNodes;
	int sum = 0;
	int ni = 0;
	int next = 0;
    for (int i = 0; i < numNodes-1; i++)
    {
        ni = s->solution[i];
        next = s->solution[i+1];
        if (this->tsp->readytime[ni] > sum)
        {
        	sum = this->tsp->readytime[ni];
        }
        sum += this->tsp->matrix[ni][next];
        this->inc[i] = sum;
	}
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
