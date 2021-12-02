#include "local2opt.h"
//------------------------------------------------------------------------------
Local2Opt::Local2Opt()
{
	
}
//------------------------------------------------------------------------------
Local2Opt::~Local2Opt()
{
	
}
//------------------------------------------------------------------------------
void Local2Opt::process(TSPTWSolution *s)
{
	this->tsp = s->tsp;
	int numNodes = this->tsp->numNodes;
	
	this->inc = new int[numNodes];
	this->compInc(s);
	
	bool improvement = false;
	
	int setSize = numNodes-2;
    int *set = new int[setSize];
    for (int i = 0; i < setSize; i++)
    {
        set[i] = i+1;
    }
    
    int pos = 0;
    int npos = 0;
    
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;
    int n4 = 0;
    
    int e1 = 0;
    int e2 = 0;
    int e3 = 0;
    int e4 = 0;
    
    int s1 = 0;
    int s2 = 0;
    
    int isf = 0;
    
    int aux = 0;
	
    while (setSize > 0)
    {
    	improvement = false;
    	
    	pos = genrand_int32()%setSize;
    	npos = set[pos];
    	
    	n1 = s->solution[npos];
        n2 = s->solution[npos+1];
        e1 = this->tsp->matrix[n1][n2];
        
        for (int j = npos+2; j < numNodes; j++)
        {
        	n3 = s->solution[j];
        	
        	if (!this->tsp->compatible[n3][n2])
        	{
        		break;
        	}
        	
            n4 = 0;
            
            if (j < numNodes-1)
            {
                n4 = s->solution[j+1];
            }
            
            e2 = this->tsp->matrix[n3][n4];
            s1 = e1 + e2;
            
            e4 = this->tsp->matrix[n2][n4];
            e3 = this->tsp->matrix[n1][n3];

            s2 = e3 + e4;
            if (s1 > s2)
            {
            	isf = this->isFeasible(s, npos, j);
            	if (isf == 0)
	            {
            		this->exchange(s, npos, j);
	                improvement = true;
	                break;
	            }
            	else if (isf == -2)
            	{
            		break;
            	}
            }
        }
        if (improvement)
        {
        	setSize = numNodes-2;
        }
        else
        {
        	aux = set[setSize-1];
            set[setSize-1] = npos;
            set[pos] = aux;
            setSize--;
        }
    }
    delete[] set;
    delete[] this->inc;
}
//------------------------------------------------------------------------------
void Local2Opt::exchange(TSPTWSolution *s, int iaux, int jaux)
{
	int start = iaux+1;
	int end = jaux;
	
	int aux = 0;
	int j = 0;
	int f = (end-start+1)/2+start-1;
	for (int i = start; i <= f; i++)
    {
		j = end-(i-start);
		aux = s->solution[i];
		s->solution[i] = s->solution[j];
		s->solution[j] = aux;
    }
    
    //compInc
	int numNodes = this->tsp->numNodes;
	
	int min = iaux;
	int max = jaux;
	
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
    
    /*
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
	*/
    
    for (int i = max; i < numNodes-1; i++)
    {
        ni = s->solution[i];
        next = s->solution[i+1];
        if (this->tsp->readytime[ni] > sum)
        {
        	sum = this->tsp->readytime[ni];
        }
        sum += this->tsp->matrix[ni][next];
        if (this->inc[i] == sum)
        {
        	break;
        }
        this->inc[i] = sum;
	}
    //
}
//------------------------------------------------------------------------------
int Local2Opt::isFeasible(TSPTWSolution *s, int iaux, int jaux)
{
	int numNodes = this->tsp->numNodes;
	
	int sum = 0;
	int ni = 0;
	int next = 0;
	
	//1
	int min = iaux;
	sum = this->inc[min-1];
	//fim - 1
	
	ni = s->solution[iaux];
    next = s->solution[jaux];
    if (this->tsp->readytime[ni] > sum)
    {
    	sum = this->tsp->readytime[ni];
    }
    sum += this->tsp->matrix[ni][next];
    if (sum > this->tsp->duedate[next])
    {
    	return -1;
    }
	for (int i = jaux; i > iaux+1; i--)
    {
        ni = s->solution[i];
        next = s->solution[i-1];
        if (this->tsp->readytime[ni] > sum)
        {
        	sum = this->tsp->readytime[ni];
        }
        sum += this->tsp->matrix[ni][next];
        if (sum > this->tsp->duedate[next])
        {
        	return -2;
        }
	}
	if (jaux+1 < numNodes)
	{
		ni = s->solution[iaux+1];
	    next = s->solution[jaux+1];
	    if (this->tsp->readytime[ni] > sum)
	    {
	    	sum = this->tsp->readytime[ni];
	    }
	    sum += this->tsp->matrix[ni][next];
	    if (sum > this->tsp->duedate[next])
	    {
	    	return -3;
	    }
	}
    for (int i = jaux+1; i < numNodes-1; i++)
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
        	return -4;
        }
        if (sum <= this->inc[i])
        {
        	break;
        }
	}
    return 0;
}
//------------------------------------------------------------------------------
void Local2Opt::compInc(TSPTWSolution *s)
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
