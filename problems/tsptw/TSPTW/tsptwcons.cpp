#include "tsptwcons.h"
//------------------------------------------------------------------------------
TSPTWCons::TSPTWCons(TSPTW *tsp)
{
    this->tsp = tsp;
    
    int numNodes = this->tsp->numNodes;
    
    this->inc = new int[numNodes];
    this->pen = new int[numNodes];
    
    this->feasible = new int[this->tsp->numNodes];
    this->feasibleSize = 0;
    
    this->unfeasible = new int[this->tsp->numNodes];
    this->unfeasibleSize = 0;
    
    this->lastExchangeCons = 0;
}
//------------------------------------------------------------------------------
TSPTWCons::~TSPTWCons()
{
	delete[] this->inc;
	delete[] this->pen;
	delete[] this->feasible;
	delete[] this->unfeasible;
}
//------------------------------------------------------------------------------
TSPTWSolution *TSPTWCons::process()
{	
	int numNodes = this->tsp->numNodes;
	
	this->feasibleSize = 0;
	this->unfeasibleSize = 0;
	this->lastExchangeCons = 0;
	
	TSPTWSolution *current = random();
	this->local(current);
	
	int penalty = this->penalty(current);
		
	if (penalty == 0)
	{
		return current;
	}
	
    TSPTWSolution *solution = current->clone();
    
    int penaltyAux = 0;

    int level = 1;
    int levelMax = numNodes/2;
    
    int p1 = 0;
    int p2 = 0;

    while (penalty > 0)
    {
    	for (int i = 0; i < level; i++)
    	{
    		p1 = (genrand_int32()%(numNodes-1))+1;
    		p2 = (genrand_int32()%(numNodes-1))+1;

    		this->exchangeCons(solution, p1, p2);
    	}
	    this->local(solution);
	    penaltyAux = this->penalty(solution);
	    
	    if (penaltyAux < penalty)
	    {
	    	if (penaltyAux == 0)
	    	{
	    		delete current;
	    		return solution;
	    	}
	    	penalty = penaltyAux;
	    	//clone
	    	for (int i = 1; i < numNodes; i++)
	    	{
	    		current->solution[i] = solution->solution[i];
	    	}
	    	//
	    	level = 1;
	    }
	    else
	    {
	    	level++;
	    	if (level < levelMax)
		    {
		    	for (int i = 1; i < numNodes; i++)
		    	{
		    		solution->solution[i] = current->solution[i];
		    	}
		    }
	    	else
	    	{
	    		delete current;
		    	current = random();
		    	this->local(current);
		    	penalty = this->penalty(current);
		    	if (penalty == 0)
		    	{
		    		delete solution;
		    		return current;
		    	}
		    	level = 1;
		    	//clone
		    	for (int i = 1; i < numNodes; i++)
		    	{
		    		solution->solution[i] = current->solution[i];
		    	}
		    	//
		    }
	    }
    }
}
//------------------------------------------------------------------------------
TSPTWSolution *TSPTWCons::random()
{
	this->lastExchangeCons = 0;
	
	int numNodes = this->tsp->numNodes;
	TSPTWSolution *solution = new TSPTWSolution(this->tsp);
	
    for (int i = 0; i < numNodes; i++)
    {
    	solution->solution[i] = i;
    }
    
    int n = numNodes/2;
    int p1 = 0;
    int p2 = 0;
    int aux = 0;
    for (int i = 0; i < n; i++)
    {
    	p1 = (genrand_int32()%(numNodes-1))+1;
    	p2 = (genrand_int32()%(numNodes-1))+1;
    	aux = solution->solution[p1];
    	solution->solution[p1] = solution->solution[p2];
    	solution->solution[p2] = aux;
    }

	return solution;
}
//------------------------------------------------------------------------------
void TSPTWCons::local(TSPTWSolution *solution)
{
	int numNodes = this->tsp->numNodes;	

	int penalty = this->penalty(solution);
    this->calcSets(solution);
    bool c = true;
    
    while (penalty > 0 && (this->unfeasibleSize > 0 || this->feasibleSize > 0))
    {
	    c = true;
	    
	    int ufAuxSize = this->unfeasibleSize;
	    int fAuxSize = this->feasibleSize;
	    
	    while (c && this->unfeasibleSize > 0)
	    {
	    	int pos = genrand_int32()%(this->unfeasibleSize);
	    	int npos = this->unfeasible[pos];
	    	int n = solution->solution[npos];
	
	        bool r = false;
	        for (int i = npos-1; i > 0; i--)
	        {
	        	if (!this->tsp->compatible[n][solution->solution[i]])
				{
	    			break;
				}
	    		this->exchangeCons(solution, npos, i);
	    		int penaltyAux = this->penalty(solution);
	    		if (penaltyAux < penalty)
	    		{
	    			penalty = penaltyAux;
	    			r = true;
	    			break;
	    		}
	    		else
	    		{
	    			this->exchangeCons(solution, i, npos);
	    		}
	        }
	        	
	        if (r)
	        {
	        	if (penalty == 0)
	        	{
	        		c = false;
	        		break;
	        	}
	        	this->calcSets(solution);
	        }
	        else
	        {
	        	this->unfeasibleSize--;
	        	int aux = this->unfeasible[pos];
	        	this->unfeasible[pos] = this->unfeasible[this->unfeasibleSize];
	        	this->unfeasible[this->unfeasibleSize] = aux;
	        }
	    }
	    
	    while (c && this->feasibleSize > 0)
	    {
	    	int pos = genrand_int32()%(this->feasibleSize);
	        int npos = this->feasible[pos];
	        int n = solution->solution[npos];
	
	        bool r = false;
	        
        	for (int i = npos+1; i < numNodes-1; i++)
            {
        		if (!this->tsp->compatible[solution->solution[i]][n])
    			{
        			break;
    			}
        		this->exchangeCons(solution, npos, i);
        		int penaltyAux = this->penalty(solution);
        		if (penaltyAux < penalty)
        		{
        			penalty = penaltyAux;
        			r = true;
        			break;
        		}
        		else
        		{
        			this->exchangeCons(solution, i, npos);
        		}
            }
	        	
	        if (r)
	        {
	        	if (penalty == 0)
	        	{
	        		c = false;
	        		break;
	        	}
	        	this->calcSets(solution);
	        }
	        else
	        {
	        	this->feasibleSize--;
	        	int aux = this->feasible[pos];
	        	this->feasible[pos] = this->feasible[this->feasibleSize];
	        	this->feasible[this->feasibleSize] = aux;
	        }
	    }
	    
	    this->unfeasibleSize = ufAuxSize;
	    this->feasibleSize = fAuxSize;
	    
	    while (c && this->unfeasibleSize > 0)
	    {
	    	int pos = genrand_int32()%(this->unfeasibleSize);
	        int npos = this->unfeasible[pos];
	        int n = solution->solution[npos];
	
	        bool r = false;
	        for (int i = npos+1; i < numNodes-1; i++)
            {
	        	if (!this->tsp->compatible[solution->solution[i]][n])
    			{
        			break;
    			}
        		this->exchangeCons(solution, npos, i);
        		int penaltyAux = this->penalty(solution);
        		if (penaltyAux < penalty)
        		{
        			penalty = penaltyAux;
        			r = true;
        			break;
        		}
        		else
        		{
        			this->exchangeCons(solution, i, npos);
        		}
            }
	        
	        if (r)
	        {
	        	if (penalty == 0)
	        	{
	        		c = false;
	        		break;
	        	}
	        	this->calcSets(solution);
	        }
	        else
	        {
	        	this->unfeasibleSize--;
	        	int aux = this->unfeasible[pos];
	        	this->unfeasible[pos] = this->unfeasible[this->unfeasibleSize];
	        	this->unfeasible[this-this>unfeasibleSize] = aux;
	        }
	    }
	    
	    while (c && this->feasibleSize > 0)
	    {
	    	int pos = genrand_int32()%(this->feasibleSize);
	        int npos = this->feasible[pos];
	        int n = solution->solution[npos];
	
	        bool r = false;
	        for (int i = npos-1; i > 0; i--)
	        {
	        	if (!this->tsp->compatible[n][solution->solution[i]])
				{
	    			break;
				}
	    		this->exchangeCons(solution, npos, i);
	    		int penaltyAux = this->penalty(solution);
	    		if (penaltyAux < penalty)
	    		{
	    			penalty = penaltyAux;
	    			r = true;
	    			break;
	    		}
	    		else
	    		{
	    			this->exchangeCons(solution, i, npos);
	    		}
	        }
	        	
	        if (r)
	        {
	        	if (penalty == 0)
	        	{
	        		c = false;
	        		break;
	        	}
	        	this->calcSets(solution);
	        }
	        else
	        {
	        	this->feasibleSize--;
	        	int aux = this->feasible[pos];
	        	this->feasible[pos] = this->feasible[this->feasibleSize];
	        	this->feasible[this->feasibleSize] = aux;
	        }
	    }
    }
}
//------------------------------------------------------------------------------
int TSPTWCons::penalty(TSPTWSolution *s)
{
    int sum = 0;
    int psum = 0;
    int psumaux = 0;
    if (this->lastExchangeCons > 1)
    {
    	sum = this->inc[this->lastExchangeCons];
    	psum = this->pen[this->lastExchangeCons];
    }
    else
    {
    	this->lastExchangeCons = 0;
    	this->inc[0] = 0;
    	this->pen[0] = 0;
    }
    int numNodes = this->tsp->numNodes;
    for (int i = this->lastExchangeCons; i < numNodes-1; i++)
    {
        int ni = s->solution[i];
        int next = s->solution[i+1];
        int d = this->tsp->matrix[ni][next];
        if (this->tsp->readytime[ni] > sum)
        {
        	sum = this->tsp->readytime[ni];
        }
        sum = sum + d;
        this->inc[i+1] = sum;
        psumaux = sum - this->tsp->duedate[next];
        if (psumaux < 0)
        {
        	psumaux = 0;
        }
        psum += psumaux;
        this->pen[i+1] = psum;
    }
    this->lastExchangeCons = numNodes-1;
    return psum;
}
//------------------------------------------------------------------------------
void TSPTWCons::exchangeCons(TSPTWSolution *s, int p, int paux)
{
	int n = s->solution[p];
	if (paux > p)
    {
		if (p-1 < this->lastExchangeCons)
		{
			this->lastExchangeCons = p-1;
		}
		for (int i = p; i < paux; i++)
		{
			s->solution[i] = s->solution[i+1];
		}
    }
	else
	{
		if (paux-1 < this->lastExchangeCons)
		{
			this->lastExchangeCons = paux-1;
		}
		for (int i = p; i > paux; i--)
		{
			s->solution[i] = s->solution[i-1];
		}
	}
	s->solution[paux] = n;
}
//------------------------------------------------------------------------------
void TSPTWCons::calcSets(TSPTWSolution *s)
{
	this->feasibleSize = 0;
	this->unfeasibleSize = 0;
	int numNodes = this->tsp->numNodes;
	int sum = 0;
    for (int i = 0; i < numNodes-1; i++)
    {
        int ni = s->solution[i];
        int next = s->solution[i+1];
        int d = this->tsp->matrix[ni][next];
        if (this->tsp->readytime[ni] > sum)
        {
        	sum = this->tsp->readytime[ni];
        }
        sum = sum + d;
        if (sum - this->tsp->duedate[next] > 0)
        {
        	this->unfeasible[this->unfeasibleSize] = i+1;
        	this->unfeasibleSize++;
        }
        else
        {
        	this->feasible[this->feasibleSize] = i+1;
        	this->feasibleSize++;
        }
	}
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
