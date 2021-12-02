#include "localgenius.h"
//------------------------------------------------------------------------------
LocalGenius::LocalGenius()
{
}
//------------------------------------------------------------------------------
LocalGenius::~LocalGenius()
{
	
}
//------------------------------------------------------------------------------
void LocalGenius::process(TSPTWSolution *s)
{	
	this->tsp = s->tsp;
	
	int numNodes = tsp->numNodes;
	
	this->inc = new int[numNodes];
	this->compInc(s);
	
    int setSize = numNodes-1;
    int *set = new int[setSize];
    for (int i = 0; i < setSize; i++)
    {
        set[i] = i+1;
    }
    
    int pos = 0;
    int npos = 0;
    
    int r = 0;
    
    int aux = 0;
    
	while (setSize > 0)
    {
		unsigned long rd = genrand_int32(); 
		pos = rd%setSize;
        npos = set[pos];
        //cout << endl << rd << ";" << setSize << ";" << pos << ";" << npos << ";" << s->getPathDistance();
        
        r = this->movement(s, npos);
        
        if (r != -1)
        {
        	setSize = numNodes - 1;
        	/*
        	if (r < npos)
        	{
        		for (int i = 0; i < numNodes-1; i++)
        		{
        			if (set[i] >= r && set[i] < npos)
	        		{
	        			set[i] = set[i]+1;
	        		}
        			else if (set[i] == npos)
        			{
        				set[i] = r;
        			}
        		}
        	}
        	else if (npos < r)
        	{
        		for (int i = 0; i < numNodes-1; i++)
        		{
        			if (set[i] <= r && set[i] > npos)
	        		{
        				set[i] = set[i]-1;
	        		}
        			else if (set[i] == npos)
        			{
        				set[i] = r;
        			}
        		}
        	}
        	
        	int psSize = 10;
        	int *ps = new int[psSize];
        	ps[0] = npos;
        	ps[1] = npos-1;
        	ps[2] = npos+1;
        	ps[3] = r;
        	ps[4] = r-1;
        	ps[5] = r+1;
        	//
        	ps[6] = npos-2;
        	ps[7] = npos+2;
        	ps[8] = r-2;
        	ps[9] = r+2;
        	
        	for (int i = setSize; i < numNodes-1; i++)
        	{
	        	for (int j = 0; j < psSize; j++)
	    		{
	    			if (set[i] == ps[j])
	        		{
	        			int aux = set[setSize-1];
	        			set[setSize-1] = ps[j];
	        			set[i] = aux;
			            setSize++;
			            break;
	        		}
	    		}
        	}
        	
        	delete[] ps;
        	*/
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
int LocalGenius::movement(TSPTWSolution *s, int npos)
{
	int n = s->solution[npos];
	
	int numNodes = this->tsp->numNodes;
	
	int nprev = 0;
	if (npos == 0)
	{
		nprev = s->solution[numNodes-1];
	}
	else
	{
		nprev = s->solution[npos-1];
	}
	
	int nnext = 0;
	if (npos == numNodes-1)
	{
		nnext = s->solution[0];
	}
	else
	{
		nnext = s->solution[npos+1];
	}
	
	int d = this->tsp->matrix[nprev][n] +
			this->tsp->matrix[n][nnext] -
			this->tsp->matrix[nprev][nnext];
	
	
	/************/
	//down
	for (int i = npos-1; i > 0; i--)
	{	
		int ni = s->solution[i];
		if (!this->tsp->compatible[n][ni])
		{
			break;
		}
		int niprev = s->solution[i-1];
		int daux = this->tsp->matrix[niprev][n] +
					this->tsp->matrix[n][ni] -
					this->tsp->matrix[niprev][ni];
		if (daux < d)
		{
			if (this->isFeasible(s, npos, i))
			{
				this->exchange(s, npos, i);
				return i;
			}
		}
	}
	
	/************/
	//up:i=npos+1
	if (npos < numNodes-1)
	{
		int ni = s->solution[npos+1];
		
		if (this->tsp->compatible[n][ni])
		{
			int ninext = 0;
			if (npos+1 == numNodes-1)
			{
				ninext = s->solution[0];
			}
			else
			{
				ninext = s->solution[npos+2];
			}
			
			int daux = this->tsp->matrix[ni][n] +
						this->tsp->matrix[n][ninext] -
						this->tsp->matrix[ni][ninext];
			if (daux < d)
			{
				if (this->isFeasible(s, npos, npos+1))
				{
					this->exchange(s, npos, npos+1);
					return npos+1;
				}
			}
		}
	}
	
	//up
	for (int i = npos+2; i < numNodes; i++)
	{
		int ni = s->solution[i];
		if (!this->tsp->compatible[ni][n])
		{
			break;
		}

		int ninext = 0;
		if (i == numNodes-1)
		{
			ninext = s->solution[0];
		}
		else
		{
			ninext = s->solution[i+1];
		}
		
		int daux = this->tsp->matrix[ni][n] +
					this->tsp->matrix[n][ninext] -
					this->tsp->matrix[ni][ninext];
		if (daux < d)
		{
			if (this->isFeasible(s, npos, i))
			{
				this->exchange(s, npos, i);
				return i;
			}
		}
	}
	/**********/
	
    return -1;
}
//------------------------------------------------------------------------------
bool LocalGenius::isFeasible(TSPTWSolution *s, int p, int paux)
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
void LocalGenius::exchange(TSPTWSolution *s, int p, int paux)
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
void LocalGenius::compInc(TSPTWSolution *s)
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
