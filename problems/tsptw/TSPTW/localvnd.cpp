#include "localvnd.h"
//------------------------------------------------------------------------------
LocalVND::LocalVND()
{
	this->genius = new LocalGenius();
	this->local2opt = new Local2Opt();
}
//------------------------------------------------------------------------------
LocalVND::~LocalVND()
{
	delete this->genius;
	delete this->local2opt;
}
//------------------------------------------------------------------------------
void LocalVND::process(TSPTWSolution *s)
{	
	int best = s->getPathDistance();
	int bestaux;
	bool improvement = true;
	
	while (improvement)
	{
		improvement = false;
		
		this->genius->process(s);
		bestaux = s->getPathDistance();
		if (bestaux < best)
		{
			best = bestaux;
			//improvement = true;
		}
		
		this->local2opt->process(s);
		bestaux = s->getPathDistance();
		if (bestaux < best)
		{
			best = bestaux;
			improvement = true;
		}
	}
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
