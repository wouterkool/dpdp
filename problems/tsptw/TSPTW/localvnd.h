#ifndef LOCALVND_
#define LOCALVND_
//------------------------------------------------------------------------------
#include "includes.h"
#include "localsearch.h"
#include "localgenius.h"
#include "local2opt.h"
//------------------------------------------------------------------------------
class LocalVND: public LocalSearch
{		
	private:
		LocalGenius *genius;
		Local2Opt *local2opt;
		
	public:
		LocalVND();
		virtual ~LocalVND();

		virtual void process(TSPTWSolution *solution);
};
//------------------------------------------------------------------------------   

#endif
