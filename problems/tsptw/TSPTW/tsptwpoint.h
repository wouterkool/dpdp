#ifndef TSPTWPOINT_
#define TSPTWPOINT_
//------------------------------------------------------------------------------
#include "includes.h"
//------------------------------------------------------------------------------
class TSPTWPoint
{
	
	public:
		TSPTWPoint(float px, float py, float ready, float due, float service);
		~TSPTWPoint();
		
		float x;
        float y;
        float ready;
        float due;
        float service;
};
//------------------------------------------------------------------------------   

#endif
