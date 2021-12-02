#ifndef TIMER_
#define TIMER_
//------------------------------------------------------------------------------
class Timer
{		
	public:

		virtual void start() = 0;
		virtual void stop() = 0;
		virtual double getTime() = 0;
};
//------------------------------------------------------------------------------  

#endif
