#ifndef DEFAULTTIMER_
#define DEFAULTTIMER_

#ifdef WIN32
#include "windowstimer.h"
class DefaultTimer
{
	public:
		
		static Timer *getDefault()
		{
			return new WindowsTimer();
		}
};
#else
#include "cputimer.h"
class DefaultTimer
{
	public:
		static Timer *getDefault()
		{
			return new CPUTimer();
		}
};
#endif // WIN32

#endif

