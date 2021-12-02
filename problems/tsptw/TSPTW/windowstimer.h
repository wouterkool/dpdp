#ifndef WINDOWSTIMER_
#define WINDOWSTIMER_
//------------------------------------------------------------------------------
#include "timer.h"
#include <windows.h>
//------------------------------------------------------------------------------
class WindowsTimer: public Timer
{
	private:
		LARGE_INTEGER time1;
		LARGE_INTEGER time2;
		
	public:
		WindowsTimer();
		virtual ~WindowsTimer();
		
		virtual void start();
		virtual void stop();
		virtual double getTime();
};
//------------------------------------------------------------------------------   

#endif
