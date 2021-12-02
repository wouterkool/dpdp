#include "windowstimer.h"
//------------------------------------------------------------------------------
WindowsTimer::WindowsTimer()
{
	
}
//------------------------------------------------------------------------------
WindowsTimer::~WindowsTimer()
{
	
}
//------------------------------------------------------------------------------
void WindowsTimer::start()
{
	QueryPerformanceCounter(&this->time1);
}
//------------------------------------------------------------------------------
void WindowsTimer::stop()
{
	QueryPerformanceCounter(&this->time2);
}
//------------------------------------------------------------------------------
double WindowsTimer::getTime()
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	long long int diffTicks = this->time2.QuadPart - this->time1.QuadPart;
	return (double)((double)diffTicks/(double)(freq.QuadPart));
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
