#include "tsptwpoint.h"
//------------------------------------------------------------------------------
TSPTWPoint::TSPTWPoint(float px, float py, float ready, float due, float service)
{
    this->x = px;
    this->y = py;
    this->ready = ready;
    this->due = due;
    this->service = service;
}
//------------------------------------------------------------------------------
TSPTWPoint::~TSPTWPoint()
{
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
