#include "tsptwreader.h"
//------------------------------------------------------------------------------
TSPTWReader::TSPTWReader(char *fileName)
{
    this->fileName = fileName;
}
//------------------------------------------------------------------------------
TSPTWReader::~TSPTWReader()
{
}
//------------------------------------------------------------------------------
TSPTW* TSPTWReader::process(bool round_distance)
{
    FILE *file;
    file = fopen(this->fileName, "r");
    if (!file)
    {
        return NULL;
    }
    vector<TSPTWPoint*> *points = new vector<TSPTWPoint*>();
    char line[1000];
    while (fgets(line, 1000, file))
	{
      	string s = line;
      	TSPTWPoint *tspPoint = this->parse(s);
      	if (tspPoint != NULL)
      	{
            points->push_back(tspPoint);
        }
	}
    fclose(file);
    
    TSPTW *tsp = new TSPTW(points->size());
    
    int pSize = points->size();
    for (int i = 0; i < pSize; i++)
    {
        TSPTWPoint *tspPoint1 = points->at(i);
        tsp->readytime[i] = (int)tspPoint1->ready;
        tsp->duedate[i] = (int)tspPoint1->due;
        tsp->setDistance(i, i, 0);
        for (int j = i+1; j < pSize; j++)
        {
            TSPTWPoint *tspPoint2 = points->at(j);
            double px = pow(tspPoint1->x - tspPoint2->x, 2);
            double py = pow(tspPoint1->y - tspPoint2->y, 2);       
            int dist = (int)floor(sqrt(px + py) + (round_distance ? 0.5 : 0.));
            tsp->setDistance(i, j, dist + (int)tspPoint1->service);
            tsp->setDistance(j, i, dist + (int)tspPoint2->service);
        }
        delete tspPoint1;
    }
    delete points;
    return tsp;
}
//------------------------------------------------------------------------------
TSPTWPoint* TSPTWReader::parse(string s)
{
    int saux = s.length();
    float pointNum = 0;
    float pointX = 0;
    float pointY = 0;
    float demand = 0;
    float ready = 0;
    float due = 0;
    float service = 0;
    
    //pointNum
    int i = 0;
    while (i < saux && s[i] == ' ')
    {
        i++;
    }
    if (s[i] < '0' || s[i] > '9')
    {
        return NULL;
    }
    bool numP = false;
    int numDecPlaces = 0;
    while (i < saux && ((s[i] >= '0' && s[i] <= '9') || (s[i] == '.')))
    {
        if (s[i] == '.')
        {
            numP = true;
        }
        else
        {
            pointNum = 10 * pointNum + (s[i] - '0');
            if (numP)
            {
                numDecPlaces++;
            }
            i++;
        }
    }
    if (numP)
    {
        pointNum = pointNum/(pow(10, numDecPlaces));
    }
    
    //pointX
    while (i < saux && s[i] == ' ')
    {
        i++;
    }
    if ((s[i] < '0' || s[i] > '9') && s[i] != '-')
    {
        return NULL;
    }
    numP = false;
    bool sign = false;
    numDecPlaces = 0;
    while (i < saux && ((s[i] >= '0' && s[i] <= '9') || (s[i] == '.') || s[i] == '-'))
    {
        if (s[i] == '-')
        {
            sign = true;
        }
        else if (s[i] == '.')
        {
            numP = true;
        }
        else
        {
            pointX = 10 * pointX + (s[i] - '0');
            if (numP)
            {
                numDecPlaces++;
            }
        }
        i++;
    }
    if (numP)
    {
        pointX = pointX/(pow(10, numDecPlaces));
    }
    if (sign)
    {
        pointX = -1 * pointX;
    }
    
    //pointY
    while (i < saux && s[i] == ' ')
    {
        i++;
    }
    if ((s[i] < '0' || s[i] > '9') && s[i] != '-')
    {
        return NULL;
    }
    numP = false;
    numDecPlaces = 0;
    sign = false;
    while (i < saux && ((s[i] >= '0' && s[i] <= '9') || (s[i] == '.') || s[i] == '-'))
    {
        if (s[i] == '-')
        {
            sign = true;
        }
        else if (s[i] == '.')
        {
            numP = true;
        }
        else
        {
            pointY = 10 * pointY + (s[i] - '0');
            if (numP)
            {
                numDecPlaces++;
            }
        }
        i++;
    }
    if (numP)
    {
        pointY = pointY/(pow(10, numDecPlaces));
    }
    if (sign)
    {
        pointY = -1 * pointY;
    }
    
    //demand
    while (i < saux && s[i] == ' ')
    {
        i++;
    }
    if ((s[i] < '0' || s[i] > '9') && s[i] != '-')
    {
        return NULL;
    }
    numP = false;
    numDecPlaces = 0;
    sign = false;
    while (i < saux && ((s[i] >= '0' && s[i] <= '9') || (s[i] == '.') || s[i] == '-'))
    {
        if (s[i] == '-')
        {
            sign = true;
        }
        else if (s[i] == '.')
        {
            numP = true;
        }
        else
        {
            demand = 10 * demand + (s[i] - '0');
            if (numP)
            {
                numDecPlaces++;
            }
        }
        i++;
    }
    if (numP)
    {
        demand = demand/(pow(10, numDecPlaces));
    }
    if (sign)
    {
        demand = -1 * demand;
    }
    
    //ready
    while (i < saux && s[i] == ' ')
    {
        i++;
    }
    if ((s[i] < '0' || s[i] > '9') && s[i] != '-')
    {
        return NULL;
    }
    numP = false;
    numDecPlaces = 0;
    sign = false;
    while (i < saux && ((s[i] >= '0' && s[i] <= '9') || (s[i] == '.') || s[i] == '-'))
    {
        if (s[i] == '-')
        {
            sign = true;
        }
        else if (s[i] == '.')
        {
            numP = true;
        }
        else
        {
            ready = 10 * ready + (s[i] - '0');
            if (numP)
            {
                numDecPlaces++;
            }
        }
        i++;
    }
    if (numP)
    {
        ready = ready/(pow(10, numDecPlaces));
    }
    if (sign)
    {
        ready = -1 * ready;
    }
    
    //due
    while (i < saux && s[i] == ' ')
    {
        i++;
    }
    if ((s[i] < '0' || s[i] > '9') && s[i] != '-')
    {
        return NULL;
    }
    numP = false;
    numDecPlaces = 0;
    sign = false;
    while (i < saux && ((s[i] >= '0' && s[i] <= '9') || (s[i] == '.') || s[i] == '-'))
    {
        if (s[i] == '-')
        {
            sign = true;
        }
        else if (s[i] == '.')
        {
            numP = true;
        }
        else
        {
            due = 10 * due + (s[i] - '0');
            if (numP)
            {
                numDecPlaces++;
            }
        }
        i++;
    }
    if (numP)
    {
        due = due/(pow(10, numDecPlaces));
    }
    if (sign)
    {
        due = -1 * due;
    }
    
    //service
    while (i < saux && s[i] == ' ')
    {
        i++;
    }
    if ((s[i] < '0' || s[i] > '9') && s[i] != '-')
    {
        return NULL;
    }
    numP = false;
    numDecPlaces = 0;
    sign = false;
    while (i < saux && ((s[i] >= '0' && s[i] <= '9') || (s[i] == '.') || s[i] == '-'))
    {
        if (s[i] == '-')
        {
            sign = true;
        }
        else if (s[i] == '.')
        {
            numP = true;
        }
        else
        {
            service = 10 * service + (s[i] - '0');
            if (numP)
            {
                numDecPlaces++;
            }
        }
        i++;
    }
    if (numP)
    {
        service = service/(pow(10, numDecPlaces));
    }
    if (sign)
    {
        service = -1 * service;
    }
    
    if (pointNum != 999) //Only for Dumas
    {
        return new TSPTWPoint(pointX, pointY, ready, due, service);    
    }
    
    return NULL;    
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

