#ifndef TSPTWREADER_
#define TSPTWREADER_
//------------------------------------------------------------------------------
#include "includes.h"
#include "tsptw.h"
#include "tsptwpoint.h"
//------------------------------------------------------------------------------
class TSPTWReader
{
	private:
        char *fileName;
        
        TSPTWPoint *parse(string s);
		
	public:
		TSPTWReader(char *fileName);
		~TSPTWReader();
		
		TSPTW* process(bool round_distance);
};
//------------------------------------------------------------------------------   

#endif
