//------------------------------------------------------------------------------
#include "includes.h"
#include "defaulttimer.h"
#include "gvns.h"
//------------------------------------------------------------------------------
#include "tsptwreader.h"
//------------------------------------------------------------------------------
#include "tsptw.h"
#include "tsptwsolution.h"
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
int gvns(char* instance, int max)
{
	TSPTWReader *r = new TSPTWReader(instance);
    TSPTW *tsp = r->process(false);
    tsp->triangleInequality();
    tsp->processCompatible();
    
    GVNS *gnvs = new GVNS(tsp, max);
    TSPTWSolution *best = gnvs->optimize();
    
    int bestfo = best->getPathDistance();
    best->print();
    
    delete gnvs;
    delete best;
    delete r;
    delete tsp;
    
    return bestfo;
}

int cons(char* instance)
{
    TSPTWReader *r = new TSPTWReader(instance);
    TSPTW *tsp = r->process(false);
    tsp->triangleInequality();
    tsp->processCompatible();
    
    TSPTWCons *cons = new TSPTWCons(tsp);
    TSPTWSolution *best = cons->process();
    
    int bestfo = best->getPathDistance();
    
    delete cons;
    delete best;
    delete r;
    delete tsp;
    
    return bestfo;
}

void readins(char* instance)
{
    TSPTWReader *r = new TSPTWReader(instance);
    TSPTW *tsp = r->process(false);
    
    tsp->triangleInequality();
    tsp->processCompatible();
    
    delete r;
    delete tsp;
}

int main(int argc, char *argv[])
{
    int seed = std::stoi(argv[1]);
    cout << "seed=" << seed << endl;
    int iter = std::stoi(argv[2]);
    cout << "iter=" << iter << endl; // Default in code was 30
    char* instance = argv[3];
	cout << "file=" << instance << endl;

	init_genrand(seed);


	Timer *timer = DefaultTimer::getDefault();
    timer->start();


    TSPTWReader *r = new TSPTWReader(instance);
    // Set true to use rounded (rather than floor) distance and DON'T apply triangle inequality
    TSPTW *tsp = r->process(true);
//    tsp->triangleInequality();
    tsp->processCompatible();

    GVNS *gnvs = new GVNS(tsp, iter);
    TSPTWSolution *best = gnvs->optimize();

    int result = best->getPathDistance();
    cout << "tour=" << endl;
    best->print();

    int time = timer->getTime();

    cout << "result=" << result << endl;
    cout << "time=" << time << endl;


}



int main_old(int argc, char *argv[])
{
	int seed = time(0);
	cout << "seed=" << seed << endl;
	
	init_genrand(seed);
	
	vector<char*> *vsn = new vector<char*>();
	
	/*****************
	//Dumas	 */
	vsn->push_back("ins/n20w20");
	vsn->push_back("ins/n20w40");
	vsn->push_back("ins/n20w60");
	vsn->push_back("ins/n20w80");
	//vsn->push_back("ins/n20w100");
	
	vsn->push_back("ins/n40w20");
	vsn->push_back("ins/n40w40");
	vsn->push_back("ins/n40w60");
	vsn->push_back("ins/n40w80");
	//vsn->push_back("ins/n40w100");
	
	vsn->push_back("ins/n60w20");
	vsn->push_back("ins/n60w40");
	vsn->push_back("ins/n60w60");
	vsn->push_back("ins/n60w80");
	//vsn->push_back("ins/n60w100");
	
	vsn->push_back("ins/n80w20");
	vsn->push_back("ins/n80w40");
	vsn->push_back("ins/n80w60");
	vsn->push_back("ins/n80w80");
	
	vsn->push_back("ins/n100w20");
	vsn->push_back("ins/n100w40");
	vsn->push_back("ins/n100w60");
	
	vsn->push_back("ins/n150w20");
	vsn->push_back("ins/n150w40");
	vsn->push_back("ins/n150w60");
	
	vsn->push_back("ins/n200w20");
	vsn->push_back("ins/n200w40");
	
	/*****************
	//Gendreau
	vsn->push_back("ins/n20w120");
	vsn->push_back("ins/n20w140");
	vsn->push_back("ins/n20w160");
	vsn->push_back("ins/n20w180");
	vsn->push_back("ins/n20w200");

	vsn->push_back("ins/n40w120");
	vsn->push_back("ins/n40w140");
	vsn->push_back("ins/n40w160");
	vsn->push_back("ins/n40w180");
	vsn->push_back("ins/n40w200");

	vsn->push_back("ins/n60w120");
	vsn->push_back("ins/n60w140");
	vsn->push_back("ins/n60w160");
	vsn->push_back("ins/n60w180");
	vsn->push_back("ins/n60w200");

	vsn->push_back("ins/n80w100");
	vsn->push_back("ins/n80w120");
	vsn->push_back("ins/n80w140");
	vsn->push_back("ins/n80w160");
	vsn->push_back("ins/n80w180");
	vsn->push_back("ins/n80w200");

	vsn->push_back("ins/n100w80");
	vsn->push_back("ins/n100w100");
	vsn->push_back("ins/n100w120");
	vsn->push_back("ins/n100w140");
	vsn->push_back("ins/n100w160");
	vsn->push_back("ins/n100w180");
	vsn->push_back("ins/n100w200");
	
	/*****************
	//Ohlmann
	vsn->push_back("ins/n150w120");
	vsn->push_back("ins/n150w140");
	vsn->push_back("ins/n150w160");
	
	vsn->push_back("ins/n200w120");
	vsn->push_back("ins/n200w140");
	/*****************/
	
	/*******************
	//VNS-VND
	vsn->push_back("ins2/n100w100");
	vsn->push_back("ins2/n100w200");
	vsn->push_back("ins2/n100w300");
	vsn->push_back("ins2/n100w400");
	vsn->push_back("ins2/n100w500");
	
	vsn->push_back("ins2/n150w100");
	vsn->push_back("ins2/n150w200");
	vsn->push_back("ins2/n150w300");
	vsn->push_back("ins2/n150w400");
	vsn->push_back("ins2/n150w500");
	
	vsn->push_back("ins2/n200w100");
	vsn->push_back("ins2/n200w200");
	vsn->push_back("ins2/n200w300");
	vsn->push_back("ins2/n200w400");
	vsn->push_back("ins2/n200w500");
	
	vsn->push_back("ins2/n250w100");
	vsn->push_back("ins2/n250w200");
	vsn->push_back("ins2/n250w300");
	vsn->push_back("ins2/n250w400");
	vsn->push_back("ins2/n250w500");
	
	vsn->push_back("ins2/n300w100");
	vsn->push_back("ins2/n300w200");
	vsn->push_back("ins2/n300w300");
	vsn->push_back("ins2/n300w400");
	vsn->push_back("ins2/n300w500");
	
	vsn->push_back("ins2/n350w100");
	vsn->push_back("ins2/n350w200");
	vsn->push_back("ins2/n350w300");
	vsn->push_back("ins2/n350w400");
	vsn->push_back("ins2/n350w500");
	
	vsn->push_back("ins2/n400w100");
	vsn->push_back("ins2/n400w200");
	vsn->push_back("ins2/n400w300");
	vsn->push_back("ins2/n400w400");
	vsn->push_back("ins2/n400w500");
	
	vsn->push_back("ins2/n450w100");
	vsn->push_back("ins2/n450w200");
	vsn->push_back("ins2/n450w300");
	vsn->push_back("ins2/n450w400");
	vsn->push_back("ins2/n450w500");
		
	vsn->push_back("ins2/n500w100");
	vsn->push_back("ins2/n500w200");
	vsn->push_back("ins2/n500w300");
	vsn->push_back("ins2/n500w400");
	vsn->push_back("ins2/n500w500");
	/*******************/
	
	/*******************
	//VNS-VND Extended
	vsn->push_back("ins2/n550w100");
	vsn->push_back("ins2/n550w200");
	vsn->push_back("ins2/n550w300");
	vsn->push_back("ins2/n550w400");
	vsn->push_back("ins2/n550w500");
	
	vsn->push_back("ins2/n600w100");
	vsn->push_back("ins2/n600w200");
	vsn->push_back("ins2/n600w300");
	vsn->push_back("ins2/n600w400");
	vsn->push_back("ins2/n600w500");
	
	vsn->push_back("ins2/n650w100");
	vsn->push_back("ins2/n650w200");
	vsn->push_back("ins2/n650w300");
	vsn->push_back("ins2/n650w400");
	vsn->push_back("ins2/n650w500");
	
	vsn->push_back("ins2/n700w100");
	vsn->push_back("ins2/n700w200");
	vsn->push_back("ins2/n700w300");
	vsn->push_back("ins2/n700w400");
	vsn->push_back("ins2/n700w500");
	
	vsn->push_back("ins2/n750w100");
	vsn->push_back("ins2/n750w200");
	vsn->push_back("ins2/n750w300");
	vsn->push_back("ins2/n750w400");
	vsn->push_back("ins2/n750w500");
	
	vsn->push_back("ins2/n800w100");
	vsn->push_back("ins2/n800w200");
	vsn->push_back("ins2/n800w300");
	vsn->push_back("ins2/n800w400");
	vsn->push_back("ins2/n800w500");
	
	vsn->push_back("ins2/n850w100");
	vsn->push_back("ins2/n850w200");
	vsn->push_back("ins2/n850w300");
	vsn->push_back("ins2/n850w400");
	vsn->push_back("ins2/n850w500");
	
	vsn->push_back("ins2/n900w100");
	vsn->push_back("ins2/n900w200");
	vsn->push_back("ins2/n900w300");
	vsn->push_back("ins2/n900w400");
	vsn->push_back("ins2/n900w500");
	
	vsn->push_back("ins2/n950w100");
	vsn->push_back("ins2/n950w200");
	vsn->push_back("ins2/n950w300");
	vsn->push_back("ins2/n950w400");
	vsn->push_back("ins2/n950w500");

	vsn->push_back("ins2/n1000w100");
	vsn->push_back("ins2/n1000w200");
	vsn->push_back("ins2/n1000w300");
	vsn->push_back("ins2/n1000w400");
	vsn->push_back("ins2/n1000w500");
	/*******************/
	
	cout.setf(ios::fixed);
	
	int s = (int)vsn->size();
	for (int h = 0; h < s; h++)
	{
	    char* ins = vsn->at(h);
		cout << ins << ";";
		
		int n = 10;
		int iter = 30;
		double best = 1000000000;
	    double sum = 0;
	    double sumTime = 0;
	    
	    double *rresult = new double[n];
	    double *rtime = new double[n];
	    for (int i = 0; i < n; i++)
	    {
	    	double s = 0;
	        
	        char ins1[50];
	        strcpy(ins1, ins);
	        strcat(ins1, ".001.txt");
	        
	        char ins2[50];
	        strcpy(ins2, ins);
	        strcat(ins2, ".002.txt");
	        
	        char ins3[50];
	        strcpy(ins3, ins);
	        strcat(ins3, ".003.txt");
	        
	        char ins4[50];
	        strcpy(ins4, ins);
	        strcat(ins4, ".004.txt");
	        
	        char ins5[50];
	        strcpy(ins5, ins);
	        strcat(ins5, ".005.txt");

	        Timer *timer = DefaultTimer::getDefault();
	        timer->start();
	         
	        /*****
	        readins(ins1);
	        readins(ins2);
	        readins(ins3);
	        readins(ins4);
	        readins(ins5);
	        /*****/
	        
	        /*****
	        s += cons(ins1);
	        s += cons(ins2);
	        s += cons(ins3);
	        s += cons(ins4);
	        s += cons(ins5);
	        /*****/
	        
	        /*****/
	        s += gvns(ins1, iter);
	        s += gvns(ins2, iter);
	        s += gvns(ins3, iter);
	        s += gvns(ins4, iter);
	        s += gvns(ins5, iter);
	        /*****/
	        
	        timer->stop();
	        
	        s = s / 5;
	        sum += s;
	        
	        if (s < best)
	        {
	        	best = s;
	        }
	        
	        sumTime += timer->getTime();
	        rtime[i] = timer->getTime();
	        rresult[i] = s;
	        
	        delete timer;
	    }
	    
	    /*********/
	    //result
	    double rr = 0;
	    double avgr = ((double)sum/n);
	    for (int i = 0; i < n; i++)
	    {
	    	rr = rr + (rresult[i] - avgr) * (rresult[i] - avgr);
	    }
	    double dpr = sqrt((double)rr/n);	    
	    cout << setprecision(4) << best << ";" << avgr << ";" << dpr << ";";
	    /*********/
	    
	    //time
	    double rt = 0;
	    double avgt = ((double)sumTime/n);
	    for (int i = 0; i < n; i++)
	    {
	    	rt = rt + (rtime[i] - avgt) * (rtime[i] - avgt);
	    }
	    double dpt = sqrt((double)rt/n);
	    cout << setprecision(4) << avgt << ";" << dpt << endl;
	    cout.flush();
	    
	}
    /*****************/
	
    return EXIT_SUCCESS;
}
//------------------------------------------------------------------------------
