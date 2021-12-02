#ifndef CPUTIME_
#define CPUTIME_

#include "timer.h"
#include <stdlib.h>

#if defined WIN32
  #include <windows.h>
#endif

class CPUTimer: public Timer
{
  public:
    CPUTimer();

    // Retorna o tempo (em segs e msegs) de CPU cronometrado para uma rotina.
    // Se apenas uma cronometragem foi realizada, entao os valores retornados
    // por getCPUCurrSecs() e getCPUTtotalSecs sao iguais.
    double getCPUCurrSecs();

    // Retorna o tempo total (em segs e msegs) de CPU cronometrado para uma rotina
    virtual double getTime();

    // Retorna o tempo (em segs e msegs) de execucao cronometrado para uma rotina.
    // Se apenas uma cronometragem foi realizada, entao os valores retornados
    // por getCPUCurrSecs() e getCPUTtotalSecs sao iguais.
    double getCronoCurrSecs();

    // Retorna o tempo total (em segs e msegs) de execucao cronometrado para uma rotina.
    double getCronoTotalSecs();

    // Inicia a cronometragem (tempo de execucao e de CPU) de uma rotina
    virtual void start();

    // Encerra a cronometragem (tempo de execucao e de CPU) de uma rotina
    virtual void stop();

    // Prepara o ambiente de cronometragem para ser utilizado em outra rotina
    void reset();


    // Operator to add cputimers
    void operator +=( CPUTimer t );

    inline void increaseCPUTotalSecs( double s){CPUTotalSecs += s; };

    bool started;             // is the timer started?
  private:

    #if defined WIN32
      HANDLE       hFile;
      FILETIME     FT_CreationTime;
      FILETIME     FT_ExitTime;
      FILETIME     FT_KernelTime;
      FILETIME     FT_UserTime;

      FILETIME     FT_CronoStartTime;
      FILETIME     FT_CronoStopTime;

      FILETIME     FT_Aux1Time;
      FILETIME     FT_Aux2Time;
    #else
      double CPUTStart;            // the start time
      double CPUTStop;             // the stop time
      double CronoTStart;
      double CronoTStop;
      double zeit();
      double real_zeit();
    #endif

#ifdef WIN32

    SYSTEMTIME   ST_CPUCurrTime;      // total do tempo de cpu (h:m:s:ms) do �ltimo intervalo cronometrado
    SYSTEMTIME   ST_CPUTotalTime;     // total do tempo de cpu (h:m:s:ms) de todos os intervalos 

    SYSTEMTIME   ST_CronoStartTime;   // hora de inicio da cronometragem (h:m:s:ms)
    SYSTEMTIME   ST_CronoStopTime;    // hora de termino da cronometragem (h:m:s:ms)
    SYSTEMTIME   ST_CronoTotalTime;   // total do tempo cronologico (h:m:s:ms)


    ULONG64      cpu_ti;              // ----------------------------------------------
    ULONG64      cpu_tf;              // Variaveis auxiliares utilizadas pelos m�todos
    ULONG64      crono_ti;            // da classe.
    ULONG64      crono_tf;            //
     LONG64      aux;                 //
     LONG64      secs;                //
     LONG64      msecs;               // ----------------------------------------------
#endif

    double       CPUCurrSecs;         // tempo de cpu cronometrado para uma rotina (segs e msegs)
    double       CPUTotalSecs;        // total do tempo de cpu cronometrado para uma rotina (segs e msegs)

    double       CronoCurrSecs;       // tempo de execucao cronometrado para uma rotina (segs e msegs)
    double       CronoTotalSecs;      // total do tempo de execucao cronometrado para uma rotina (segs e msegs)

    bool         gottime;             // do we have a measured time we can return?
};
#endif

