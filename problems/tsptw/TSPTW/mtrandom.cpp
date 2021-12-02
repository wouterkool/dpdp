//------------------------------------------------------------------------------
#include "includes.h"
#include "mtrandom.h"
//------------------------------------------------------------------------------
#define MTRANDOM_N 624
#define MTRANDOM_M 397
#define MTRANDOM_MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define MTRANDOM_UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define MTRANDOM_LOWER_MASK 0x7fffffffUL /* least significant r bits */

static unsigned long mtrandom_mt[MTRANDOM_N]; /* the array for the state vector  */
static int mtrandom_mti = MTRANDOM_N+1; /* mti==N+1 means mt[N] is not initialized */
//------------------------------------------------------------------------------
void init_genrand(unsigned long s)
{
    mtrandom_mt[0]= s & 0xffffffffUL;
    for (mtrandom_mti=1; mtrandom_mti < MTRANDOM_N; mtrandom_mti++) {
    	mtrandom_mt[mtrandom_mti] = 
	    (1812433253UL * (mtrandom_mt[mtrandom_mti-1] ^ (mtrandom_mt[mtrandom_mti-1] >> 30)) + mtrandom_mti); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
    	mtrandom_mt[mtrandom_mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}
//------------------------------------------------------------------------------
unsigned long genrand_int32(void)
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MTRANDOM_MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mtrandom_mti >= MTRANDOM_N) { /* generate N words at one time */
        int kk;

        if (mtrandom_mti == MTRANDOM_N+1)   /* if init_genrand() has not been called, */
            init_genrand(5489UL); /* a default initial seed is used */

        for (kk=0;kk<MTRANDOM_N-MTRANDOM_M;kk++) {
            y = (mtrandom_mt[kk]&MTRANDOM_UPPER_MASK)|(mtrandom_mt[kk+1]&MTRANDOM_LOWER_MASK);
            mtrandom_mt[kk] = mtrandom_mt[kk+MTRANDOM_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<MTRANDOM_N-1;kk++) {
            y = (mtrandom_mt[kk]&MTRANDOM_UPPER_MASK)|(mtrandom_mt[kk+1]&MTRANDOM_LOWER_MASK);
            mtrandom_mt[kk] = mtrandom_mt[kk+(MTRANDOM_M-MTRANDOM_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mtrandom_mt[MTRANDOM_N-1]&MTRANDOM_UPPER_MASK)|(mtrandom_mt[0]&MTRANDOM_LOWER_MASK);
        mtrandom_mt[MTRANDOM_N-1] = mtrandom_mt[MTRANDOM_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mtrandom_mti = 0;
    }
  
    y = mtrandom_mt[mtrandom_mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
