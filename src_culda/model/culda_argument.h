#ifndef _CULDA_ARGUMENT_H_

#define _CULDA_ARGUMENT_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <queue>
#include <set>

#include <cuda_runtime_api.h>


const int SCacheSize       = 64;

const int UpdateNumWorkers = 28*16;
/*/
Optimal on Titan X: 24*16
Optimal on P100   : 56*16
Optimal on V100   : 80*16
*/


const int TrainBlockSize   = 1024;
const int NumConWorkers    = 28*2;
const int ShaMemPad        = 0;


const int MaxNumGPU        = 32;
const int ReduceParameter  = 1024;

//typedef unsigned short PHITYPE;
typedef int PHITYPE;

typedef int TokenIdxType;

using namespace std;

#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class Argument
{
public:
	int numGPUs;
	int k;
	int iteration;
	int numWorkers;
	int numChunks;
	std::string inputFilePrefix;
	
	std::string outputFilePrefix;

	std::string outputWordFileName;
	std::string outputDocFileName;
	
	float alpha;
	float beta;


	void printArgument(){

		printf("numGPUs     :%d\n", numGPUs);
		printf("k           :%d\n", k);
		printf("iteration   :%d\n", iteration);
		printf("numWorkers  :%d\n", numWorkers);
		printf("numChunks   :%d\n", numChunks);
		printf("alpha       :%.2f\n", alpha);
		printf("beta        :%.2f\n", beta);
		printf("prefix      :%s\n", inputFilePrefix.c_str());		
		printf("outfile     :%s\n", outputFilePrefix.c_str());
		printf("\n");
	}
};

#endif