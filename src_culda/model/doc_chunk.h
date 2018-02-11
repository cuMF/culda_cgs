#ifndef _DOC_CHUNK_
#define _DOC_CHUNK_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
#include <sstream>

#include <cuda_runtime_api.h>

#include "culda_argument.h"

using namespace std;

class DocChunk
{
public:
    int chunkId;
    int docIdStart;
    int docIdEnd;

    int numWorkers;
    int numDocs;
    int numWords;
    int numSlots;
    int numChunks;

    long long chunkNumTokens;
    int       chunkNumDocs;

    /* original input data */
    long long    *wordIndices;         // numberWords + 1
    int          *slotIdToWordId;      // numSlots
    long long    *slotIndices;         // numSlots*2
    int          *wordTokens;          // chunkNumTokens
    short        *wordTopics;          // chunkNumTokens
    double       *wordPerplexity;      // chunkNumTokens

    long long    *deviceWordIndices;      // numWords + 1
    int          *deviceSlotIdToWordId;   // numSlots
    long long    *deviceSlotIndices;      // numSlots*2
    int          *deviceWordTokens;       // chunkNumTokens
    short        *deviceWordTopics;       // chunkNumTokens
    double       *deviceWordPerplexity;   // chunkNumTokens

    double       *deviceWordPerplexityMid;

    /* reverse doc data */
    long long    *docRevIndices;       // numDocs + 1
    TokenIdxType *docRevIdx;           // chunkTokenSize

    long long    *deviceDocRevIndices; // numDocs + 1
    TokenIdxType *deviceDocRevIdx;     // chunkTokenSize 

    DocChunk();
    DocChunk(int argChunkId, 
             int argDocIdStart, 
             int argDocIdEnd, 
             int argNumDocs, 
             int argNumChunks);
    ~DocChunk()
    {
        
        if(wordIndices             != NULL)delete []wordIndices;
        if(slotIdToWordId          != NULL)delete []slotIdToWordId;
        if(slotIndices             != NULL)delete []slotIndices;
        if(wordTokens              != NULL)delete []wordTokens;
        if(wordTopics              != NULL)delete []wordTopics;
        if(wordPerplexity          != NULL)delete []wordPerplexity;
        
        if(deviceWordIndices       != NULL)cudaFree(deviceWordIndices);
        if(deviceSlotIdToWordId    != NULL)cudaFree(deviceSlotIdToWordId);
        if(deviceSlotIndices       != NULL)cudaFree(deviceSlotIndices);
        if(deviceWordTokens        != NULL)cudaFree(deviceWordTokens);
        if(deviceWordTopics        != NULL)cudaFree(deviceWordTokens);
        if(deviceWordPerplexity    != NULL)cudaFree(deviceWordPerplexity);
        if(deviceWordPerplexityMid != NULL)cudaFree(deviceWordPerplexityMid);

        if(deviceDocRevIndices     != NULL)cudaFree(deviceDocRevIndices);
        if(deviceDocRevIdx         != NULL)cudaFree(deviceDocRevIdx);

    }

    void loadChunk(string, string, int*);
    void generateTopics(int k);

    void allocGPU(int);
    void toGPU();
    void toCPU();


};

#endif