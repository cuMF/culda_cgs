#ifndef _LDA_TRAIN_KERNEL_H_
#define _LDA_TRAIN_KERNEL_H_


#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <unistd.h>


#include "../model/culda_argument.h"
#include "../model/doc.h"



__global__ void initRandState(curandState *state);


__global__ void LDAKernelTrain(
    int          k, //parameters
    float        alpha,
    float        beta,
    int          numDocs,  // corpora 
    int          numWords,
    long long    numTokens,
    long long   *wordIndices,        // data, numWords + 1
    int         *slotIdToWordId,     // data, numSlots
    long long   *slotIndices,        // data, numSlots*2
    int         *wordTokens,         // data, numTokens
    short       *wordTopics,         // data, numTokens
    short       *thetaA,             //model, values, thetaNNZ
    int         *thetaMaxIA,         //model, offsets, numDocs + 1,
    int         *thetaCurIA,         //model, offsets, numDocs,
    short       *thetaJA,            //model, column indices, thetaNNZ
    int          docIdStart,
    PHITYPE     *phiTopicWord,       //model, numWords*k
    int         *phiTopic,           //model, k
    half        *phiHead,            //model, numWords*k
    curandState *randState,
    int          randStateSize,
    int          GPUid,
    double      *wordPerplexity,
    long long   *docRevIndices);

double LDATrainPerplexity(Document &, cudaStream_t *streams = NULL);

#endif