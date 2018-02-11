#ifndef _LDA_THETA_KERNEL_H_
#define _LDA_THETA_KERNEL_H_

#include "../model/culda_argument.h"

__global__ void LDAUpdateThetaKernel(
    int           k,
    int           numDocs,
    int           chunkNumDocs,
    int           docIdStart,
    int           docIdEnd,
    long long    *wordIndices,
    int          *wordTokens,
    short        *wordTopics,
    long long    *docRevIndices,
    TokenIdxType *docRevIdx,
    short        *thetaA,
    int          *thetaCurIA,
    int          *thetaMaxIA,
    short        *thetaJA,
    int          *denseTheta,
    int           numThetaWorkers
    );

void LDAUpdateThetaAPI(
    int           k,
    int           numDocs,
    int           chunkNumDocs,
    int           docIdStart,
    int           docIdEnd,
    long long    *wordIndices,
    int          *wordTokens,
    short        *wordTopics,
    long long    *docRevIndices,
    TokenIdxType *docRevIdx,
    short        *thetaA,
    int          *thetaCurIA,
    int          *thetaMaxIA,
    short        *thetaJA,
    int          *denseTheta,
    cudaStream_t stream = 0
    );


#endif