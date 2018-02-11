#ifndef _LDA_THETA_KERNEL_H_
#define _LDA_THETA_KERNEL_H_

__global__ void LDAUpdateThetaIncreaseKernel(
    int        k,
    int        numDocs,
    int        docIdStart,
    int        chunkNumDocs,
    long long *wordIndices,
    int       *wordTokens,
    short     *wordTopics,
    long long *docRevIndices,
    TokenIdxType *docRevIdx,
    short     *thetaA,
    int       *thetaCurIA,
    int       *thetaMaxIA,
    short     *thetaJA,
    int       *denseTheta);

__global__ void LDAUpdateThetaAlignKernel(
    int        k,
    int        numDocs,
    int        docIdStart,
    int        chunkNumDocs,
    long long *wordIndices,
    int       *wordTokens,
    short     *wordTopics,
    short     *thetaA,
    int       *thetaCurIA,
    int       *thetaMaxIA,
    short     *thetaJA,
    int       *denseTheta);

void LDAUpdateThetaAPI(
    int        k,
    int        numDocs,
    int        docIdStart,
    int        chunkNumDocs,
    long long *wordIndices,
    int       *wordTokens,
    short     *wordTopics,
    long long *docRevIndices,
    TokenIdxType *docRevIdx,
    short     *thetaA,
    int       *thetaCurIA,
    int       *thetaMaxIA,
    short     *thetaJA,
    int       *denseTheta,
    cudaStream_t stream = 0);

#endif