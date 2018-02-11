#ifndef _LDA_PHI_KERNEL_H_
#define _LDA_PHI_KERNEL_H_

#include <stdio.h>
#include "../model/culda_argument.h"


/* phihead comput kernels */
__global__ void LDAcomputePhiHeadKernel(
    int        k,
    float      beta,
    int        numWords,
    int        numWordsPerWorker,
    PHITYPE   *phiTopicWordShort,
    int       *phiTopic,
    half      *phiHead
);

__global__ void LDAcheckPhiHeadKernel(
    int        k, 
    int        numWords, 
    half      *phiHead
);

void LDAComputePhiHeadAPI( 
    int        k,
    float      beta,
    int        numWords,
    PHITYPE   *phiTopicWordShort,
    int       *phiTopic,
    half      *phiHead,
    cudaStream_t stream=0
);

/* phi update kernels */
__global__ void LDAUpdatePhiKernel(
    int        k,
    int        numWords,
    long long *wordIndices,
    short     *wordTopics,
    PHITYPE   *phiTopicWordShort,
    int       *phiTopicWordSub,
    int       *phiTopic,
    int        numWorkers
);

void LDAUpdatePhiAPI(
    int          k,
    int          numWords,
    long long   *wordIndices,
    short       *wordTopics,
    PHITYPE     *phiTopicWordShort,
    int         *phiTopicWordSub,
    int         *phiTopic,
    cudaStream_t stream=0
);

/*
__global__ void LDAPhiCheckKernel(
    int           k,
    int           numWords,
    PHITYPE      *phiTopicWordShort);
*/

/* MultiGPU Reduce Kernels */
__global__ void LDAUpdatePhiReduceKernelShort(
    int           k,
    int           numWords,
    PHITYPE      *phiTopicWordShort,
    PHITYPE      *phiTopicWordShortCopy
);

__global__ void LDAUpdatePhiReduceKernelInt(
    int         k,
    int         numWords,
    int        *phiTopic,
    int        *phiTopicCopy
);

void LDAUpdatePhiReduceAPI(
    int           k,
    int           numWords,
    PHITYPE      *phiTopicWordShort,
    PHITYPE      *phiTopicWordShortCopy,
    int          *phiTopic,
    int          *phiTopicCopy,
    cudaStream_t stream=0
);

#endif