
#include <stdio.h>

#include "lda_train_kernel.h"
#include "lda_phi_kernel.h"

/* phihead comput kernels */
__global__ void LDAcomputePhiHeadKernel(
    int          k,
    float        beta,
    int          numWords,
    int          numWordsPerWorker,
    PHITYPE     *phiTopicWordShort,
    int         *phiTopic,
    half        *phiHead)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int workerId = tid/32;
    int laneId = tid%32;
    int wordId = workerId;

    if(workerId >= numWords)return;

    for(int tmpk = laneId; tmpk < k; tmpk += 32){ 

        float tmpHead = (phiTopicWordShort[wordId*k + tmpk] + beta)/(phiTopic[tmpk] + beta*numWords);

        //if(tmpk = 1024)tmpHead *=1.01;
        phiHead[wordId*k + tmpk]  = __float2half(tmpHead);
    }
}

__global__ void LDAcheckPhiHeadKernel(
    int        k, 
    int        numWords, 
    half      *phiHead)
{

    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int wordId = tid/1024;
    int tmpk = tid%1024;

    float tmp = __half2float(phiHead[wordId*k + tmpk]);
    if(tmp < 0){
        printf("phihead check error:wordid(%d), k(%d), head(%.6f), index(%d)\n", wordId, tmpk, tmp, wordId*k + tmpk);
    }
}

void LDAComputePhiHeadAPI( 
    int        k,
    float      beta,
    int        numWords,
    PHITYPE   *phiTopicWordShort,
    int       *phiTopic,
    half      *phiHead,
    cudaStream_t stream)
{

    //printf("call LDAComputePhiHeadAPI ...\n");
    LDAcomputePhiHeadKernel<<<(numWords+3)/4,128,0, stream>>>(
        k, 
        beta, 
        numWords, 
        1, 
        phiTopicWordShort, 
        phiTopic, 
        phiHead
    );

    //printf("LDAcheckPhiHeadKernel ...\n");
    //LDAcheckPhiHeadKernel<<<numWords,1024>>>(k, numWords, phiHead);
}

/* phi update kernels */

__global__ void LDAUpdatePhiKernel(
    int        k,
    int        numWords,
    long long *wordIndices,
    short     *wordTopics,
    PHITYPE   *phiTopicWordShort,
    int       *phiTopicWordSub,
    int       *phiTopic,
    int        numWorkers)
{
    
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    int workerId = tid/32;
    int laneId = tid%32;

    if(workerId >= numWorkers) return;

    for(int wordId = workerId; wordId < numWords; wordId += numWorkers){

        long long tokenStart = __ldg(&wordIndices[wordId]);     
        long long tokenEnd   = __ldg(&wordIndices[wordId + 1]);

        //clean
        for(int offset = laneId; offset < k;offset += 32)
            phiTopicWordSub[workerId*k + offset] = 0;

        //add
        for(long long tokenIdx =  tokenStart + laneId;
                      tokenIdx <  tokenEnd;
                      tokenIdx += 32) //iterate over tokens
        {
            int tmpK = __ldg(&wordTopics[tokenIdx]);
            atomicAdd(&(phiTopicWordSub[workerId*k + tmpK]),1);
            atomicAdd(&(phiTopic[tmpK]),1);
        }

        //transform
        for(int offset = laneId; offset < k;offset += 32)
            phiTopicWordShort[wordId*k + offset] = phiTopicWordSub[workerId*k + offset];
    }
}

/*
__global__ void LDAPhiCheckKernel(
    int        k,
    int        numWords,
    PHITYPE   *phiTopicWordShort)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid > k*numWords)return;

    if(phiTopicWordShort[tid] < 0)
    {
        printf("phi check error: word(%d), k(%d), int(%d), short(%d)\n", 
            tid/k, tid%k, phiTopicWordShort[tid], phiTopicWordShort[tid]);
    }
}
*/

void LDAUpdatePhiAPI(
    int       k,
    int       numWords,
    long long *wordIndices,
    short     *wordTopics,
    PHITYPE   *phiTopicWordShort,
    int       *phiTopicWordSub,
    int       *phiTopic,
    cudaStream_t stream)
{

    LDAUpdatePhiKernel<<<(UpdateNumWorkers+3)/4, 128, 0,stream>>>(
        k,
        numWords,
        wordIndices,
        wordTopics,
        phiTopicWordShort,
        phiTopicWordSub,
        phiTopic,
        UpdateNumWorkers
    );

    //LDAPhiCheckKernel<<<(k*numWords + 127)/128, 128,0, stream>>>(k,numWords,phiTopicWordShort);
}

/* MultiGPU Reduce Kernels */
__global__ void LDAUpdatePhiReduceKernelShort(
    int        k,
    int        numWords,
    PHITYPE      *phiTopicWordShort,
    PHITYPE      *phiTopicWordShortCopy)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid < k*numWords) phiTopicWordShort[tid] += phiTopicWordShortCopy[tid];
}

__global__ void LDAUpdatePhiReduceKernelInt(
    int        k,
    int        numWords,
    int       *phiTopic,
    int       *phiTopicCopy)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    //if(tid == 0)
    //    printf("phiTopic[0]:%d, phiTopicShort[0]:%d\n", phiTopic[0], phiTopicCopy[0]);
    
    if(tid < k) phiTopic[tid] += phiTopicCopy[tid];
}

void LDAUpdatePhiReduceAPI(
    int           k,
    int           numWords,
    PHITYPE      *phiTopicWordShort,
    PHITYPE      *phiTopicWordShortCopy,
    int          *phiTopic,
    int          *phiTopicCopy,
    cudaStream_t  stream)
{

    LDAUpdatePhiReduceKernelShort<<<(k*numWords + 127)/128,128,0,stream>>>(
        k,
        numWords,
        phiTopicWordShort,
        phiTopicWordShortCopy
    );

    LDAUpdatePhiReduceKernelInt<<<(k + 127)/128,128,0,stream>>>(
        k,
        numWords,
        phiTopic,
        phiTopicCopy
    );
}