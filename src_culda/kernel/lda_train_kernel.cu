
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

#include "lda_train_kernel.h"
#include "../model/culda_argument.h"

using namespace std;

__global__ void initRandState(curandState *state)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    curand_init(clock() + tid, tid, 0,&state[tid]);
}

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
    double      *wordPerplexity,     //numWords
    long long   *docRevIndices)
{    


    int tid      = threadIdx.x +  blockIdx.x*blockDim.x;
    int workerId = tid/TrainBlockSize;
    int laneId   = threadIdx.x%32;
    int localId  = threadIdx.x/32;
    
    //samling index
    volatile __shared__ float prefixSumQTree[32];
    volatile __shared__ float prefixSumSTree[TrainBlockSize/32][32];
    volatile __shared__ float prefixSumSample[TrainBlockSize/32][32];
    
    //cache to store phi.
    volatile __shared__ float phiHeadCache[1024 + ShaMemPad + 0];

    int wordId     = slotIdToWordId[workerId];
    long long tokenStart = __ldg(&slotIndices[workerId*2]);
    long long tokenEnd   = __ldg(&slotIndices[workerId*2 + 1]);
        
    //load phi head into cache
    int tmpEnd = k/32;
        
    for(int QIdx = localId; QIdx < tmpEnd; QIdx += TrainBlockSize/32){

        int   tmpK   = QIdx*32 + laneId;
        float tmpVal = __half2float(phiHead[k*wordId + tmpK]);
        phiHeadCache[tmpK] = tmpVal;

        tmpVal = alpha*tmpVal;
        tmpVal += __shfl_down(tmpVal, 16);
        tmpVal += __shfl_down(tmpVal, 8);
        tmpVal += __shfl_down(tmpVal, 4);
        tmpVal += __shfl_down(tmpVal, 2);
        tmpVal += __shfl_down(tmpVal, 1);  
        tmpVal = __shfl(tmpVal, 0);
        prefixSumQTree[QIdx] = tmpVal;       
    }
    __syncthreads();
        
    //accumulation prefixSumQTree
    if(localId == 0){
        if(laneId >= 1) 
            prefixSumQTree[laneId] = prefixSumQTree[laneId - 1] + prefixSumQTree[laneId]; 
        if(laneId >= 2) 
            prefixSumQTree[laneId] = prefixSumQTree[laneId - 2]  + prefixSumQTree[laneId]; 
        if(laneId >= 4) 
            prefixSumQTree[laneId] = prefixSumQTree[laneId - 4]  + prefixSumQTree[laneId]; 
        if(laneId >= 8) 
            prefixSumQTree[laneId] = prefixSumQTree[laneId - 8]  + prefixSumQTree[laneId]; 
        if(laneId >= 16)
            prefixSumQTree[laneId] = prefixSumQTree[laneId - 16] + prefixSumQTree[laneId];  
    }
    __syncthreads();
    float Q = prefixSumQTree[31];

    float sumPerplexity = 0.0;
    
    //int stateId = (workerId*TrainBlockSize/32 + localId)%randStateSize;
    for(int tokenIdx = tokenStart + localId;
            tokenIdx < tokenEnd;
            tokenIdx += TrainBlockSize/32) //iterate over tokens
    {
        int docId = __ldg(&wordTokens[tokenIdx]);
        
        //computing S.
        float S = 0;
        int IAStart  = __ldg(&thetaMaxIA[docId]); //L1 cache
        int IACurEnd = __ldg(&thetaCurIA[docId]); //L1 cache
        prefixSumSTree[localId][laneId] = 0;

        for(int tmpIdx = IAStart + laneId, SIdx = 0; 
                tmpIdx < IACurEnd; 
                tmpIdx += 32){

            int   colVal = __ldg(&thetaA[tmpIdx]); //L1 cache
            int   colK   = __ldg(&thetaJA[tmpIdx]); //L1 cache
            //int colVal = thetaA[tmpIdx];
            //int colK   = thetaJA[tmpIdx];
            float tmpP1k = colVal*phiHeadCache[colK];
            //go reduce.
            tmpP1k += __shfl_down(tmpP1k, 16);
            tmpP1k += __shfl_down(tmpP1k, 8);
            tmpP1k += __shfl_down(tmpP1k, 4);
            tmpP1k += __shfl_down(tmpP1k, 2);
            tmpP1k += __shfl_down(tmpP1k, 1);
            tmpP1k = __shfl(tmpP1k, 0);

            S += tmpP1k;
            prefixSumSTree[localId][SIdx] = S;
            SIdx ++;
        }
        S = __shfl(S,0);

        //randomly generate u.
        float u;
        if(laneId == 0)u = curand_uniform(&(randState[workerId%randStateSize]));
        u = __shfl(u, 0);
        int newZ  = 0;

        if(u < S/(S+Q)) 
        {

            //totalS ++;
            //tmpClock = clock64();
            
            float transU = u*(S+Q);
            
            float tmpSumHigh, tmpSumLow = 0.0;
            tmpSumHigh = prefixSumSTree[localId][laneId];
            tmpSumLow  = __shfl_up(tmpSumHigh, 1, 32);
            if(laneId == 0)tmpSumLow = 0;

            int voteFlag = 0;
            if(transU < tmpSumHigh) voteFlag = 1;
            int lvl1Idx = __ffs(__ballot(voteFlag)) - 1;
            
            int overflowFlag = 0;

            if(lvl1Idx < 0) lvl1Idx = (IACurEnd - IAStart)/32 - 1;
            
            //float originalU = transU;
            transU = transU - tmpSumLow;
            transU = __shfl(transU, lvl1Idx);

            int tmpIdx = IAStart + lvl1Idx*32 + laneId;
            int tmpNewZ = __ldg(&thetaJA[tmpIdx]);
            int colVal = __ldg(&thetaA[tmpIdx]);
            float p1k = colVal*phiHeadCache[tmpNewZ];
            
            prefixSumSample[localId][laneId] = p1k;
            
            if(laneId >= 1) prefixSumSample[localId][laneId] = 
                prefixSumSample[localId][laneId - 1] + prefixSumSample[localId][laneId]; 
            if(laneId >= 2) prefixSumSample[localId][laneId] = 
                prefixSumSample[localId][laneId - 2]  + prefixSumSample[localId][laneId]; 
            if(laneId >= 4) prefixSumSample[localId][laneId] = 
                prefixSumSample[localId][laneId - 4]  + prefixSumSample[localId][laneId]; 
            if(laneId >= 8) prefixSumSample[localId][laneId] = 
                prefixSumSample[localId][laneId - 8]  + prefixSumSample[localId][laneId]; 
            if(laneId >= 16)prefixSumSample[localId][laneId] = 
                prefixSumSample[localId][laneId - 16] + prefixSumSample[localId][laneId];  

            float tmpSum = prefixSumSample[localId][laneId];
            
            voteFlag = 0;
            if(transU < tmpSum) voteFlag = 1;
            int offset = __ffs(__ballot(voteFlag)) - 1;

            //int offset1 = offset;
            //offset = 31 - __clz(__ballot(colVal>0));
            newZ = __shfl(tmpNewZ, offset);

        }
        else //bucket Q
        {

            float transU = (u - S/(S+Q))*(S+Q);
            //totalQ ++;
            //float originalU = transU;
        
            //level 1: decide position
            float tmpSumHigh, tmpSumLow = 0.0;
            tmpSumHigh = prefixSumQTree[laneId];
            tmpSumLow  = __shfl_up(tmpSumHigh, 1, 32);
            if(laneId == 0)tmpSumLow = 0;
            
            //voting for lvl1Idx
            int voteFlag = 0;
            if(transU < tmpSumHigh) voteFlag = 1; //voteFlag = transU < tmpSumHigh;
            int lvl1Idx = __ffs(__ballot(voteFlag)) - 1;
            if(lvl1Idx < 0) lvl1Idx = 31;

            

            float originalU = transU;
            transU = transU - tmpSumLow;
            transU = __shfl(transU, lvl1Idx);

            prefixSumSample[localId][laneId] = alpha*phiHeadCache[32*lvl1Idx + laneId];

            // accumulation
            if(laneId >= 1) prefixSumSample[localId][laneId] = 
                prefixSumSample[localId][laneId - 1] + prefixSumSample[localId][laneId]; 
            if(laneId >= 2) prefixSumSample[localId][laneId] = 
                prefixSumSample[localId][laneId - 2]  + prefixSumSample[localId][laneId]; 
            if(laneId >= 4) prefixSumSample[localId][laneId] = 
                prefixSumSample[localId][laneId - 4]  + prefixSumSample[localId][laneId]; 
            if(laneId >= 8) prefixSumSample[localId][laneId] = 
                prefixSumSample[localId][laneId - 8]  + prefixSumSample[localId][laneId]; 
            if(laneId >= 16)prefixSumSample[localId][laneId] = 
                prefixSumSample[localId][laneId - 16] + prefixSumSample[localId][laneId]; 

            voteFlag = 0;
            tmpSumLow = 0;
            tmpSumHigh = prefixSumSample[localId][laneId];
            tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);

            if(laneId == 0)tmpSumLow = 0;
            
            if( transU < tmpSumHigh)voteFlag = 1; //voteFlag = transU < tmpSumHigh;
            int lvl2Idx = __ffs(__ballot(voteFlag)) - 1;  

            if(lvl2Idx < 0)lvl2Idx = 31;

            newZ = lvl1Idx*32 + lvl2Idx;
            
            //if(tmpFlag == 1)return;
        }
            
        //update & get perplexity
        if(laneId == 0){
            wordTopics[tokenIdx] = newZ;
            sumPerplexity += log((S+Q)/(docRevIndices[docId + 1] - docRevIndices[docId] + k*alpha));
        }
    } 
    if(threadIdx.x%32 == 0)
        wordPerplexity[(threadIdx.x+blockDim.x*blockIdx.x)/32] = sumPerplexity;

    /*
    float totalTime = (clock64() - startClock)/1000000000.0;
    if(GPUid == 0 && laneId == 0 && localId == 0 && workerId < -1){
        //printf("worker id:%5d, time: %.2fB S1time: %.2fB S2time: %.2fB Qtime: %.2fB, other: %.2fB, innerLoopTime: %.2fB\n", workerId, totalTime, S1Time, S2Time, QTime, otherTime, innerLoopTime);
        printf("worker id:%5d, time: %.2fB\n", workerId, totalTime);
    }
    */
}

__global__ void LDATrainPerplexityReduce1(double *perplexity, double *perplexityMid, int numVals){

    
    int numWarps = gridDim.x*blockDim.x/32;
    int tid      = threadIdx.x + blockIdx.x*blockDim.x;
    int warpId   = tid/32;
    int laneId   = tid%32; 


    int perWarpSize  = ((numVals + numWarps - 1)/numWarps + 31)/32*32;
    int startIdx     = perWarpSize*warpId + laneId;
    int endIdx       = perWarpSize*warpId + perWarpSize;

    double totalProd = 0;
    for(long long i = startIdx;i < endIdx; i += 32){

        int tmpProd = 0;
        if(i < numVals)tmpProd = perplexity[i];
        
        tmpProd += __shfl_down(tmpProd, 16);
        tmpProd += __shfl_down(tmpProd, 8);
        tmpProd += __shfl_down(tmpProd, 4);
        tmpProd += __shfl_down(tmpProd, 2);
        tmpProd += __shfl_down(tmpProd, 1);

        totalProd += tmpProd;
    }

    if(laneId == 0) perplexityMid[warpId] = totalProd;
}

__global__ void LDATrainPerplexityReduce2(double *perplexityMid)
{

    double sum = 0;
    for(int i = threadIdx.x; i < ReduceParameter; i += 32){
        double tmpProd = perplexityMid[i];

        tmpProd += __shfl_down(tmpProd, 16);
        tmpProd += __shfl_down(tmpProd, 8);
        tmpProd += __shfl_down(tmpProd, 4);
        tmpProd += __shfl_down(tmpProd, 2);
        tmpProd += __shfl_down(tmpProd, 1);

        sum += tmpProd;
    }

    if(threadIdx.x == 0)perplexityMid[0] = sum;
}

double LDATrainPerplexity(Document &doc, cudaStream_t *streams)
{

    double tmpSum[MaxNumGPU];
    double sum = 0;
    if(streams == NULL){

        for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++){

            cudaSetDevice(chunkId);
        
            //reduce 1.
            LDATrainPerplexityReduce1<<<ReduceParameter/2, 64, 0>>>(
                doc.docChunkVec[chunkId]->deviceWordPerplexity,
                doc.docChunkVec[chunkId]->deviceWordPerplexityMid,
                doc.docChunkVec[chunkId]->numWords*(TrainBlockSize/32));

            double testMid[ReduceParameter];
            cudaMemcpy(testMid, doc.docChunkVec[chunkId]->deviceWordPerplexityMid, sizeof(double)*ReduceParameter, cudaMemcpyDeviceToHost);

            //cudaDeviceSynchronize();
            //gpuErr(cudaPeekAtLastError());

            //reduce 2.
            LDATrainPerplexityReduce2<<<1,32,0>>>(doc.docChunkVec[chunkId]->deviceWordPerplexityMid);

            cudaMemcpy(tmpSum, doc.docChunkVec[chunkId]->deviceWordPerplexityMid, sizeof(double), cudaMemcpyDeviceToHost);


            sum += tmpSum[0];
            //printf("loglike:%.4f e10\n", sum);
        }
        return (sum/doc.numTokens);
    }
    else
    {

        for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++)
        {
            cudaSetDevice(chunkId);
            //reduce 1.
            LDATrainPerplexityReduce1<<<ReduceParameter/2, 64, 0, streams[chunkId]>>>(
                doc.docChunkVec[chunkId]->deviceWordPerplexity,
                doc.docChunkVec[chunkId]->deviceWordPerplexityMid,
                doc.docChunkVec[chunkId]->numWords);

            //cudaDeviceSynchronize();
            //gpuErr(cudaPeekAtLastError());

            //reduce 2.
            LDATrainPerplexityReduce2<<<1,32,0, streams[chunkId]>>>(doc.docChunkVec[chunkId]->deviceWordPerplexityMid);
            cudaMemcpyAsync(tmpSum + chunkId, doc.docChunkVec[chunkId]->deviceWordPerplexityMid, sizeof(double), cudaMemcpyDeviceToHost, streams[chunkId]);
        }

        for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++)
            cudaStreamSynchronize(streams[chunkId]);

        for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++)
            sum += tmpSum[chunkId];

        return sum/doc.numTokens;
    }
    //return exp(-1*sum/doc.numTokens);
}















