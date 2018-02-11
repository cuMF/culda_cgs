#include "lda_train_kernel.h"
#include "lda_theta_kernel.h"


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
    )
{
    volatile __shared__ int shaPrefixSum[64];

    int tid      = threadIdx.x + blockIdx.x*blockDim.x;
    int workerId = tid/32;
    int laneId   = threadIdx.x%32;
    int localId  = threadIdx.x/32;

    if(workerId >= numThetaWorkers)return;

    for(int iteDocId = docIdStart + workerId; 
            iteDocId < docIdEnd; 
            iteDocId += numThetaWorkers){

        //clean the array
        int startDenseIdx = workerId*k;
        int endDenseIdx   = workerId*k + k;
        for(int denseIdx = startDenseIdx + laneId; denseIdx < endDenseIdx; denseIdx += 32)
            denseTheta[denseIdx] = 0;

        //generate the dense array
        for(long long idx = docRevIndices[iteDocId] + laneId;
                      idx < docRevIndices[iteDocId + 1];
                      idx += 32){
            int topic = wordTopics[docRevIdx[idx]];
            atomicAdd(&(denseTheta[startDenseIdx + topic]), 1);
        }

        //generate the sparse array
        int IAStart = thetaMaxIA[iteDocId];
        int tmpPrefixSum = 0;
        
        for(int i = laneId; i < k;i += 32){
            
            //read
            int tmpVal = denseTheta[startDenseIdx + i];
            int tmpBin = tmpVal > 0;
            shaPrefixSum[localId*32 + laneId] = tmpBin;
            
            //prefixsum
            if(laneId >= 1)
                shaPrefixSum[localId*32 + laneId] = shaPrefixSum[localId*32 + laneId - 1]  + shaPrefixSum[localId*32 + laneId];
            if(laneId >= 2) 
                shaPrefixSum[localId*32 + laneId] = shaPrefixSum[localId*32 + laneId - 2]  + shaPrefixSum[localId*32 + laneId]; 
            if(laneId >= 4) 
                shaPrefixSum[localId*32 + laneId] = shaPrefixSum[localId*32 + laneId - 4]  + shaPrefixSum[localId*32 + laneId]; 
            if(laneId >= 8) 
                shaPrefixSum[localId*32 + laneId] = shaPrefixSum[localId*32 + laneId - 8]  + shaPrefixSum[localId*32 + laneId];
            if(laneId >= 16)
                shaPrefixSum[localId*32 + laneId] = shaPrefixSum[localId*32 + laneId - 16] + shaPrefixSum[localId*32 + laneId];  


            int offset  = tmpPrefixSum + shaPrefixSum[localId*32 + laneId] - 1;
            if(tmpVal > 0){
                thetaA[IAStart + offset] = tmpVal;
                thetaJA[IAStart + offset] = i;
            }
            tmpPrefixSum += shaPrefixSum[localId*32 + 31];
        }

        if(laneId == 0)
            thetaCurIA[iteDocId] = IAStart + ((tmpPrefixSum + 31)/32*32);
    }

}

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
    cudaStream_t stream
    )
{
    LDAUpdateThetaKernel<<<(UpdateNumWorkers+1)/2,64,0,stream>>>(
        k,
        numDocs,
        chunkNumDocs,
        docIdStart,
        docIdEnd,
        wordIndices,
        wordTokens,
        wordTopics,
        docRevIndices,
        docRevIdx,
        thetaA,
        thetaCurIA,
        thetaMaxIA,
        thetaJA,
        denseTheta,
        UpdateNumWorkers
        );
}