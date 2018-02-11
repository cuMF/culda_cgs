#include "lda_train_kernel.h"
#include "lda_theta_kernel.h"


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
    int       *denseTheta)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int BlockSize = blockDim.x;

    int workerId = tid/BlockSize;
    int laneId = tid%BlockSize;

    if(workerId >= chunkNumDocs)return;


    int docId = workerId + docIdStart;
    

    for(long long idx = docRevIndices[docId] + laneId; 
                  idx < docRevIndices[docId + 1]; 
                  idx += BlockSize){
        int topic = wordTopics[docRevIdx[idx]];
        atomicAdd(&(denseTheta[(docId - docIdStart)*k + topic]), 1);
    }

}

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
    int       *denseTheta)
{
    
    int laneId   = threadIdx.x%32;
    int localId  = threadIdx.x/32;
    int tid      = threadIdx.x + blockIdx.x*blockDim.x;
    int workerId = tid/32;

    if(workerId >= chunkNumDocs)return;

    //if( workerId <= 1000 || workerId >= 5000)return;

    volatile __shared__ int shaPrefixSum[64];

    int docId = docIdStart + workerId;

    //if(laneId == 0)printf("docId:%d, IAStart:%d, IACurEnd:%d\n", docId, thetaMaxIA[docId], thetaMaxIA[docId + 1]);
    int IAStart = thetaMaxIA[docId];
    //compute
    int tmpPrefixSum = 0;
    for(int i = laneId;i < k;i += 32){

        //read
        int tmpVal = denseTheta[(docId - docIdStart)*k + i];
        int tmpBin = tmpVal > 0;
        shaPrefixSum[localId*32 + laneId] = tmpBin;

        //prefix sum
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
        

        //
        //debug
        //if(laneId == 0)
        //    printf("old tmpPrefixSum:%d\n", tmpPrefixSum);
        //printf("laneId:%2d, denseTheta:%d, tmpVal:%d, tmpBin:%d, prefix:%d\n",
        //                 laneId, denseTheta[docId*k + i], tmpVal, tmpBin, shaPrefixSum[laneId]);
        
        //write
        

        int offset = tmpPrefixSum + shaPrefixSum[localId*32 + laneId] - 1;
        
        
        if(tmpVal > 0){

            //printf("blockid:%5d, threadIdx.x:%4d, IAStart + offset:%lld\n", blockIdx.x, threadIdx.x, IAStart + offset);
            thetaA[IAStart  + offset] = tmpVal;
            thetaJA[IAStart + offset] = i;
        }

        tmpPrefixSum += shaPrefixSum[localId*32 + 31];
        

        //debug
        //if(laneId == 0)
        //    printf("new tmpPrefixSum:%d\n", tmpPrefixSum);
        //if(laneId == 0) printf("-------------------------------\n");
        
    }

    
    
    if(laneId == 0){
        //printf("docId:%d\n", docId);
        thetaCurIA[docId] = IAStart + ((tmpPrefixSum + 31)/32*32);
    }
    

    

    //print for debug  
    //if(laneId == 0){
    //    for(int i = 0;i < 32;i ++)
    //    {
    //        printf("%4d:",i);
    //        for(int j = 0;j < 32; j++)
    //            printf("%d ", denseTheta[docId*k + i*32 + j]);
    //        printf("\n");
    //    }
    //    for(int i = thetaMaxIA[docId]; i < thetaMaxIA[docId + 1];i++){
    //        printf("%d,JA(%d), A(%d)\n", i, thetaJA[i], thetaA[i]);
    //    }
    //}
    
    //break;
    

}



void LDAUpdateThetaAPI(
    int       k,
    int       numDocs,
    int       docIdStart,
    int       chunkNumDocs,
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
    cudaStream_t stream)
{

    cudaMemsetAsync(denseTheta, 0, sizeof(int)*chunkNumDocs*k, stream);

    LDAUpdateThetaIncreaseKernel<<<chunkNumDocs, 256,0,stream>>>(
        k,
        numDocs,
        docIdStart,
        chunkNumDocs,
        wordIndices,
        wordTokens,
        wordTopics,
        docRevIndices,
        docRevIdx,
        thetaA,
        thetaCurIA,
        thetaMaxIA,
        thetaJA,
        denseTheta);

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());


    //printf("chunkNumDocs:%d\n",chunkNumDocs);

    //LDAUpdateThetaAlignKernel<<<(chunkNumDocs+1)/2, 64, 0, stream>>>(
    LDAUpdateThetaAlignKernel<<<(chunkNumDocs+1)/2, 64, 0, stream>>>(
        k,
        numDocs,
        docIdStart,
        chunkNumDocs,
        wordIndices,
        wordTokens,
        wordTopics,
        thetaA,
        thetaCurIA,
        thetaMaxIA,
        thetaJA,
        denseTheta
        );

    cudaDeviceSynchronize();
    cudaPeekAtLastError();    

    //sleep(10);
    //cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //exit(0);
}