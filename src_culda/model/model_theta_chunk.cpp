/* Implementations of class ModelThetaChunk */

#include "model_theta_chunk.h"

ModelThetaChunk::ModelThetaChunk():
               k(0),
               numDocs(0),
               numWords(0),
               numChunks(0),
               chunkId(0),
               docIdStart(0),
               docIdEnd(0),
               chunkNumDocs(0),
               chunkNNZ(0),
               deviceThetaA(NULL),
               deviceThetaJA(NULL),
               deviceThetaCurIA(NULL),
               deviceThetaMaxIA(NULL),
               deviceDenseTheta(NULL),
               hostThetaA(NULL),
               hostThetaJA(NULL),
               hostThetaCurIA(NULL),
               hostThetaMaxIA(NULL)             
{
    clearPtr();
}

ModelThetaChunk::ModelThetaChunk(
               int argK,
               int argNumDocs, 
               int argNumWords, 
               int argNumChunks, 
               int argChunkId, 
               int argDocIdStart, 
               int argDocIdEnd,
               int argChunkNumDocs, 
               int argChunkNNZ):
               k(argK),
               numDocs(argNumDocs),
               numWords(argNumWords),
               numChunks(argNumChunks),
               chunkId(argChunkId),
               docIdStart(argDocIdStart),
               docIdEnd(argDocIdEnd),
               chunkNumDocs(argChunkNumDocs),
               chunkNNZ(argChunkNNZ),
               deviceThetaA(NULL),
               deviceThetaJA(NULL),
               deviceThetaCurIA(NULL),
               deviceThetaMaxIA(NULL),
               deviceDenseTheta(NULL),
               hostThetaA(NULL),
               hostThetaJA(NULL),
               hostThetaCurIA(NULL),
               hostThetaMaxIA(NULL)             
{
    clearPtr();
}



void ModelThetaChunk::InitData(const vector<int> &docLenVec)
{
    //alloc space
    hostThetaA = new short[chunkNNZ];
    hostThetaJA = new short[chunkNNZ];
    hostThetaCurIA = new int[numDocs];
    hostThetaMaxIA = new int[numDocs + 1];

    //CPU side
    memset(hostThetaMaxIA, 0, sizeof(int)*(numDocs + 1));

    int offset = 0;
    for(int docId = docIdStart; docId < docIdEnd; docId ++){
        hostThetaMaxIA[docId] = offset;
        hostThetaMaxIA[docId + 1] = offset + docLenVec[docId];
        offset += docLenVec[docId];
    }
    for(int docId = docIdEnd; docId <= numDocs ;docId ++)
        hostThetaMaxIA[docId] = offset;

    cudaSetDevice(chunkId);
    //GPU side
    cudaMalloc((void**)&deviceThetaA,     sizeof(short)*chunkNNZ);
    cudaMalloc((void**)&deviceThetaJA,    sizeof(short)*chunkNNZ);   
    cudaMalloc((void**)&deviceThetaMaxIA, sizeof(int)*(numDocs + 1));
    cudaMalloc((void**)&deviceThetaCurIA, sizeof(int)*numDocs);
    cudaMalloc((void**)&deviceDenseTheta, sizeof(int)*UpdateNumWorkers*k);

    long long totalByte = sizeof(short)*chunkNNZ +
                          sizeof(short)*chunkNNZ +
                          sizeof(int)*(numDocs + 1) +
                          sizeof(int)*numDocs +
                          sizeof(int)*UpdateNumWorkers*k;
    printf("theta chunk size:%.3f GB\n",totalByte/(1024.0*1024.0*1024.0));

    //exit(0);

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //transfer MaxIA
    toGPU();
}

void ModelThetaChunk::UpdateThetaGPU(Document &doc, cudaStream_t stream)
{
    
    cudaSetDevice(chunkId);

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cudaMemsetAsync(deviceThetaA,  0, sizeof(short)*chunkNNZ, stream);
    cudaMemsetAsync(deviceThetaJA, 0, sizeof(short)*chunkNNZ, stream);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    LDAUpdateThetaAPI(
        k,
        numDocs,
        chunkNumDocs,
        docIdStart,
        docIdEnd,
        doc.docChunkVec[chunkId]->deviceWordIndices,
        doc.docChunkVec[chunkId]->deviceWordTokens,
        doc.docChunkVec[chunkId]->deviceWordTopics,
        doc.docChunkVec[chunkId]->deviceDocRevIndices,
        doc.docChunkVec[chunkId]->deviceDocRevIdx,
        deviceThetaA,
        deviceThetaCurIA,
        deviceThetaMaxIA,
        deviceThetaJA,
        deviceDenseTheta,
        stream
    );
    
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
}

void ModelThetaChunk::validTheta(Document &doc)
{

    printf("Calling ModelThetaChunk::validTheta() for chunk %d ...\n", chunkId);
    clock_t clockStart = clock();
    

    printf("theta zero check\n");
    for(int docId = docIdStart; docId < docIdEnd; docId ++)
    {   

        int foundFlag = 0, errorFlag = 0;
        long long tmpStart = hostThetaMaxIA[docId];
        long long tmpEnd   = hostThetaMaxIA[docId + 1];

        if(tmpStart%32 != 0){
            printf("tmpStart non-aligned error\n");
            exit(0);

        }
        if(tmpEnd%32   != 0){
            printf("tmpEnd   non-aligned error\n");
            exit(0);
        }

        for(long long tmpIdx = tmpStart; tmpIdx < tmpEnd; tmpIdx ++){

            if(hostThetaJA[tmpIdx] == 0 && hostThetaA[tmpIdx] != 0)
            {
                if(foundFlag == 1)errorFlag = 1;
                else foundFlag = 1;    
            }
        }

        if(errorFlag == 1)
        {
            printf("error in validTheta\n");

            for(long long tmpIdx = tmpStart; tmpIdx < tmpEnd; tmpIdx ++){
                printf("IA(%lld), JA(%d), A(%d)\n", tmpIdx, hostThetaJA[tmpIdx], hostThetaA[tmpIdx]);
            }
            exit(0);
        }
    }


    //doc by doc check
    int *tmpThetaArray = new int[k]();  
    int *tmpMask = new int[k]();  
    for(int docId = docIdStart;docId < docIdEnd; docId ++){

        //generate the dense array
        for(int i = 0;i < k;i++){
            tmpThetaArray[i] = 0;
            tmpMask[i]       = 0;
        }

        for(long long revIdx = doc.docChunkVec[chunkId]->docRevIndices[docId];
                      revIdx < doc.docChunkVec[chunkId]->docRevIndices[docId + 1]; 
                      revIdx ++){

            int tokenIdx = doc.docChunkVec[chunkId]->docRevIdx[revIdx];
            int tmpTopic = doc.docChunkVec[chunkId]->wordTopics[tokenIdx];
            tmpThetaArray[tmpTopic] ++;
        }

        long long tmpStart = hostThetaMaxIA[docId];
        long long tmpEnd   = hostThetaMaxIA[docId + 1];

        //round 1, check non-zero element
        for(long long tmpIdx = tmpStart; tmpIdx < tmpEnd; tmpIdx ++){
            int tmpK = hostThetaJA[tmpIdx];
            int tmpVal = hostThetaA[tmpIdx];

            if(tmpVal == 0)continue;

            tmpMask[tmpK] = 1;
            if(tmpThetaArray[tmpK] != tmpVal){
                printf("ValidTheta Error 1: docId(%d), topic(%d), tmpTheta(%d), theta(%d), IA(%lld)\n",
                       docId, 
                       tmpK, 
                       tmpThetaArray[tmpK], 
                       tmpVal, 
                       tmpIdx);

                printf("tmpTheta:\n");
                
                for(int j = 0;j < k/32;j ++){
                    printf("%2d:",j);
                    for(int m = 0;m < 32;m ++){
                        printf("%d,",tmpThetaArray[j*32 + m]);
                    }
                    printf("\n");
                }
                exit(0);
            }
        }

        //round2, check zero element
        for(int tmpK = 0;tmpK < k; tmpK ++){
            if(tmpMask[tmpK] == 1)continue;

            if(tmpThetaArray[tmpK] != 0){
                printf("ValidTheta Error 2: docId(%d), topic(%d), val(%d)\n", 
                    docId, tmpK, tmpThetaArray[tmpK]);
                
                printf("docId:(%d)\n", docId);
                for(int idx = hostThetaMaxIA[docId]; idx < hostThetaMaxIA[docId +1]; idx ++){
                    printf("IA(%d), JA(%d), A(%d)\n", idx, hostThetaJA[idx], hostThetaA[idx]);

                    if(hostThetaJA[idx] == 0 && hostThetaA[idx] == 0)break;
                }

                printf("tmpTheta:\n");
                for(int j = 0;j < k/32;j ++){
                    printf("%2d:",j);
                    for(int m = 0;m < 32;m ++){
                        printf("%d,",tmpThetaArray[j*32 + m]);
                    }
                    printf("\n");
                }
                exit(0);
            }
        }        
    }

    printf("Validate Theta passed ...\n");
    printf("ValidateTheta time:%.2fs\n", (clock() - clockStart)/(double)CLOCKS_PER_SEC);

    delete []tmpThetaArray;    
    delete []tmpMask;
}

void ModelThetaChunk::toGPU()
{
    //printf("ModelThetaChunk(%d)::toGPU() ...\n", chunkId);
     
    cudaMemcpy(deviceThetaMaxIA, 
               hostThetaMaxIA, 
               sizeof(int)*(numDocs + 1),  
               cudaMemcpyHostToDevice);
    //gpuErr(cudaPeekAtLastError());

    //printf("ModelThetaChunk::toGPU() finished ...\n\n");
}

void ModelThetaChunk::toCPU()
{
    //printf("ModelThetaChunk::thetaToCPU() ...\n");
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //theta
    cudaMemcpy(hostThetaA,
               deviceThetaA,
               sizeof(short)*chunkNNZ,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    
    cudaMemcpy(hostThetaJA,
               deviceThetaJA,
               sizeof(short)*chunkNNZ,
               cudaMemcpyDeviceToHost);

    cudaMemcpy(hostThetaMaxIA,
               deviceThetaMaxIA,
               sizeof(int)*(numDocs + 1),
               cudaMemcpyDeviceToHost);
    gpuErr(cudaPeekAtLastError());

    cudaMemcpy(hostThetaCurIA,
               deviceThetaCurIA,
               sizeof(int)*numDocs,
               cudaMemcpyDeviceToHost);
    gpuErr(cudaPeekAtLastError());

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    //printf("ModelThetaChunk::thetaToCPU() finished ...\n");

}

void ModelThetaChunk::clearPtr()
{
    if(deviceThetaA     != NULL) cudaFree(deviceThetaA);
    if(deviceThetaJA    != NULL) cudaFree(deviceThetaJA);
    if(deviceThetaMaxIA != NULL) cudaFree(deviceThetaMaxIA);
    if(deviceThetaCurIA != NULL) cudaFree(deviceThetaCurIA);
    if(deviceDenseTheta != NULL) cudaFree(deviceDenseTheta);
    
    //CPU data release
    if(hostThetaA     != NULL) delete []hostThetaA;
    if(hostThetaJA    != NULL) delete []hostThetaJA;
    if(hostThetaMaxIA != NULL) delete []hostThetaMaxIA;
    if(hostThetaCurIA != NULL) delete []hostThetaCurIA;

    deviceThetaA       = NULL;
    deviceThetaJA      = NULL;
    deviceThetaMaxIA   = NULL;
    deviceThetaCurIA   = NULL;
    deviceDenseTheta   = NULL;

    hostThetaA         = NULL;
    hostThetaJA        = NULL;
    hostThetaMaxIA     = NULL;
    hostThetaCurIA     = NULL;
}

