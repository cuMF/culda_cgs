

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <cstring>      // std::memset
#include <fstream>

#include <cuda_runtime_api.h>
#include "model_phi.h"
#include "vocab.h"



/* Implementations of class ModelPhi */

ModelPhi::ModelPhi():
               k(0),
               numGPUs(1),
               numDocs(0),
               numWords(0),
               numChunks(1)
{
    for(int i = 0;i < MaxNumGPU;i++){
        hostPhiTopicWordShort[i] = NULL;
        hostPhiTopic[i] = NULL;
    }
    clearPtr();
}

ModelPhi::ModelPhi(
            int argK, int argGPUs, int argDocs, int argWords, int argChunks):
              k(argK),
              numGPUs(argGPUs),
              numDocs(argDocs),
              numWords(argWords),
              numChunks(argChunks)
{
    for(int i = 0;i < MaxNumGPU;i++){
        hostPhiTopicWordShort[i] = NULL;
        hostPhiTopic[i] = NULL;
    }
    clearPtr();
}

void ModelPhi::InitData(Document &doc)
{
    
    clearPtr();
    if(k <= 0 || numDocs <= 0 || numWords <= 0)return;

    //prepare data space for phi.
    for(int chunkId = 0; chunkId < numChunks; chunkId ++){
        hostPhiTopicWordShort[chunkId] = new PHITYPE[k*numWords]();
        hostPhiTopic[chunkId]          = new int[k]();
    }
    
    for(int GPUid = 0; GPUid < numGPUs; GPUid ++){
        ModelPhiGPU *tmpPtr = new ModelPhiGPU(k, numGPUs, GPUid, numDocs, numWords);
        tmpPtr->allocGPU();
        phiChunkVec.push_back(tmpPtr);
    }
}

void ModelPhi::UpdatePhiGPU(Document &doc, int chunkId, cudaStream_t stream)
{
    phiChunkVec[chunkId]->UpdatePhiGPU(doc, chunkId, stream);
}

void ModelPhi::UpdatePhiHead(float beta, cudaStream_t *stream)
{

    if(stream != NULL){
        for(int i = 0;i < numGPUs; i++)
            phiChunkVec[i]->UpdatePhiHead(beta, stream[i]);
    }
    else{
        for(int i = 0;i < numGPUs; i++)
            phiChunkVec[i]->UpdatePhiHead(beta);
    }
}

void ModelPhi::MasterGPUToCPU(cudaStream_t stream)
{

    //phi
    cudaMemcpyAsync(hostPhiTopicWordShort[0],
                    phiChunkVec[0]->devicePhiTopicWordShort,
                    sizeof(PHITYPE)*k*numWords, 
                    cudaMemcpyDeviceToHost,
                    stream);

    cudaMemcpyAsync(hostPhiTopic[0],
                    phiChunkVec[0]->devicePhiTopic,
                    sizeof(int)*k, 
                    cudaMemcpyDeviceToHost,
                    stream);
}

void ModelPhi::MasterGPUCollect(int GPUid, cudaStream_t stream)
{
    cudaMemcpyAsync(phiChunkVec[0]->devicePhiTopicWordShortCopy,
                    phiChunkVec[GPUid]->devicePhiTopicWordShort,
                    sizeof(PHITYPE)*k*numWords,
                    cudaMemcpyDeviceToDevice,
                    stream);
    cudaMemcpyAsync(phiChunkVec[0]->devicePhiTopicCopy,
                    phiChunkVec[GPUid]->devicePhiTopic,
                    sizeof(int)*k,
                    cudaMemcpyDeviceToDevice,
                    stream);
}

void ModelPhi::MasterGPUDistribute(int GPUid, cudaStream_t stream)
{
    cudaMemcpyAsync(phiChunkVec[GPUid]->devicePhiTopicWordShort,
                    phiChunkVec[0]->devicePhiTopicWordShort,
                    sizeof(PHITYPE)*k*numWords,
                    cudaMemcpyDeviceToDevice,
                    stream);
    cudaMemcpyAsync(phiChunkVec[GPUid]->devicePhiTopic,
                    phiChunkVec[0]->devicePhiTopic,
                    sizeof(int)*k,
                    cudaMemcpyDeviceToDevice,
                    stream);
}

void ModelPhi::MasterGPUReduce(cudaStream_t stream)
{
    cudaSetDevice(0);
    LDAUpdatePhiReduceAPI(
        k,
        numWords,
        phiChunkVec[0]->devicePhiTopicWordShort,
        phiChunkVec[0]->devicePhiTopicWordShortCopy,
        phiChunkVec[0]->devicePhiTopic,
        phiChunkVec[0]->devicePhiTopicCopy,
        stream);
}

void ModelPhi::clearPtr()
{
    for(int i = 0;i < phiChunkVec.size(); i++)
        if(phiChunkVec[i] != NULL)delete phiChunkVec[i];

    //CPU data release
    for(int i = 0;i < numChunks;i++){
        if(hostPhiTopicWordShort[i] != NULL) delete hostPhiTopicWordShort[i];
        if(hostPhiTopic[i] != NULL)          delete hostPhiTopic[i];
    }
    //printf("ModelPhi::clearPtr() finished\n");
}



void ModelPhi::savePhi(string fileName)
{
    printf("Saving phi ...\n");

    ofstream phiStream(fileName.c_str(), ios::out);

    int tmpNumWords = numWords;
    tmpNumWords = 1000;
    for(int wordId = 0;wordId < tmpNumWords; wordId++){
        phiStream << "Word id:" << wordId << ", ";

        for(int kite = 0; kite < k;kite++ ){
            if(kite%32 == 0){
                phiStream << endl;
                phiStream.width(2);
                phiStream << kite/32;
                phiStream.width(0);
                phiStream << ":";
            }

            int tmpVal = hostPhiTopicWordShort[0][wordId*k + kite];
            if(tmpVal == 0)
                phiStream << "_,";
            else
                phiStream << hostPhiTopicWordShort[0][wordId*k + kite] << ",";
        }
        phiStream << endl;
    }
}




void ModelPhi::validPhi(Document&doc)
{

    printf("Calling validPhi() ...");
    int tmpPhi[1024];
    int tmpPhiTopic[1024];

    clock_t clockStart = clock();
    int maxPhi = 0;    
    // validate hostPhiTopicWord
    for(int wordId = 0; wordId < numWords; wordId ++){

        for(int i = 0;i < 1024;i++)tmpPhi[i] = 0;
        
        //add
        for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++){
            for(long long tokenId = doc.docChunkVec[chunkId]->wordIndices[wordId]; 
                          tokenId < doc.docChunkVec[chunkId]->wordIndices[wordId + 1]; 
                          tokenId ++){

                tmpPhi[doc.docChunkVec[chunkId]->wordTopics[tokenId]] ++;
            }
        }

        //validate hostPhiTopicWord
        
        for(int i = 0;i < 1024; i++){
            if(hostPhiTopicWordShort[0][wordId*k + i] > maxPhi) maxPhi = hostPhiTopicWordShort[0][wordId*k + i];

            if(tmpPhi[i] != hostPhiTopicWordShort[0][wordId*k + i]){
                
                printf("ValidPhi Error: wordId(%d), topic(%d), tmpphi[i](%d), phi(%d)\n", 
                        wordId, i, tmpPhi[i], hostPhiTopicWordShort[0][wordId*k + i]);

                printf("topic:%d, level1:%d, level2:%d\n", i, i/32, i%32);
                
                for(int j = 0;j < 32;j ++){

                    printf("tmpphi:\n");
                    printf("%2d:",j);
                    for(int m = 0;m < 32;m ++){
                        printf("%d,",tmpPhi[j*32 + m]);
                    }
                    printf("\n");

                    printf("phi   :\n");
                    printf("%2d:",j);
                    for(int m = 0;m < 32;m ++){
                        printf("%d,",hostPhiTopicWordShort[0][wordId*k + j*32 + m]);
                    }
                    printf("\n");

                }
                exit(0);
            }
        }        
    }

    //validate hostPhiTopic
    for(int i = 0;i < 1024; i++)tmpPhiTopic[i] = 0;
    //Step 1: add

    for(int wordId = 0; wordId < numWords; wordId ++){
        
        for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++){        
            for(long long tokenId = doc.docChunkVec[chunkId]->wordIndices[wordId]; 
                          tokenId < doc.docChunkVec[chunkId]->wordIndices[wordId + 1]; 
                          tokenId ++){

                tmpPhiTopic[doc.docChunkVec[chunkId]->wordTopics[tokenId]] ++;
            }
        }
    }

    //Step 2: validate
    for(int i = 0;i < 1024;i++){
        if(tmpPhiTopic[i] != hostPhiTopic[0][i]){
            printf("ValidPhi Error 2: topic(%d), tmpPhiTopic(%d), hostPhiTopic[i](%d)\n", 
                        i, tmpPhiTopic[i], hostPhiTopic[i]);
            exit(0);
        }
    }
    
    printf("Validate Phi passed ...\n");
    printf("ValidatePhi   time:%.2fs\n", (clock() - clockStart)/(double)CLOCKS_PER_SEC);
    printf("max phi:%d\n", maxPhi);
}


/*
class FreqTuple{

public:
    int id;
    int count;

    FreqTuple(int arg1, int arg2):id(arg1),count(arg2){}
    friend bool operator<(const FreqTuple &left ,const FreqTuple &right){return left.count > right.count;}
};

void ModelPhi::saveWord(std::string wordFileName, Vocabulary &vocab)
{

    printf("Saving word model ...\n");
    ofstream wordFileStream(wordFileName.c_str(), ios::out);

    wordFileStream << "topic, total, top words" << endl;
    for(int kite = 0; kite < k; kite ++){

        int totalCount = 0;
        vector<FreqTuple> wordVec;

        for(int wordId = 0; wordId < numWords; wordId ++){

            int tmpCount = hostPhiTopicWord[wordId*k + kite];
            if(tmpCount <= 0)continue;

            totalCount += tmpCount;
            wordVec.push_back(FreqTuple(wordId, tmpCount));
        }

        sort(wordVec.begin(), wordVec.end());

        //output
        if(totalCount <= 0)continue;

        wordFileStream.width(4);
        wordFileStream << kite << " " << totalCount << " ";
        for(int i = 0;i < wordVec.size(); i ++){
            wordFileStream << "(" << vocab.getWordById(wordVec[i].id) << ",";
            wordFileStream << wordVec[i].count << ") ";
        }
        wordFileStream << endl;
    }
}
*/


struct pthreadArgShort
{
    PHITYPE *matrixA;
    PHITYPE *matrixB;
    int idxStart;
    int idxEnd;
    int matrixSize;
};

struct pthreadArgInt
{
    int *matrixA;
    int *matrixB;
    int idxStart;
    int idxEnd;
    int matrixSize;
};



static void *PhiReduceThreadShort(void *arg)
{
    pthreadArgShort *localArg = (pthreadArgShort*)arg;

    int startIdx = localArg->idxStart;
    int endIdx   = localArg->idxEnd;
    if(endIdx >= localArg->matrixSize)
        endIdx = localArg->matrixSize;

    PHITYPE *matrixA = localArg->matrixA;
    PHITYPE *matrixB = localArg->matrixB;
    for(int i = startIdx; i < endIdx; i ++)
        matrixA[i] += matrixB[i];
}

static void *PhiReduceThreadInt(void *arg)
{
    pthreadArgInt *localArg = (pthreadArgInt*)arg;

    int startIdx = localArg->idxStart;
    int endIdx   = localArg->idxEnd;
    if(endIdx >= localArg->matrixSize)
        endIdx = localArg->matrixSize;

    int *matrixA = localArg->matrixA;
    int *matrixB = localArg->matrixB;
    for(int i = startIdx; i < endIdx; i ++)
        matrixA[i] += matrixB[i];
}

static void PhiReduceShort(PHITYPE *matrixA, PHITYPE *matrixB, int matrixSize)
{


    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    const int numThreads = 48;
    pthread_t threads[numThreads];
    pthreadArgShort threadArgs[numThreads];
    int perThreadSize = (matrixSize + numThreads - 1)/numThreads;

    //launch
    for(int threadId = 0; threadId < numThreads; threadId ++){
        threadArgs[threadId].matrixA    = matrixA;
        threadArgs[threadId].matrixB    = matrixB;
        threadArgs[threadId].idxStart   = perThreadSize*threadId;
        threadArgs[threadId].idxEnd     = perThreadSize*threadId + perThreadSize;
        threadArgs[threadId].matrixSize = matrixSize;

        pthread_create(&(threads[threadId]), 
                       NULL, 
                       PhiReduceThreadShort,
                       (void*)(&(threadArgs[threadId])));
    }

    //join
    for(int threadId = 0; threadId < numThreads; threadId ++)
        pthread_join(threads[threadId], NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("PhiReduceShort: %.8lfs\n",elapsed);

}

static void PhiReduceInt(int *matrixA, int *matrixB, int matrixSize)
{


    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    const int numThreads = 16;
    pthread_t threads[numThreads];
    pthreadArgInt threadArgs[numThreads];
    int perThreadSize = (matrixSize + numThreads - 1)/numThreads;

    //launch
    for(int threadId = 0; threadId < numThreads; threadId ++){
        threadArgs[threadId].matrixA    = matrixA;
        threadArgs[threadId].matrixB    = matrixB;
        threadArgs[threadId].idxStart   = perThreadSize*threadId;
        threadArgs[threadId].idxEnd     = perThreadSize*threadId + perThreadSize;
        threadArgs[threadId].matrixSize = matrixSize;

        pthread_create(&(threads[threadId]), 
                       NULL, 
                       PhiReduceThreadInt,
                       (void*)(&(threadArgs[threadId])));
    }

    //join
    for(int threadId = 0; threadId < numThreads; threadId ++)
        pthread_join(threads[threadId], NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("PhiReduceInt:    %.8lfs\n",elapsed);

}

void ModelPhi::reduceCPU()
{
    for(int i = 1;i < numChunks;i++){
        PhiReduceShort(hostPhiTopicWordShort[0], hostPhiTopicWordShort[i], numWords*k);
        PhiReduceInt(hostPhiTopic[0],          hostPhiTopic[i],          k);
    }
}




