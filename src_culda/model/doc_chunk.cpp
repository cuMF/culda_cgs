
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <time.h>

#include "doc_chunk.h"

using namespace std;

DocChunk::DocChunk():
    wordIndices(NULL),
    slotIdToWordId(NULL),
    slotIndices(NULL),
    wordTokens(NULL),
    wordTopics(NULL),
    wordPerplexity(NULL),
    deviceWordIndices(NULL),
    deviceSlotIdToWordId(NULL),
    deviceSlotIndices(NULL),
    deviceWordTokens(NULL),
    deviceWordTopics(NULL),
    deviceWordPerplexity(NULL),
    deviceWordPerplexityMid(NULL),
    docRevIndices(NULL),
    docRevIdx(NULL),
    deviceDocRevIndices(NULL),
    deviceDocRevIdx(NULL)
{   
}

DocChunk::DocChunk(int argChunkId, int argDocIdStart, int argDocIdEnd, int argNumDocs, int argNumChunks):
    wordIndices(NULL),
    slotIdToWordId(NULL),
    slotIndices(NULL),
    wordTokens(NULL),
    wordTopics(NULL),
    wordPerplexity(NULL),
    deviceWordIndices(NULL),
    deviceSlotIdToWordId(NULL),
    deviceSlotIndices(NULL),
    deviceWordTokens(NULL),
    deviceWordTopics(NULL),
    deviceWordPerplexity(NULL),
    deviceWordPerplexityMid(NULL),
    docRevIndices(NULL),
    docRevIdx(NULL),
    deviceDocRevIndices(NULL),
    deviceDocRevIdx(NULL)
{   

    chunkId      = argChunkId;
    docIdStart   = argDocIdStart;
    docIdEnd     = argDocIdEnd;
    chunkNumDocs = docIdEnd - docIdStart;
    numDocs      = argNumDocs;
    numChunks    = argNumChunks;
}

struct pthreadArgTheta
{
    int numWords;
    int docStart;
    int docEnd;
    long long *wordIndices;
    int *wordTokens;

    vector<vector<long long> > *tmpDocPtr;
};

static void *ThetaDocReverse(void *arg)
{
    pthreadArgTheta *localArg = (pthreadArgTheta*)arg;

    for(int wordId = 0; wordId < localArg->numWords; wordId ++){

        for(long long tokenId = localArg->wordIndices[wordId];
                      tokenId < localArg->wordIndices[wordId + 1];
                      tokenId ++){

            int tmpDocId = localArg->wordTokens[tokenId];
            if( tmpDocId >= localArg->docStart && tmpDocId < localArg->docEnd)
                ((*(localArg->tmpDocPtr))[tmpDocId]).push_back(tokenId);
        }
    }
}


class SortClass
{
public:
    int wordId;
    int wordLen;
    SortClass(int a, int b):wordId(a),wordLen(b){}

    friend bool operator<(const SortClass &a, const SortClass &b)
    {
        return a.wordLen > b.wordLen;
    }
};

void DocChunk::loadChunk(string wordIdxFileName, string wordDataFileName, int *docLength)
{

    printf("    loading chunk %d ...\n", chunkId);
    clock_t clockStart = clock();
    /* load wordIndices & wordLength*/
    ifstream wordIdxStream(wordIdxFileName.c_str(), ios::in);
    if(!wordIdxStream.is_open()){
        cout << "File " << wordIdxFileName << " open failed" << endl;
        exit(0);
    }

    numWords = 0;
    long long wordId, endIdx;
    vector<long long>wordIndicesVec;
    wordIndicesVec.push_back(0);
    while(wordIdxStream >> wordId >> endIdx)
        wordIndicesVec.push_back(endIdx);
    
    numWords = wordIndicesVec.size() - 1;
    wordIdxStream.close();

    vector<int> wordLengthVec;
    wordIndices = new long long[numWords + 1];

    for(int i = 0;i < numWords; i++)
        wordLengthVec.push_back(wordIndicesVec[i + 1] - wordIndicesVec[i]);
    for(int i = 0;i < numWords + 1; i++)
        wordIndices[i] = wordIndicesVec[i];

    /* load token number */
    long long wordDataFileSize;
    ifstream wordDataStream(wordDataFileName.c_str(), ios::in|ios::ate);
    if(!wordDataStream.is_open()){
        cout << "File " << wordDataFileName << " open failed" << endl;
        exit(0);
    }
    wordDataFileSize = wordDataStream.tellg();
    chunkNumTokens = wordDataFileSize/sizeof(int);
    wordDataStream.close();

    /* sort words & slice words into slots when necessary */
    vector<SortClass> sortVec;
    for(int i = 0;i < numWords;i++)
        sortVec.push_back(SortClass(i, wordLengthVec[i]));
    sort(sortVec.begin(), sortVec.end());

    for(int i = 0;i < 10;i++)
        printf("i:%d, wordId:%d, len:%d\n", i, sortVec[i].wordId, sortVec[i].wordLen);

    int aveTokens = chunkNumTokens/NumConWorkers;
    printf("aveTokens:%d\n", aveTokens);

    vector<int>       slotToWordVec;
    vector<long long> slotIndicesVec;    
    numSlots = 0;
    for(int i = 0; i < numWords; i ++){
        //printf("%d\n",i);
        int tmpWordId  = sortVec[i].wordId;
        int tmpWordLen = sortVec[i].wordLen;

        if(tmpWordLen > 1.05*aveTokens){

            int tmpNumsSlices = (tmpWordLen + aveTokens - 1)/aveTokens;
            int tmpSliceSize  = (tmpWordLen + tmpNumsSlices - 1)/tmpNumsSlices;

            for(int i = 0;i < tmpNumsSlices; i ++){

                //printf("i:%d\n", i);
                long long tmpStartIdx = wordIndices[tmpWordId] + tmpSliceSize*i;
                long long tmpEndIdx   = tmpStartIdx + tmpSliceSize;  
                if(tmpEndIdx >= wordIndices[tmpWordId + 1])
                    tmpEndIdx = wordIndices[tmpWordId + 1];

                slotToWordVec.push_back(tmpWordId);
                slotIndicesVec.push_back(tmpStartIdx);
                slotIndicesVec.push_back(tmpEndIdx);
                numSlots ++;
            }

        }
        else{
            slotToWordVec.push_back(tmpWordId);
            slotIndicesVec.push_back(wordIndices[tmpWordId]);
            slotIndicesVec.push_back(wordIndices[tmpWordId + 1]);
            numSlots ++;
        }

    }

    for(int i = 0;i < 11;i ++)printf("%d\n",wordIndicesVec[i]);
    for(int i = 0;i < 10;i++)
        printf("i:%d, wordId:%d, len:%d, start:%lld, end:%lld\n", 
            i, sortVec[i].wordId, sortVec[i].wordLen, wordIndices[sortVec[i].wordId] , wordIndices[sortVec[i].wordId + 1]);

    for(int i = 0;i < 10; i++)
        printf("slot:%d, wordId:%d, len:%d, start:%lld, end:%lld\n", i, slotToWordVec[i], slotIndicesVec[i*2 + 1] - slotIndicesVec[i*2], slotIndicesVec[i*2], slotIndicesVec[i*2+1]);

    printf("numSlots:%d\n", numSlots);
    
    slotIdToWordId   = new int[numSlots];
    slotIndices      = new long long[numSlots*2];

    for(int i = 0;i < numSlots; i++){
        slotIdToWordId[i]    = slotToWordVec[i];
        slotIndices[i*2]     = slotIndicesVec[i*2];
        slotIndices[i*2 + 1] = slotIndicesVec[i*2 + 1]; 
    }

    /* load tokens */
    if(wordTokens != NULL)delete []wordTokens;
    if(wordTopics != NULL)delete []wordTopics;
    wordTokens = new int[chunkNumTokens];
    wordTopics = new short[chunkNumTokens];

    wordDataStream.open(wordDataFileName.c_str(), ios::in);
    for(long long i = 0;i < chunkNumTokens;i++)
        wordDataStream.read((char*)(&(wordTokens[i])), sizeof(int));

    //generate doc reverse info
    if(docRevIndices != NULL)delete []docRevIndices;
    if(docRevIdx     != NULL)delete []docRevIdx;
    docRevIndices = new long long[numDocs + 1]();
    docRevIdx     = new TokenIdxType[chunkNumTokens]();

    for(int docId = 0, offset = 0; docId < numDocs; docId ++){
        if(docId >= docIdStart && docId < docIdEnd){
            docRevIndices[docId]     = offset;
            docRevIndices[docId + 1] = offset + docLength[docId];
            offset += docLength[docId];
        }
        else{
            docRevIndices[docId]     = offset;
            docRevIndices[docId + 1] = offset;
        }
    }
    vector<int> tmpDocPtr;
    for(int docId = 0; docId < numDocs;docId ++)
        tmpDocPtr.push_back(docRevIndices[docId]);

    for(int wordId = 0;wordId < numWords; wordId ++){
        for(long long tokenId = wordIndices[wordId]; 
                      tokenId < wordIndices[wordId + 1]; 
                      tokenId ++){

            int tmpDocId = wordTokens[tokenId];
            docRevIdx[tmpDocPtr[tmpDocId]] = int(tokenId);
            tmpDocPtr[tmpDocId] ++;
        }
    }
}


void DocChunk::generateTopics(int k){

    srand (time(NULL));
    for(long long i = 0; i < chunkNumTokens;i++)
        wordTopics[i] = short(rand()%k);
}

struct wordStruct{   
    int wordId;
    int numTokens;

    wordStruct(int arg1, int arg2){
        wordId = arg1;
        numTokens = arg2;
    }
};


void DocChunk::allocGPU(int GPUid)
{
    cudaSetDevice(GPUid);
    
    if(deviceWordIndices       != NULL)cudaFree(deviceWordIndices);
    if(deviceSlotIdToWordId    != NULL)cudaFree(deviceSlotIdToWordId);
    if(deviceSlotIndices       != NULL)cudaFree(deviceSlotIndices);
    if(deviceWordTokens        != NULL)cudaFree(deviceWordTokens);
    if(deviceWordTopics        != NULL)cudaFree(deviceWordTopics);
    if(deviceWordPerplexity    != NULL)cudaFree(deviceWordPerplexity);
    if(deviceWordPerplexityMid != NULL)cudaFree(deviceWordPerplexityMid);

    if(deviceDocRevIndices     != NULL)cudaFree(deviceDocRevIndices);
    if(deviceDocRevIdx         != NULL)cudaFree(deviceDocRevIdx);
    
    cudaMalloc((void**)&deviceWordIndices,       (numWords + 1)*sizeof(long long));
    cudaMalloc((void**)&deviceSlotIdToWordId,    numSlots*sizeof(int));
    cudaMalloc((void**)&deviceSlotIndices,       numSlots*2*sizeof(long long));

    cudaMalloc((void**)&deviceWordTokens,        chunkNumTokens*sizeof(int));
    cudaMalloc((void**)&deviceWordTopics,        chunkNumTokens*sizeof(short));

    cudaMalloc((void**)&deviceWordPerplexity,    numWords*(TrainBlockSize/32)*sizeof(double));
    cudaMalloc((void**)&deviceWordPerplexityMid, ReduceParameter*sizeof(double));

    gpuErr(cudaPeekAtLastError());

    cudaMalloc((void**)&deviceDocRevIndices,     (numDocs+1)*sizeof(long long));
    cudaMalloc((void**)&deviceDocRevIdx,         chunkNumTokens*sizeof(TokenIdxType));

    long long totalByte = (numWords + 1)*sizeof(long long) + 
                          chunkNumTokens*sizeof(int) +
                          chunkNumTokens*sizeof(short) +
                          numWords*sizeof(double) + 
                          ReduceParameter*sizeof(double) +
                          (numDocs + 1)*sizeof(long long) +
                          chunkNumTokens*sizeof(TokenIdxType);
    printf("docChunk size:%.3f GB\n", totalByte/(1024.0*1024.0*1024.0));

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
}

void DocChunk::toGPU()
{

    //tokens
    cudaMemcpy(deviceWordIndices, 
               wordIndices, 
               sizeof(long long)*(numWords + 1), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSlotIdToWordId,
               slotIdToWordId,
               sizeof(int)*numSlots,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSlotIndices,
               slotIndices,
               sizeof(long long)*numSlots*2,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceWordTokens,
               wordTokens,
               sizeof(int)*chunkNumTokens,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceWordTopics,
               wordTopics,
               sizeof(short)*chunkNumTokens,
               cudaMemcpyHostToDevice);


    //doc rev data
    cudaMemcpy(deviceDocRevIndices,
               docRevIndices,
               sizeof(long long)*(numDocs + 1),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDocRevIdx,
               docRevIdx,
               sizeof(TokenIdxType)*chunkNumTokens,
               cudaMemcpyHostToDevice);

}

void DocChunk::toCPU()
{
    printf("DocChunk::toCPU() ChunkId:%d...\n", chunkId);
    cudaMemcpy(wordTopics,
               deviceWordTopics,
               sizeof(short)*chunkNumTokens,
               cudaMemcpyDeviceToHost);
    
    printf("finished DocChunk::toCPU() ...\n");
}
