

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
#include "model_theta.h"
#include "vocab.h"



/* Implementations of class ModelTheta */

ModelTheta::ModelTheta():
               k(0),
               numDocs(0),
               numWords(0),
               numChunks(0),
               thetaNNZ(0)
{
    clearPtr();
}

ModelTheta::ModelTheta(int argK,int argDocs, int argWords, int argNumChunks):
              k(argK),
              numDocs(argDocs),
              numWords(argWords),
              numChunks(argNumChunks),
              thetaNNZ(0)
{
    clearPtr();
}



void ModelTheta::InitData(Document &doc)
{
    
    clearPtr();

    vector<int> docLenVec;
    for(int docId = 0; docId < numDocs; docId ++){
        int tmpLen = doc.docLength[docId];
        if(tmpLen >= k)tmpLen = k;

        tmpLen = ((tmpLen + 31)/32)*32;
        docLenVec.push_back(tmpLen);
    }

    //chunk by chunk
    for(int chunkId = 0; chunkId < numChunks; chunkId ++){
        int tmpChunkNNZ = 0;

        for(int docId = doc.docChunkVec[chunkId]->docIdStart; 
                docId < doc.docChunkVec[chunkId]->docIdEnd; 
                docId ++)
            tmpChunkNNZ += docLenVec[docId];

        ModelThetaChunk *tmpPtr = new ModelThetaChunk(
            k,
            doc.numDocs,
            doc.numWords,
            doc.numChunks,
            chunkId,
            doc.docChunkVec[chunkId]->docIdStart,
            doc.docChunkVec[chunkId]->docIdEnd,
            doc.docChunkVec[chunkId]->docIdEnd - doc.docChunkVec[chunkId]->docIdStart,
            tmpChunkNNZ);

        //printf("chunkId:%d, tmpChunkNNZ:%d\n", chunkId, tmpChunkNNZ);

        tmpPtr->InitData(docLenVec);
        thetaChunkVec.push_back(tmpPtr);
    }

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
}

void ModelTheta::UpdateThetaGPU(Document &doc, cudaStream_t *stream)
{
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    
    if(stream != NULL){
        for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++)
            thetaChunkVec[chunkId]->UpdateThetaGPU(doc, stream[chunkId]);
    }
    else{
        for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++)
            thetaChunkVec[chunkId]->UpdateThetaGPU(doc);
    }
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
}

void ModelTheta::toGPU()
{
    for(int chunkId = 0; chunkId < numChunks; chunkId ++)
        thetaChunkVec[chunkId]->toGPU();
}

void ModelTheta::toCPU()
{
    for(int chunkId = 0; chunkId < numChunks; chunkId ++)
        thetaChunkVec[chunkId]->toCPU();

}

void ModelTheta::clearPtr()
{
    for(int i = 0;i < thetaChunkVec.size(); i++)
        delete thetaChunkVec[i];
}


/*
float ModelTheta::countZero()
{
    toCPU();
    long long totalEntry = 0;
    long long zeroEntry = 0;
    for(int i = 0;i < numDocs;i++){
        for(int j = hostThetaMaxIA[i];j < hostThetaMaxIA[i+1];j++){
            totalEntry++;
            if(hostThetaA[j] == 0)zeroEntry ++;
        }
    }
    return 1.0*zeroEntry/totalEntry;
}
*/

/*
float ModelTheta::countIA()
{
    toCPU();

    long long total = 0;
    long long cur = 0;
    for(int i = 0;i < numDocs;i++){
        int startIdx = hostThetaMaxIA[i];
        int maxEndIdx = hostThetaMaxIA[i+1];
        int curEndIdx = hostThetaCurIA[i];
        total += maxEndIdx - startIdx;
        cur += curEndIdx - startIdx;
    }

    return 1.0*cur/total;
}
*/


/*
void ModelTheta::saveTheta(string fileName)
{   
    
    printf("Saving theta ...\n");

    ofstream thetaStream(fileName.c_str(), ios::out);
    int tmpNumDocs = numDocs;
    tmpNumDocs = 1000;  
    for(int i = 0;i < tmpNumDocs;i++){
        thetaStream << "Doc id:" << i << ", ";
        thetaStream << "len:"    << hostThetaMaxIA[i+1] - hostThetaMaxIA[i] << ", ";
        thetaStream << "MaxIA:"  << hostThetaMaxIA[i] << " - " << hostThetaMaxIA[i + 1] << ", ";
        thetaStream << "CurIA:"  << hostThetaCurIA[i];
        thetaStream << endl;
        
        int totalA = 0;
        for(int j = hostThetaMaxIA[i]; j < hostThetaMaxIA[i+1] ; j++){

            thetaStream << "IA(" << j << "), " 
                        << "JA(" << hostThetaJA[j] << "), " 
                        << "A("  << hostThetaA[j] << ")\n"; 
            totalA += hostThetaA[j];
        }
        thetaStream << "total: " << totalA << "\n";
    }
}
*/


void ModelTheta::validTheta(Document &doc)
{
    for(int chunkId = 0; chunkId < numChunks; chunkId ++)
        thetaChunkVec[chunkId]->validTheta(doc);
}

class FreqTuple{

public:
    int id;
    int count;

    FreqTuple(int arg1, int arg2):id(arg1),count(arg2){}
    friend bool operator<(const FreqTuple &left ,const FreqTuple &right){return left.count > right.count;}
};

/*
void ModelTheta::saveDoc(std::string docFileName)
{
    ofstream docFileStream(docFileName.c_str(), ios::out);

    printf("Saving doc model ...\n");
    docFileStream << "Doc, total, top topics" << endl;

    for(int docId = 0; docId < numDocs; docId ++){

        int totalCount = 0;
        vector<FreqTuple> topicVec;
        for(int tmpIdx = hostThetaMaxIA[docId]; tmpIdx < hostThetaMaxIA[docId + 1]; tmpIdx ++){

            int tmpCount = hostThetaA[tmpIdx];
            int tmpK     = hostThetaJA[tmpIdx];
            if(tmpCount == 0)continue;

            totalCount += tmpCount;
            topicVec.push_back(FreqTuple(tmpK, tmpCount));
        }

        sort(topicVec.begin(), topicVec.end());

        //output
        docFileStream.width(4);
        docFileStream << docId << "| " << totalCount << " ";
        for(int i = 0;i < topicVec.size();i++){
            docFileStream << "(" << topicVec[i].id << ",";
            docFileStream << topicVec[i].count << "), ";
        }
        docFileStream << endl;
    }

}
*/





