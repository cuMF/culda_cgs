#ifndef _DATA_CHUNK_H_
#define _DATA_CHUNK_H_

#include <sstream>
class DataChunk{

public:
    int numWords;
    int numChunks;
    int chunkId;

    int docIdStart;
    int docIdEnd;

    int chunkDocSize;
    long long chunkTokenSize;

    string outFilePrefix;
    string outFileIdxName;
    string outFileDataName;

    vector<vector<int> > wordFirstVec;

    DataChunk(int argNumWords, int argNumChunks, int argChunkId, string argFilePrefix)
    {
        numWords = argNumWords;
        numChunks = argNumChunks;
        chunkId = argChunkId;
        outFilePrefix = argFilePrefix;

        for(int i = 0;i < numWords;i++)
            wordFirstVec.push_back(vector<int>());
        
        stringstream tmpNameStream;
        outFileIdxName  = outFilePrefix + ".word.idx";
        outFileDataName = outFilePrefix + ".word.data";
        if(numChunks > 1){
            tmpNameStream << outFileIdxName << chunkId;
            tmpNameStream >> outFileIdxName;
            tmpNameStream.clear();
            tmpNameStream << outFileDataName << chunkId;
            tmpNameStream >> outFileDataName;   
        }

        docIdStart = 0;
        docIdEnd = 0;
    }

    void writeChunk()
    {
        ofstream outFileIdxStream(outFileIdxName.c_str(), ios::out);
        ofstream outFileDataStream(outFileDataName.c_str(), ios::out|ios::binary);

        long long offset = 0;
        for(int wordId = 0; wordId < wordFirstVec.size();wordId++){
            offset += wordFirstVec[wordId].size();
            outFileIdxStream << wordId << " " << offset << endl;

            for(int localTokenId = 0; localTokenId < wordFirstVec[wordId].size(); localTokenId ++){
                int tmpWord = wordFirstVec[wordId][localTokenId];
                outFileDataStream.write((char*)&(tmpWord), sizeof(int));
            }
        }

        outFileIdxStream.close();
        outFileDataStream.close();
    }
    ~DataChunk()
    {
    }

};

#endif