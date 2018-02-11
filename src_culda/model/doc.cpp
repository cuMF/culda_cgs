

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>

#include <sstream>

#include "doc.h"

using namespace std;
//Definitions of Class Document's methods.


Document::Document():
	numDocs(0),
	numWords(0),
	numTokens(0),
    docIndices(NULL),
    docLength(NULL)
{
}

Document::Document(const string &filePrefix, int argNumChunks):
	numDocs(0),
	numWords(0),
	numTokens(0),
	docIndices(NULL),
	docLength(NULL)
{
	numChunks = argNumChunks;
	loadDocument(filePrefix, argNumChunks);
}

Document::Document(const Document &doc):
	numDocs(0),
	numWords(0),
	numTokens(0),
	docIndices(NULL),
	docLength(NULL)
{

	//TBD!!!
	/*
	numWords = doc.numWords;
	numTokens = doc.numTokens;
	ids = doc.ids;

	tokens = new int[numTokens];
	indices = new long long[numWords*2];

	copy(doc.tokens, doc.tokens + numTokens, tokens);
	copy(doc.indices, doc.indices + numWords*2, indices);
	*/
}


void Document::loadDocument(const string &filePrefix, int argNumChunks)
{
	numChunks = argNumChunks;

	/* Load docIndices and docLength*/
	string docIdxFileName = filePrefix + ".doc.idx";
	ifstream docIdxStream(docIdxFileName.c_str(), ios::in);
	if(!docIdxStream.is_open()){
		cout << "File " << docIdxFileName << " open failed" << endl;
		exit(0);
	}

	numDocs = 0;
	int docId;
	long long endIdx;
	vector<long long> docIndicesVec;
	docIndicesVec.push_back(0);
	while(docIdxStream >> docId >> endIdx)
	{	
		numDocs ++;
		docIndicesVec.push_back(endIdx);
	}
	docIdxStream.close();

	docIndices = new long long[numDocs + 1];
	docLength  = new int[numDocs];

	for(int i = 0;i < numDocs; i++)
		docLength[i] = docIndicesVec[i + 1] - docIndicesVec[i];

	for(int i = 0;i < numDocs + 1; i++)
		docIndices[i] = docIndicesVec[i];
	
	docIndicesVec.clear();

	//load .chunk meta data.
	vector<int> chunkDocVec;
	ifstream chunkFileStream((filePrefix + ".chunk").c_str(), ios::in);
	if(!chunkFileStream.is_open()){
		cout << "File " << filePrefix << ".chunk open failed" << endl;
		exit(0);
	}
	chunkDocVec.push_back(0);
	int tmp1, tmp2;
	while(chunkFileStream >> tmp1 >> tmp2){
		chunkDocVec.push_back(tmp2);	
	}
	chunkFileStream.close();

	if(chunkDocVec.size() != (numChunks + 1)){
		printf("Error: numChunks(%d) does not match the chunk file\n",numChunks);
		exit(0);
	}

	//load data
	for(int chunkId = 0;chunkId < numChunks; chunkId ++){
		docChunkVec.push_back(
			new DocChunk(
			chunkId, 
			chunkDocVec[chunkId], 
			chunkDocVec[chunkId + 1],
			numDocs, 
			numChunks));
	}

	for(int chunkId = 0; chunkId < numChunks; chunkId ++){

		stringstream tmpStream;
		string wordIdxFileName, wordDataFileName;
		if(numChunks == 1){
			tmpStream << filePrefix << ".word.idx";
			tmpStream >> wordIdxFileName;
			tmpStream.clear();
			tmpStream << filePrefix << ".word.data";
			tmpStream >> wordDataFileName;
		}
		else{
			tmpStream << filePrefix << ".word.idx" << chunkId;
			tmpStream >> wordIdxFileName;
			tmpStream.clear();
			tmpStream << filePrefix << ".word.data" << chunkId;
			tmpStream >> wordDataFileName;
		}
		docChunkVec[chunkId]->loadChunk(wordIdxFileName, wordDataFileName, docLength);

		printf("    chunk %d loaded ...\n", chunkId);
	}
	numWords = docChunkVec[0]->numWords;
	printf("\n");

	numTokens = 0;
	for(int chunkId = 0; chunkId < numChunks; chunkId ++)
		numTokens += docChunkVec[chunkId]->chunkNumTokens;

}

void Document::generateTopics(int k) //TBD: parallelization
{

	printf("Initialize the topic for tokens ...\n\n");
	for(int chunkId = 0; chunkId < numChunks; chunkId ++)
		docChunkVec[chunkId]->generateTopics(k);
}

void Document::clear()
{
	numDocs   = 0;
	numWords  = 0;
	numTokens = 0;

	for(int chunkId = 0; chunkId < numChunks;chunkId ++)
		delete docChunkVec[chunkId];

	if(docIndices  != NULL) delete []docIndices;
	if(docLength   != NULL) delete []docLength;

	docIndices  = NULL;
	docLength   = NULL;
}	

void Document::printDocumentAbbr()
{
	printf("numDocs  :  %d\n",numDocs);
	printf("numWords :  %d\n",numWords);
	printf("numTokens:  %lld\n",numTokens);

	printf("\n");	
}


void Document::printDocumentFull()
{
	printf("----doc info-----\n");
	printf("numDocs  :  %d\n",numDocs);
	printf("numWords :  %d\n",numWords);
	printf("numTokens:  %lld\n",numTokens);

	for(int chunkId = 0; chunkId < numChunks; chunkId ++){
		printf("**chunkId:%d\n", chunkId);
		printf("word range:\n");
		for(long long i = 0;i < numWords; i++)
			printf("word%2d %6lld - %lld\n",i, 
				docChunkVec[chunkId]->wordIndices[i], 
				docChunkVec[chunkId]->wordIndices[i+1]);

	printf("token list:\n");
	printf("tokenId, wordId, DocId, topics\n");
	for(long long wordId = 0;wordId < numWords; wordId++){

		long long start = docChunkVec[chunkId]->wordIndices[wordId];
		long long end   = docChunkVec[chunkId]->wordIndices[wordId + 1];
		for(long long tokenId = start; tokenId < end; tokenId ++)
			printf("%lld, %d, %d, %d\n",tokenId, wordId, 
				docChunkVec[chunkId]->wordTokens[tokenId], 
				docChunkVec[chunkId]->wordTopics[tokenId]);
	}
	}

	/*
	for(long long i = 0;i < numTokens;i++)
		printf("%4lld %lld %lld\n", i, wordTokens[i], wordTopics[i]);
	*/
	
}

void Document::saveTopics(string fileName)
{
	printf("Saving topics ...\n");
}



