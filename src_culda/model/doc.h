#ifndef _DOC_H_

#define _DOC_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "doc_chunk.h"

using namespace std;

/*
 * word-first format
 * fileName.word.data: docId list
 * fileName.word.idx: each line is consisted of: wordId, startIdx, endIdx;
 * fileName.vocab:    word-2-id mapping
*/



class Document
{

public:

	int        numDocs;
	int        numWords;
	long long  numTokens;
	int        numChunks;
    int        numWorkers;

	long long *docIndices;     // numDocs + 1
	int       *docLength;      // numDocs

	vector<DocChunk*> docChunkVec;

	Document();
	Document(const string &filePrefix, int argNumChunks);
	Document(const Document &doc);

	void loadDocument(const string &filePrefix, int argNumChunks);

	void clear();

	~Document(){ clear(); }

	void printDocumentAbbr();
	void printDocumentFull();
	void generateTopics(int k);
	void saveTopics(string fileName);

};


#endif