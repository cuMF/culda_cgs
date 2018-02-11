#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include <stdlib.h>

//#include "../src_culda/data_format.h"
#include "../src_culda/model/vocab.h"
#include "../src_culda/model/culda_argument.h"


#include "data_chunk.h"

using namespace std;


int main(int argc, char**argv)
{

	int numChunks = 0;
	if(argc != 3 && argc != 4){
		cout << "Usage:./format input output_prefix numChunks[default=1]" << endl;
		return 0;
	}
	if(argc == 4) numChunks = atoi(argv[3]);
	if(numChunks <= 1)numChunks = 1;

	//open file
	ifstream inputFile(argv[1], ios::in);

	if(!inputFile.is_open()){
		cout << argv[1] << " open failed" << endl;;
		exit(0);	
	}

	ofstream outFileVocab((argv[2] + string(".vocab")).c_str(), ios::out);
	ofstream outFileChunk((argv[2] + string(".chunk")).c_str(), ios::out);
	ofstream outFileDocIdx((argv[2] + string(".doc.idx")).c_str(), ios::out);

	if(!outFileVocab.is_open()){
		cout << argv[2] << ".vocab open failed" << endl;;
		exit(0);	
	}
	if(!outFileChunk.is_open()){
		cout << argv[2] << ".chunk open failed" << endl;;
		exit(0);	
	}
	if(!outFileDocIdx.is_open()){
		cout << argv[2] << ".doc.idx open failed" << endl;;
		exit(0);	
	}

	printf("reading input file ...\n");
	//read input file
	Vocabulary vocab;
	long long numTokens = 0;
	vector<vector<int> > wordFirstVec;
	vector<int> docLengthVec;
	int docId = 0;
	
	string docLine;
	while(getline(inputFile, docLine)){

		string docTitle, token;

		std::istringstream docStream(docLine);
		if(!(docStream >> docTitle))continue;

		int docLength = 0;
		while(docStream >> token){

			docLength ++;
			vocab.insertWord(token);
			int id = vocab.getIdByWord(token);		
			numTokens ++;

			//insert it to wordFirstVec
			if(wordFirstVec.size() >= id){

				int numLoops = id + 1 - wordFirstVec.size();
				for(int i = 0;i < numLoops; i++)
					wordFirstVec.push_back(vector<int>());
			}
			wordFirstVec[id].push_back(docId);
		}
		outFileDocIdx << docId << " " << numTokens << endl;

		docLengthVec.push_back(docLength);
		docId ++;
	}
	inputFile.close();

	printf("write vocabulary ...\n");
	//write vocabulary
	for(int i = 0;i < vocab.wordList.size();i++){
		outFileVocab << vocab.wordList[i].token << " " << vocab.wordList[i].id << endl;
	}
	outFileVocab.close();

	//cout << "wordnumbers:" << vocab.word_list.size() << endl;
	//cout << "wordFirstVec:" << wordFirstVec.size() << endl;

	printf("mapping chunks ...\n");
	//decide the doc -> chunk mapping
	long long tokenPerChunk = (numTokens + numChunks - 1)/numChunks;
	vector<DataChunk> dataChunkVec;
	vector<int> docToChunkVec;
	docId = 0;
	printf("numChunks:%lld\n", numChunks);
	printf("numTokens:%lld\n", numTokens);
	printf("perChunk :%lld\n", tokenPerChunk);
	for(int chunkId = 0; chunkId < numChunks; chunkId ++){
		dataChunkVec.push_back(DataChunk(vocab.wordList.size(), numChunks, chunkId, argv[2]));

		long long tmpChunkSize = 0;
		dataChunkVec[chunkId].docIdStart = docId;
		while(docId < docLengthVec.size()){
			tmpChunkSize += docLengthVec[docId];
			docToChunkVec.push_back(chunkId);
			docId ++;
			
			if(tmpChunkSize >= tokenPerChunk)break;
		}
		dataChunkVec[chunkId].docIdEnd = docId;
		dataChunkVec[chunkId].chunkTokenSize = tmpChunkSize;
		dataChunkVec[chunkId].chunkDocSize = 
			dataChunkVec[chunkId].docIdEnd - dataChunkVec[chunkId].docIdStart;

		outFileChunk << chunkId << " " << docId << endl;
	}

	for(int chunkId = 0;chunkId < numChunks; chunkId ++){
		printf("----\n");
		printf("chunkId:%d\n", chunkId);
		printf("numWords:%d\n", dataChunkVec[chunkId].numWords);
		printf("doc range:%d - %d\n", dataChunkVec[chunkId].docIdStart, dataChunkVec[chunkId].docIdEnd);
		printf("chunkSize:%lld\n", dataChunkVec[chunkId].chunkTokenSize);
		printf("%s\n", dataChunkVec[chunkId].outFileIdxName.c_str());
		printf("%s\n", dataChunkVec[chunkId].outFileDataName.c_str());
	}
	
	printf("chunk partitioning ...\n");	
	//distribute the data to each chunk
	long long offset = 0;
	for(int wordId = 0; wordId < wordFirstVec.size(); wordId ++){

		for(int localTokenId = 0; localTokenId < wordFirstVec[wordId].size(); localTokenId ++){

			int docId = wordFirstVec[wordId][localTokenId];
			int chunkId = docToChunkVec[docId];

			dataChunkVec[chunkId].wordFirstVec[wordId].push_back(docId);
		}
	}
	
	printf("write chunks ...\n");
	//write
	for(int chunkId = 0;chunkId < numChunks;chunkId++)
	{
		printf("writing chunk %d ...\n", chunkId);
		dataChunkVec[chunkId].writeChunk();
	}
	
	outFileChunk.close();

	return 0;
}