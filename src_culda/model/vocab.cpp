

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>

#include "vocab.h"


//Definitions of methods of class Voculary
void Vocabulary::clear()
{
	wordList.clear();
	wordSet.clear();
}

bool Vocabulary::hasWord(const string & token)
{
	return wordSet.find(Word(token,0)) != wordSet.end();
}

void Vocabulary::insertWord(const string & token)
{
	if(!hasWord(token)){
		Word tmpWord(token, wordList.size());
		wordList.push_back(tmpWord);
		wordSet.insert(tmpWord);
	}
}

string Vocabulary::getWordById(int argId)
{

	if(argId >= wordList.size())cout << "overflow:" << argId << endl;
	return wordList[argId].token;
}

int Vocabulary::getIdByWord(string argToken)
{
	set<Word>::iterator setIte = wordSet.find(argToken);
	if(setIte == wordSet.end())return -1;
	else return setIte->id;
}

void Vocabulary::loadVocab(string fileName)
{

	clear();
	ifstream inputFile(fileName.c_str(), ios::in);

	if(!inputFile.is_open()){
		cout << "Vocabulary file " << fileName << " open failed" << endl;
		exit(0);
	}

	string token;
	int id;
	while(inputFile >> token >> id){
		insertWord(token);
	}
}

void Vocabulary::writeVocab(string fileName)
{

	ofstream outputFile(fileName.c_str(), ios::out);

	if(!outputFile.is_open()){
		cout << "Vocabulary file " << fileName << " open failed" << endl;
		exit(0);
	}

	for(int i = 0;i < wordList.size(); i++)
		outputFile << wordList[i].token << " " << wordList[i].id << endl;

}

void Vocabulary::printVocabAbbr()
{
	printf("----vocab info-----\n");
	printf("numWords:%d\n", wordList.size());
	
}

void Vocabulary::printVocabFull()
{

	for(int i = 0;i < wordList.size();i++)
		cout << "(" << wordList[i].token << "," << wordList[i].id << ")" << endl;
	
}

Vocabulary::Vocabulary(const string &fname)
{
	loadVocab(fname);
}


