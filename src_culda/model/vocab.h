#ifndef _VOCAB_H_

#define _VOCAB_H_

#include <vector>
#include <string>
#include <queue>
#include <set>

using namespace std;

class Word{
public:
	string token;
	int id;
	bool operator < (const Word & arg) const{
        return token.compare(arg.token) < 0;
    }
    bool operator > (const Word & arg) const{
        return token.compare(arg.token) > 0;
    }
    bool operator == (const Word & arg) const{
        return token.compare(arg.token) == 0;
    }

    Word(){}
    Word(string argString){ token = argString;}
    Word(string argString, int argId){ token = argString; id = argId;}
};


class Vocabulary
{
public:
	std::vector<Word> wordList;
	std::set<Word> wordSet;	

	Vocabulary(){}
	Vocabulary(const string &fname);

	void clear();
	bool hasWord(const string & token);
	void insertWord(const string & token);

	string getWordById(int argId);
	int    getIdByWord(string argToken);

	void loadVocab(string fname);
	void writeVocab(string fname);

	void printVocabAbbr();
	void printVocabFull();
};


#endif