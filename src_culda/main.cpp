

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

#include <cuda_runtime_api.h>

#include "./model/model_theta.h"
#include "./model/culda_argument.h"
#include "./model/vocab.h"
#include "./model/doc.h"

#include "./train/lda_train.h"


using namespace std;


bool ISNumerical(char *str)
{
    int c = 0;
    while(*str != '\0')
    {
        if(isdigit(*str))c++;
       	else return false;
        str++;
    }
    return c > 0;
}

Argument ParseArgument(int argc, char **argv)
{
	vector<string> args;
	for(int i = 0;i < argc; i++){
		args.push_back(string(argv[i]));
	}

	if(argc == 1)
		throw invalid_argument("No argument found");
	
	Argument argument;
	int i;

	for(i = 1;i < argc; i++){

		if(args[i].compare("-g") == 0){

			if((i + 1) >= argc)throw invalid_argument("need to specify a value after -g");
			if(!ISNumerical(argv[i+1]))
				throw invalid_argument("-k should be followed by a positive integer");
			argument.numGPUs = atoi(argv[i+1]);
			i++;
		}
		else if(args[i].compare("-k") == 0){

			if((i + 1) >= argc)throw invalid_argument("need to specify a value after -k");
			if(!ISNumerical(argv[i+1]))
				throw invalid_argument("-k should be followed by a positive integer");
			argument.k = atoi(argv[i+1]);

			//TBD: check k
			i++;
		}
		else if(args[i].compare("-t") == 0){
			if((i + 1) >= argc)throw invalid_argument("need to specify a value after -t");
			if(!ISNumerical(argv[i+1]))
				throw invalid_argument("-t should be followed by a positive integer");
			argument.iteration = atoi(argv[i+1]);

			//TBD: check t
			i++;
		}
		else if(args[i].compare("-s") == 0){
			if((i + 1) >= argc)throw invalid_argument("need to specify a value after -s");
			if(!ISNumerical(argv[i+1]))
				throw invalid_argument("-s should be followed by a positive integer");
			argument.numWorkers = atoi(argv[i+1]);
			
			i++;
		}
		else if(args[i].compare("-a") == 0){
			if((i + 1) >= argc)throw invalid_argument("need to specify a value after -a");
			
			argument.alpha = atof(argv[i+1]);

			//TBD: check it
			i++;
		}
		else if(args[i].compare("-b") == 0){
			if((i + 1) >= argc)throw invalid_argument("need to specify a value after -b");
			//if(!ISNumerical(argv[i+1]))
			//	throw invalid_argument("-b should be followed by a number");
			argument.beta = atof(argv[i+1]);

			//TBD: check it
			i++;
		}
		else if(args[i].compare("-c") == 0){
			if((i + 1) >= argc)throw invalid_argument("need to specify a value after -b");
			//if(!ISNumerical(argv[i+1]))
			//	throw invalid_argument("-b should be followed by a number");
			argument.numChunks = atoi(argv[i+1]);

			//TBD: check it
			i++;
		}
		else if(args[i].compare("-i") == 0){
			if((i + 1) >= argc)throw invalid_argument("need to specify a file name prefix after -i");
			argument.inputFilePrefix = args[i+1];
			i++;
		}
		else if(args[i].compare("-o") == 0){
			if((i + 1) >= argc)throw invalid_argument("need to specify a file name prefix after -o");
			argument.outputFilePrefix = args[i+1];
			i++;
		}
		else break;

	}


	//process k
	if(argument.k%32 != 0){
		printf("Warning: number of topics(k) has been rounded to multiples of 32.\n");
		argument.k = (argument.k + 31)/32*32;
	}


	if (argument.numWorkers <= 0){
		printf("Warning: wrong number of workers.\n");
		argument.numWorkers = 1;
	}
	
	//process output file names.
	argument.outputWordFileName = argument.outputFilePrefix + ".word.full.txt";
	argument.outputDocFileName = argument.outputFilePrefix + ".doc.full.txt";

	//GPU number
	int deviceCount = 1;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if(deviceCount >= MaxNumGPU)deviceCount = MaxNumGPU;
    
	if(argument.numGPUs <= 0)argument.numGPUs = 1;
	if(argument.numGPUs > deviceCount){
		printf("Warning: number of GPUs(%d) is larger than device count(%d), rounded to %d\n", argument.numGPUs,deviceCount,deviceCount);
		argument.numGPUs = deviceCount;
	}
	

	return argument;
}


int main(int argc, char**argv)
{
	clock_t clockStart;
	Argument argument;

	printf("Parsing arguments ...\n");
	try{
		argument = ParseArgument(argc, argv);
	}
	catch(invalid_argument &e){
		cout << "Error: " <<e.what() << endl;
		return 1;
	}
	argument.printArgument();
	
	//load files: word-first encoding
	printf("Loading vocabulary ...\n");
	Vocabulary vocab(argument.inputFilePrefix + ".vocab");

	Document doc;
	printf("Loading document ...\n");
	doc.loadDocument(argument.inputFilePrefix, argument.numChunks);
	doc.printDocumentAbbr();

	doc.generateTopics(argument.k);

	//computation
	LDATrain(doc, vocab, argument);

	return 0;
}
