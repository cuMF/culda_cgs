#ifndef _GPU_THETA_H_

#define _GPU_THETA_H_


#include <vector>
#include <string>
#include <queue>
#include <set>

#include <cuda_runtime_api.h>
#include "culda_argument.h"
#include "vocab.h"
#include "doc.h"

#include "../kernel/lda_train_kernel.h"
#include "../kernel/lda_theta_kernel.h"
#include "model_theta_chunk.h"

using namespace std;

class ModelTheta
{
public:
	int k;
	int numDocs;
	int numWords;
	int numChunks;

	int thetaNNZ;

	vector<ModelThetaChunk*> thetaChunkVec;

	ModelTheta();
	ModelTheta(int argK, int argDocs, int argWords, int argNumChunks);

	void InitData(Document&);
    void validTheta(Document&);
    void UpdateThetaGPU(Document &doc, cudaStream_t *stream=NULL);
	void clearPtr();

	~ModelTheta(){ clearPtr();}

	void toGPU();
	void toCPU();

	//float countZero();
	//float countIA();
	
	//void saveTheta(string fileName);
	//void saveDoc(std::string docFileName);

};


#endif