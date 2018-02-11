#ifndef _GPU_THETA_CHUNK_H_

#define _GPU_THETA_CHUNK_H_

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

using namespace std;

class ModelThetaChunk
{
public:
    int k;
    int numDocs;
    int numWords;
    int numChunks;

    int chunkId;
    int docIdStart;
    int docIdEnd;
    int chunkNumDocs;

    int chunkNNZ;

    //GPU data
    short *deviceThetaA;     //chunkNNZ
    short *deviceThetaJA;    //chunkNNZ
    int   *deviceThetaCurIA; //numDocs
    int   *deviceThetaMaxIA; //numDocs + 1
    int   *deviceDenseTheta; //chunkNumDocs + 1

    //CPU data
    short *hostThetaA;     //chunkNNZ
    short *hostThetaJA;    //chunkNNZ
    int   *hostThetaCurIA; //numDocs
    int   *hostThetaMaxIA; //numDocs + 1

    ModelThetaChunk();
    ModelThetaChunk(int, int, int, int, int, int, int, int, int);


    void InitData(const vector<int> &);
    void toGPU();
    void toCPU();
    void UpdateThetaGPU(Document &, cudaStream_t stream=0);
    void validTheta(Document&);

    void clearPtr();
};

#endif