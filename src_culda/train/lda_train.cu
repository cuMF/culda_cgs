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
#include <math.h>

#include <cuda_runtime_api.h>

#include "../model/vocab.h"
#include "../model/doc.h"
#include "lda_train.h"
#include "../kernel/lda_train_kernel.h"


struct pthreadArgTheta
{
    int id;
    Document *docPtr;
    ModelTheta *thetaPtr;
    cudaStream_t mainStream;
    cudaStream_t branStream;
};

static void *UpdateThetaThread(void *arg)
{
    pthreadArgTheta *localArg = (pthreadArgTheta*)arg;
    cudaStreamSynchronize(localArg->mainStream);
    localArg->thetaPtr->thetaChunkVec[localArg->id]->UpdateThetaGPU(*(localArg->docPtr), localArg->branStream);

    return NULL;
}

#include "SingleChunkSingleGPU.h"
#include "MultiChunkMultiGPUequal.h"
//#include "MultiChunkMultiGPUNotequal.h"

void LDATrain(Document &doc, Vocabulary &vocab, Argument &arg)
{
    if(arg.numGPUs > doc.numChunks) arg.numGPUs = doc.numChunks;

    //ModelPhi preparation.
    ModelPhi   modelPhi(arg.k, arg.numGPUs, doc.numDocs, doc.numWords, doc.numChunks);
    ModelTheta modelTheta(arg.k, doc.numDocs, doc.numWords, doc.numChunks);

    if(doc.numChunks == 1) //One chunk, one GPU.
        SingleChunkSingleGPU(doc, vocab, arg, modelPhi, modelTheta);
    else if(doc.numChunks != 1 && arg.numGPUs != 1 && arg.numGPUs == doc.numChunks)
        MultiChunkMultiGPUequal(doc, vocab, arg, modelPhi, modelTheta);
    
}
