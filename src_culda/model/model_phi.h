#ifndef _MODEL_PHI_H_

#define _MODEL_PHI_H_


#include <vector>
#include <string>
#include <queue>
#include <set>

#include <cuda_runtime_api.h>
#include "culda_argument.h"
#include "vocab.h"
#include "doc.h"

#include "../kernel/lda_train_kernel.h"
#include "../kernel/lda_phi_kernel.h"
#include "model_phi_gpu.h"

using namespace std;

class ModelPhi
{
public:
    int k;
    int numGPUs;
    int numDocs;
    int numWords;
    int numChunks;

    //cpu data
    PHITYPE *hostPhiTopicWordShort[MaxNumGPU];
    int     *hostPhiTopic[MaxNumGPU];
    
    vector<ModelPhiGPU*> phiChunkVec;

    ModelPhi();
    ModelPhi(int argK, int argGPU, int argDocs, int argWords, int numChunks);
        
    void InitData(Document&); 
    void UpdatePhiGPU(Document&, int chunkId, cudaStream_t s=0);
    void UpdatePhiHead(float beta,cudaStream_t *stream=NULL);

    void clearPtr();

    ~ModelPhi(){ clearPtr();}

    void MasterGPUCollect(int GPUid, cudaStream_t stream=0);
    void MasterGPUDistribute(int GPUid, cudaStream_t stream=0);
    void MasterGPUToCPU(cudaStream_t stream=0);
    void MasterGPUReduce(cudaStream_t stream=0);
    
    void reduceCPU();

    void validPhi(Document&);
    void savePhi(string fileName);
};


#endif