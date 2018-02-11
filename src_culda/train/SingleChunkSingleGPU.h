#ifndef _SingleChunkSingleGPU_H_
#define _SingleChunkSingleGPU_H_


void static SingleChunkSingleGPU(Document &doc, Vocabulary &vocab, Argument &arg,
                                 ModelPhi &modelPhi, ModelTheta &modelTheta)
{

    /* data preparation and transfer */

    printf("Call SingleChunkSingleGPU() ...\n");
    

    printf("alloc gpu for doc ...\n");
    doc.docChunkVec[0]->allocGPU(0);
    printf("to gpu for doc ...\n");
    doc.docChunkVec[0]->toGPU();

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    /* model phi */
    printf("Prepare model phi ...\n");
    modelPhi.InitData(doc);
    modelPhi.UpdatePhiGPU(doc, 0);
    modelPhi.UpdatePhiHead(arg.beta);
    //modelPhi.MasterGPUToCPU();
    //modelPhi.validPhi(doc);


    /* model theta */
    printf("Prepare model theta ...\n");
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    modelTheta.InitData(doc);

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    modelTheta.UpdateThetaGPU(doc);

    //exit(0);

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //modelTheta.toCPU();
    //modelTheta.validTheta(doc);

    //exit(0);

    
    /* prepare the randstate */
    int randStateSize = 256*20;
    curandState *deviceRandState[MaxNumGPU];
    cudaMalloc(&deviceRandState[0], sizeof(curandState)*randStateSize);
    initRandState<<<randStateSize/256, 256>>>(deviceRandState[0]);
    
    cudaStream_t extraStream;
    cudaStreamCreate(&extraStream);

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    struct timespec begin, end;
    double elapsed = 0, stamp = 0;

    //launch train kernels
    for(int ite = 0;ite < arg.iteration; ite++)
    {
        clock_gettime(CLOCK_MONOTONIC, &begin);

        //numBlocks = 100;
        LDAKernelTrain<<<doc.docChunkVec[0]->numSlots, TrainBlockSize>>>(
            arg.k,
            arg.alpha,
            arg.beta,
            doc.numDocs,
            doc.numWords,
            doc.docChunkVec[0]->chunkNumTokens,
            doc.docChunkVec[0]->deviceWordIndices,
            doc.docChunkVec[0]->deviceSlotIdToWordId,
            doc.docChunkVec[0]->deviceSlotIndices,
            doc.docChunkVec[0]->deviceWordTokens,
            doc.docChunkVec[0]->deviceWordTopics,
            modelTheta.thetaChunkVec[0]->deviceThetaA,
            modelTheta.thetaChunkVec[0]->deviceThetaMaxIA,
            modelTheta.thetaChunkVec[0]->deviceThetaCurIA,
            modelTheta.thetaChunkVec[0]->deviceThetaJA,
            modelTheta.thetaChunkVec[0]->docIdStart,
            modelPhi.phiChunkVec[0]->devicePhiTopicWordShort,
            modelPhi.phiChunkVec[0]->devicePhiTopic,
            modelPhi.phiChunkVec[0]->devicePhiHead,
            deviceRandState[0],
            randStateSize, //arg.numWorkers,
            0,
            doc.docChunkVec[0]->deviceWordPerplexity,
            doc.docChunkVec[0]->deviceDocRevIndices
            );

        //cudaDeviceSynchronize();
        //gpuErr(cudaPeekAtLastError());

        double logLike = LDATrainPerplexity(doc);
        //cudaDeviceSynchronize();
        //gpuErr(cudaPeekAtLastError());

        //doc.docChunkVec[0]->toCPU();

        modelPhi.UpdatePhiGPU(doc, 0);
        modelPhi.UpdatePhiHead(arg.beta);
        //modelPhi.MasterGPUToCPU();
        //modelPhi.validPhi(doc);

        modelTheta.UpdateThetaGPU(doc);
        //modelTheta.toCPU();
        //modelTheta.validTheta(doc);

        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end);
        stamp = end.tv_sec - begin.tv_sec;
        stamp += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        elapsed += stamp;

        printf("Iteration %3d: %6.2f sec, %3.2f sec, logLikelyhood = %.8f, %5.3f M\n", ite+1,elapsed, stamp, logLike,  doc.numTokens/stamp/1000000);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

//        if((ite + 1)%30 == 0)sleep(120);

    }

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++)
        doc.docChunkVec[chunkId]->toCPU();
    printf("\n");

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    modelTheta.toCPU();
    //modelTheta.validTheta(doc);

    modelPhi.MasterGPUToCPU();
    //cudaDeviceSynchronize();
    //modelPhi.validPhi(doc);

    

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    //modelPhi.savePhi("phi.data");
}


#endif
