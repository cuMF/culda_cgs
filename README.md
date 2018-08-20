# CuLDA_CGS

CuLDA_CGS is GPU solution for CGS-based LDA sampling. It's efficient and is able to achieve 686M tokens/sec. To the best of our knowledge, it's the first LDA solution that support GPUs.


## Input Data Preparation
./src_format contantions a program to transform to text corpus to the input format of CuLDA_CGS. The transformed data format is more efficient for subsequent processing and partitioned to multiple chunks to support multi-GPU scaling.

Run Command "make" in the directory and use the following command to transform the data:

    ./format input output_prefix numChunks[default=1]

The input format of ./format is like:

    doc-name1 token1 token2 token3\n
    doc-name2 token4 token5 token6\n
    ...

Tokens are separated by space, documents are separated by line.
  
## Compile and Run CuLDA_CGS
Everything about CuLDA_CGS is in ./src_culda. It does not relies on any 3rd party denpendency. What you need is only a CUDA environment and a CUDA-enabled GPU. 

Before you run command "make" in the directory, remember to change CXX_FLAG to your targeted architecture and change CUDA_INSTALL_PATH to your CUDA directory. 

Then you can run ./culda for LDA sampling, the usage is:

    ./culda [options]
  
 Possible options<br />
 
    -g <numer of GPUs> <br />
    -k <topic number>: currently only support 1024<br />
    -t <number of iterations><br />
    -s <number of thread blocks>: it has been deprecated<br />
    -a <alpha>: 50/1024 for our tested data sets<br />
    -b <beta>: 0.01 for our tested data sets<br />
    -c <number of input data chunks>: must be equal with -g, and must be consistency with the specified chunk number in the data prepration stage<br />
    -i <input file name prefix>: Same with the output_prefix in the data preparation stage.<br />
    -o <output file name prefix>: It's not used now. Rewrite ModelPhi::savePhi and ModelTheta::saveTheta as you need it.<br />

CuLDA_CGS outputs the number of processed token per sec and the loglikelyhood after each iteration.
  
