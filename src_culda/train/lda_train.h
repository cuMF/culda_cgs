
#ifndef _LDA_TRAIN_H_
#define _LDA_TRAIN_H_


#include "../model/model_theta.h"
#include "../model/model_phi.h"
#include "../model/culda_argument.h"
#include "../model/vocab.h"
#include "../model/doc.h"

#include "../kernel/lda_train_kernel.h"

void LDATrain(Document &doc, Vocabulary &vocab, Argument &argu);

double LDALikelihood(Argument&, Document&, ModelTheta&, ModelPhi&);

#endif