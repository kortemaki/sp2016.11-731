from pycnn import *
import random
import pdb, pickle, os
import numpy
from progressbar import ProgressBar, SimpleProgress

# number of training iterations
N_ITERS = 101

SRC_DIM = 50
TGT_DIM = 2*SRC_DIM

LAYERS = 1
INPUT_DIM = 50
HIDDEN_DIM = 50
VOCAB_SIZE = 0

from collections import defaultdict
from itertools import count
import sys
import util

class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM,
                 VOCAB_SIZE, builder=SimpleRNNBuilder):
        self.m = model
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        model.add_lookup_parameters("lookup", (VOCAB_SIZE, INPUT_DIM))
        model.add_parameters("R", (VOCAB_SIZE, HIDDEN_DIM))
        model.add_parameters("bias", (VOCAB_SIZE))

    def BuildLMGraph(self, sent):
        renew_cg()

        # Initialize the decoder with the average of source word vectors
        vec = average([lookup(self.m["lookup"],ind) for ind in isentsrc])
        init = [vec, tanh(vec)]
    
        init_state = self.builder.initial_state(init)

        R = parameter(self.m["R"])
        bias = parameter(self.m["bias"])
        errs = [] # will hold expressions
        es=[]
        state = init_state
        for (cw,nw) in zip(sent,sent[1:]):
            x_t = lookup(self.m["lookup"], int(cw))
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        # the error to backpropagate
        nerr = esum(errs)
        return nerr

    def encode(self, sent):

        # Initialize the decoder with the average of source word vectors
        vec = average([lookup(self.m["lookup"],ind) for ind in isentsrc])
        init = [vec, tanh(vec)]
        init_state = self.builder.initial_state(init)

        state = init_state
        for (cw,nw) in zip(sent,sent[1:]):
            x_t = lookup(self.m["lookup"], int(cw))
            state = state.add_input(x_t)

        # the final encoding of sent
        state = state.add_input(lookup(self.m["lookup"], sent[-1]))
        h = state.output()
        return h
    

class RNNDecoder:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM,
                 VOCAB_SIZE, builder=SimpleRNNBuilder):
        self.m = model
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        model.add_lookup_parameters("lookup", (VOCAB_SIZE, INPUT_DIM))
        model.add_parameters("R", (VOCAB_SIZE, HIDDEN_DIM))
        model.add_parameters("bias", (VOCAB_SIZE))

    def BuildLMGraph(self, sentsrc, senttgt):
        renew_cg()
        # Initialize the decoder with the average of source word vectors
        vec = average([lookup(self.m["lookup"],ind) for ind in isentsrc])
        init = [vec, tanh(vec)]

        init_state = self.builder.initial_state(init)

        R = parameter(self.m["R"])
        bias = parameter(self.m["bias"])
        errs = [] # will hold expressions
        state = init_state
        # build up source sentence language model representation
        for (cw,nw) in zip(sentsrc,sentsrc[1:]):
            x_t = lookup(self.m["lookup"], int(cw))
            state = state.add_input(x_t)
            # no need to propagate error for source sentence
            """
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
            """
        state = state.add_input(lookup(self.m["lookup"],int(sentsrc[-1])))
        # propagate error for target sentence translation
        for (cw,nw) in zip(senttgt,senttgt[1:]):
            x_t = lookup(self.m["lookup"], int(cw))
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        nerr = esum(errs)
        return nerr

    def decode(self, sentsrc, starttok, stoptok):
        renew_cg()
        # Initialize the decoder with the average of source word vectors
        vec = average([lookup(self.m["lookup"],ind) for ind in isentsrc])
        init = [vec, tanh(vec)]
        init_state = self.builder.initial_state(init)

        # build up source sentence language model representation
        state = init_state
        for cw in sentsrc:
            x_t = lookup(self.m["lookup"], int(cw))
            state = state.add_input(x_t)
        
        # generate target sentence translation
        R = parameter(self.m["R"])
        bias = parameter(self.m["bias"])
        y = [] # will hold output sentence
        y_t = int(starttok)
        y.append(y_t)
        while not y_t==int(stoptok):
            x_t = lookup(self.m["lookup"], y_t)
            state = state.add_input(x_t)
            r_t = bias + (R * state.output())
            ydist = softmax(r_t)
            dist = ydist.vec_value()
            rnd = random.random()
            for y_t,p in enumerate(dist):
                rnd -= p
                if rnd <=0: break
            y.append(y_t)
        return y

class RNNAttentionDecoder:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM,
                 VOCAB_SIZE, builder=SimpleRNNBuilder):
        self.m = model
        self.builder = builder(LAYERS, HIDDEN_DIM, HIDDEN_DIM, model)
        self._input_dim = INPUT_DIM
        model.add_lookup_parameters("lookup", (VOCAB_SIZE, INPUT_DIM))
        model.add_parameters("R", (VOCAB_SIZE, HIDDEN_DIM))
        model.add_parameters("bias", (VOCAB_SIZE))

    def BuildLMGraph(self, srcencode, srcmat, senttgt):
        # length of source sentence
        N_SRC = srcmat.npvalue().shape[1]

        # Initialize the decoder with the embedding and neutral attention
        neutral = sum_cols(srcmat)/N_SRC
        vec = concatenate([srcencode,neutral])
        init = [vec, tanh(vec)]
        init_state = self.builder.initial_state(init)

        R = parameter(self.m["R"])
        bias = parameter(self.m["bias"])
        errs = [] # will hold expressions
        state = init_state
        
        # we will treat attention as constants
        # ideally, backprop could bleed into the source RNNs
        # but we should have enough plasticity in the target parameters
        attention = vecInput(N_SRC)
        # propagate error for target sentence translation
        for (cw,nw) in zip(senttgt,senttgt[1:]):
            a = numpy.linalg.lstsq(srcmat.npvalue(),
                                   (srcencode-pickrange(state.output(),0,self._input_dim)).npvalue() 
                                   if state.output() else srcencode.npvalue())[0]
            attention.set([v for v in (a if sum(a)==0 else a/sum(a))])
            x_t = lookup(self.m["lookup"], int(cw))
            state = state.add_input(concatenate([x_t,srcmat*attention]))
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        nerr = esum(errs)
        return nerr

    def decode(self, srcencode, srcmat, starttok, stoptok):
        # length of source sentence
        N_SRC = srcmat.npvalue().shape[1]
        attention = vecInput(N_SRC)

        # Initialize the decoder with the average of source word vectors
        neutral = sum_cols(srcmat)/N_SRC
        vec = concatenate([srcencode,neutral])
        init = [vec, tanh(vec)]
        init_state = self.builder.initial_state(init)
        state = init_state

        # generate target sentence translation
        R = parameter(self.m["R"])
        bias = parameter(self.m["bias"])
        y = [] # will hold output sentence
        y_t = int(starttok)
        y.append(y_t)
        while not y_t==int(stoptok):
            a = numpy.linalg.lstsq(srcmat.npvalue(),
                                   (srcencode-pickrange(state.output(),0,self._input_dim)).npvalue()
                                   if state.output() else srcencode.npvalue())[0]
            attention.set([v for v in (a if sum(a)==0 else a/sum(a))])
            x_t = lookup(self.m["lookup"], y_t)
            state = state.add_input(concatenate([x_t,srcmat*attention]))
            r_t = bias + (R * state.output())
            ydist = softmax(r_t)
            dist = ydist.vec_value()
            rnd = random.random()
            for y_t,p in enumerate(dist):
                rnd -= p
                if rnd <=0: break
            y.append(y_t)
        return y
        

if __name__ == '__main__':
    if not len(sys.argv)>3:
        print "Usage: python attendtm.py [train.src] [train.tgt] {train/test} (test.src)"
        sys.exit()

    # prepare training corpus and vocabulary
    if sys.argv[3]=="train":
        trainsrc,traintgt = (util.CorpusReader(sys.argv[1]),util.CorpusReader(sys.argv[2]))
        vocabsrc,vocabtgt = (util.Vocab.from_corpus(trainsrc),util.Vocab.from_corpus(traintgt))
        pickle.dump(vocabsrc,open("{0}-vocab.pickle".format(sys.argv[1].split("/")[-1]),'wb'))
        pickle.dump(vocabtgt,open("{0}-vocab.pickle".format(sys.argv[2].split("/")[-1]),'wb'))
    else:
        vocabsrc = pickle.load(open("{0}-vocab.pickle".format(sys.argv[1].split("/")[-1]),'rb'))
        vocabtgt = pickle.load(open("{0}-vocab.pickle".format(sys.argv[2].split("/")[-1]),'rb'))
        testsrc = util.CorpusReader(sys.argv[4])
    VOCAB_SIZE_SRC,VOCAB_SIZE_TGT = (vocabsrc.size(),vocabtgt.size())
    
    # Initialize monolingual Source Forward RNN
    srcformodel = Model()
    srcforsgd = AdadeltaTrainer(srcformodel)
    srcforlm = RNNLanguageModel(srcformodel, LAYERS, SRC_DIM, SRC_DIM,
                                VOCAB_SIZE_SRC, builder=LSTMBuilder)

    # Initialize monolingual Source Backward RNN
    srcbackmodel = Model()
    srcbacksgd = AdadeltaTrainer(srcbackmodel)
    srcbacklm = RNNLanguageModel(srcbackmodel, LAYERS, SRC_DIM, SRC_DIM,
                                 VOCAB_SIZE_SRC, builder=LSTMBuilder)

    # Initialize monolingual Target Forward RNN
    tgtmodel = Model()
    tgtsgd = AdadeltaTrainer(tgtmodel)
    decoder = RNNAttentionDecoder(tgtmodel, LAYERS, TGT_DIM, 2*TGT_DIM,
                                  VOCAB_SIZE_TGT, builder=LSTMBuilder)

    if sys.argv[3]=="train":
        train = list(zip(trainsrc,traintgt))
        print "Training source encoder RNNs"
        bestloss = None
        bestiter = 0
        for ITER in xrange(N_ITERS):
            random.shuffle(train)
            loss = 0.0
            #pbar = ProgressBar(widgets=[SimpleProgress()], maxval=(len(train[(0 if ITER%10==0 else int(len(train)*.95))]))).start()
            for INST,instance in enumerate(train[(0 if ITER%10==0 else int(len(train)*.95)):]):
                sentsrc,senttgt = instance
                isentsrc = [vocabsrc.w2i[w] for w in sentsrc]

                # forward RNN
                errs = srcforlm.BuildLMGraph(isentsrc)
                loss+= errs.scalar_value()
                errs.backward()
                srcforsgd.update(1.0)

                # backward RNN
                errs = srcbacklm.BuildLMGraph(isentsrc[::-1])
                loss+= errs.scalar_value()
                errs.backward()
                srcbacksgd.update(1.0)
                if INST%10000==0:
                    print "    processed {0} instances of {1}".format(INST,
                       len(train[(0 if ITER%10==0 else int(len(train)*.95)):]))
                #pbar.update(INST+1)
            #pbar.finish()
            if ITER%10==0:
                if ITER==0 or loss < bestloss:
                    srcforlm.m.save("{0}-{1}_{2}-forward.rnn".format(
                        sys.argv[1].split("/")[-1],sys.argv[2].split("/")[-1],ITER))
                    srcbacklm.m.save("{0}-{1}_{2}-backward.rnn".format(
                        sys.argv[1].split("/")[-1],sys.argv[2].split("/")[-1],ITER))
                    if ITER > 0:
                        os.remove("{0}-{1}_{2}-forward.rnn".format(
                            sys.argv[1].split("/")[-1],sys.argv[2].split("/")[-1],bestiter))
                        os.remove("{0}-{1}_{2}-backward.rnn".format(
                            sys.argv[1].split("/")[-1],sys.argv[2].split("/")[-1],bestiter))
                    bestiter = ITER
                    bestloss = loss
                    print "  ITER",ITER,loss," (saved)"
                else:
                    print "  ITER",ITER,loss
        # clean up saved model files
        srcforlm.m.load("{0}-{1}_{2}-forward.rnn".format(sys.argv[1].split("/")[-1],
                                                         sys.argv[2].split("/")[-1],bestiter))
        srcbacklm.m.load("{0}-{1}_{2}-backward.rnn".format(sys.argv[1].split("/")[-1],
                                                           sys.argv[2].split("/")[-1],bestiter))
        srcforlm.m.save("{0}-{1}_forward.rnn".format(sys.argv[1].split("/")[-1],
                                                     sys.argv[2].split("/")[-1]))
        srcbacklm.m.save("{0}-{1}_backward.rnn".format(sys.argv[1].split("/")[-1],
                                                      sys.argv[2].split("/")[-1]))
        os.remove("{0}-{1}_{2}-forward.rnn".format(sys.argv[1].split("/")[-1],
                                                   sys.argv[2].split("/")[-1],bestiter))
        os.remove("{0}-{1}_{2}-backward.rnn".format(sys.argv[1].split("/")[-1],
                                                    sys.argv[2].split("/")[-1],bestiter))
        
        
        # save source language RNN encoders
        srcforlm.m.save("{0}-{1}_forward.rnn".format(sys.argv[1].split("/")[-1],
                                                     sys.argv[2].split("/")[-1]))
        srcbacklm.m.save("{0}-{1}_backward.rnn".format(sys.argv[1].split("/")[-1],
                                                       sys.argv[2].split("/")[-1]))

        print "Training target decoder with attention"
        bestloss = 0
        bestiter = None
        for ITER in xrange(N_ITERS):
            #pbar = ProgressBar(widgets=[SimpleProgress()],maxval=len(train)).start()
            random.shuffle(train)
            loss = 0.0
            for INST,instance in enumerate(train):
                sentsrc,senttgt = instance
                isentsrc = [vocabsrc.w2i[w] for w in sentsrc]
                isenttgt = [vocabtgt.w2i[w] for w in senttgt]
                
                renew_cg()            
                srcmat = concatenate_cols([concatenate([lookup(srcforlm.m["lookup"],i),
                                                        lookup(srcbacklm.m["lookup"],i)])
                                           for i in isentsrc])
                srcencode = concatenate([srcforlm.encode(isentsrc),srcbacklm.encode(isentsrc[::-1])])
                errs = decoder.BuildLMGraph(srcencode, srcmat, isenttgt)
                loss += errs.scalar_value()
                errs.backward()
                tgtsgd.update(1.0)
                if INST%10000==0:
                    print "    processed {0} instances of {1}".format(INST,len(train))
                #pbar.update(INST+1)
            #pbar.finish()
            if ITER==0 or loss < bestloss:
                decoder.m.save("{0}-{1}_{2}.aRNN".format(sys.argv[1].split("/")[-1],
                                                        sys.argv[2].split("/")[-1],ITER))
                if ITER > 0:
                    os.remove("{0}-{1}_{2}.aRNN".format(sys.argv[1].split("/")[-1],
                                                       sys.argv[2].split("/")[-1],bestiter))
                bestiter = ITER
                bestloss = loss
                print "  ITER",ITER,loss," (saved)"
            else:
                print "  ITER",ITER,loss

        decoder.m.load("{0}-{1}.aRNN".format(sys.argv[1].split("/")[-1],
                                            sys.argv[2].split("/")[-1],bestiter))
        decoder.m.save("{0}-{1}.aRNN".format(sys.argv[1].split("/")[-1],
                                            sys.argv[2].split("/")[-1]))
        os.remove("{0}-{1}_{2}.rnn".format(sys.argv[1].split("/")[-1],
                                           sys.argv[2].split("/")[-1],N_ITERS-1))
        print "Program completed under normal operation"
    else:
        #print "Loading encoder-decoder ensemble..."

        srcforlm.m.load("{0}-{1}_forward.rnn".format(sys.argv[1].split("/")[-1],
                                                     sys.argv[2].split("/")[-1]))

        srcbacklm.m.load("{0}-{1}_backward.rnn".format(sys.argv[1].split("/")[-1],
                                                       sys.argv[2].split("/")[-1]))
        
        decoder.m.load("{0}-{1}.aRNN".format(sys.argv[1].split("/")[-1],
                                            sys.argv[2].split("/")[-1]))

        test = list(testsrc)
        for sentsrc in test:
            isentsrc = [vocabsrc.w2i[w] for w in sentsrc]
            renew_cg()
            srcmat = concatenate_cols([concatenate([lookup(srcforlm.m["lookup"],i),
                                                    lookup(srcbacklm.m["lookup"],i)])
                                       for i in isentsrc])
            srcencode = concatenate([srcforlm.encode(isentsrc),srcbacklm.encode(isentsrc[::-1])])
            print " ".join([vocabtgt.i2w[i]
                   for i in decoder.decode(srcencode,srcmat,vocabtgt.w2i['<s>'],vocabtgt.w2i['</s>'])])

        #to decode sentsrc
        #renew_cg()
        #isentsrc = [vocabsrc.w2i[w] for w in sentsrc]
        #
        #[vocabtgt.i2w[i-VOCAB_SIZE_SRC] 
        #    for i in decoder.decode(isentsrc,vocabtgt.w2i['<s>']+VOCAB_SIZE_SRC,
        #                                     vocabtgt.w2i['</s>']+VOCAB_SIZE_SRC)]

    sys.exit()
