from pycnn import *
import random
import pdb, pickle, os

# number of training iterations
N_ITERS = 100

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

    def BuildLMGraph(self, sent, init):
        renew_cg()
    
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
        # the final encoding of sent
        state = state.add_input(sent[-1])
        h = state.output()

        return nerr, h

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
        es=[]
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
        

if __name__ == '__main__':
    trainsrc,traintgt = (util.CorpusReader(sys.argv[1]),util.CorpusReader(sys.argv[2]))
    
    vocabsrc,vocabtgt = (util.Vocab.from_corpus(trainsrc),util.Vocab.from_corpus(traintgt))
    VOCAB_SIZE_SRC,VOCAB_SIZE_TGT = (vocabsrc.size(),vocabtgt.size())

    #Bilingual Forward RNN
    model = Model()
    sgd = AdadeltaTrainer(model)
    decoder = RNNDecoder(model, LAYERS, INPUT_DIM, HIDDEN_DIM,
                          VOCAB_SIZE_SRC+VOCAB_SIZE_TGT, builder=LSTMBuilder)

    pickle.dump(vocabsrc,open("{0}-vocab.pickle".format(sys.argv[1].split("/")[-1]),'wb'))
    pickle.dump(vocabtgt,open("{0}-vocab.pickle".format(sys.argv[2].split("/")[-1]),'wb'))
    train = list(zip(trainsrc,traintgt))
    for ITER in xrange(N_ITERS):
        random.shuffle(train)
        loss = 0.0
        for i,instance in enumerate(train):
            sentsrc,senttgt = instance
            isentsrc = [vocabsrc.w2i[w] for w in sentsrc]
            isenttgt = [vocabtgt.w2i[w]+VOCAB_SIZE_SRC for w in senttgt]
            errs = decoder.BuildLMGraph(isentsrc,isenttgt)
            loss += errs.scalar_value()
            errs.backward()
            sgd.update(1.0)
        print "ITER",ITER,loss
        decoder.m.save("{0}-{1}_{2}.rnn".format(sys.argv[1].split("/")[-1],
                                                sys.argv[2].split("/")[-1],ITER))
        if ITER>0:
            os.remove("{0}-{1}_{2}.rnn".format(sys.argv[1].split("/")[-1],
                                               sys.argv[2].split("/")[-1],ITER-1))
        
    decoder.m.save("{0}-{1}.rnn".format(sys.argv[1].split("/")[-1],
                                        sys.argv[2].split("/")[-1]))
    os.remove("{0}-{1}_{2}.rnn".format(sys.argv[1].split("/")[-1],
                                       sys.argv[2].split("/")[-1],N_ITERS-1))
    pdb.set_trace()

    #to decode sentsrc
    #isentsrc = [vocabsrc.w2i[w] for w in sentsrc]
    #[vocabtgt.i2w[i-VOCAB_SIZE_SRC] 
    #    for i in decoder.decode(isentsrc,vocabtgt.w2i['<s>']+VOCAB_SIZE_SRC,
    #                                     vocabtgt.w2i['</s>']+VOCAB_SIZE_SRC)]
