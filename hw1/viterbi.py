#!/usr/bin/env python
import optparse
import sys
import em

#The model to use for decoding
DECODER = "BeamDecoder" #DiagonalCompoundDecoder"

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-m", "--translation_model", dest="translation_model", default="translation.model", type="string", help="Pickled translation probabilities trained using em.py")
(opts, _) = optparser.parse_args()

bitext = [line for line in open(opts.bitext,'r')][:opts.num_sents]

model = getattr(em,DECODER)(opts.translation_model)

for line in bitext:
  (f,e) = model.get_parallel_instance(line)
  for alignment in model.get_alignment(f,e):
    sys.stdout.write("{0}-{1} ".format(*alignment))
  sys.stdout.write("\n")
