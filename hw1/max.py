#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
      fe_count[(f_i,e_j)] += 1
  for e_j in set(e):
    e_count[e_j] += 1
  if n % 500 == 0:
    sys.stderr.write(".")

dice = defaultdict(int)
for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
  dice[(f_i,e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])
  if k % 5000 == 0:
    sys.stderr.write(".")
sys.stderr.write("\n")

for (f, e) in bitext:
  if False and len(f) >= len(e):
    for (i, f_i) in enumerate(f): 
      best = -1
      bestscore = 0
      for (j, e_j) in enumerate(e):
        val = dice[(f_i,e_j)]
        if best==-1 or val>bestscore:
          bestscore = val
          best = j
      sys.stdout.write("%i-%i " % (i,best))
  else:
    for (j, e_j) in enumerate(e):
      best = -1
      bestscore = 0
      for (i, f_i) in enumerate(f): 
        val = dice[(f_i,e_j)]
        if best==-1 or val>bestscore:
          bestscore = val
          best = i
      sys.stdout.write("%i-%i " % (best,j))

  sys.stdout.write("\n")
