output.txt contains output from a mostly trained baseline system.  The other systems are still training and final output will be submitted in a few days in accordance with the late submission policy.

There are several Python programs here:

 - `python bleu.py your-output.txt ref.txt` to compute the BLEU score of your output against the reference translation.
 - `python rnnlm.py ref.txt` trains an LSTM language model, just for your reference if you want to use pyCNN to perform this assignment.

The joint system consists of three LSTMs which are trained jointly to function as an encoder-decoder neural translation system with attention.  The two source language LSTMs encode a representation of the source sentence by processing it forwards and backwards respectively.  Then, the third LSTM decodes a matrix produced by stacking the outputs of the two source LSTMs using an attention model selected to maximize the ``coverage'' of the input with respect to the expected output embedding.  More specifically, a least-squares approximation is used to solve for A: X-X' = EA, where E is the matrix encoding of the sentence, X' is the current output of the target LSTM, and X is the (final output activations, stacked vertically) embedding of the source LSTMs.
 - `python attendjoint.py input.src input.tgt train` trains the joint encoder-decoder neural translation system with attention.
 - `python attendjoint.py input.src input.tgt test test.src` runs the joint encoder-decoder neural translation system with attention on a test dataset.  It is recommended that output be captured by redirecting to a file.

The pipelined system is identical to the joint system, but trains the two source language LSTMs respectively as forward and backward language models prior to training the decoder LSTM.
 - `python attendtm.py input.src input.tgt train` trains a pipelined encoder-decoder neural translation system with attention.
 - `python attendtm.py input.src input.tgt test test.src` runs the pipelined encoder-decoder neural translation system with attention on a test dataset.  It is recommended that output be captured by redirecting to a file.

The baseline system does not incorporate any attention model.  This system utilizes a single LSTM model to process the source sentence and generate the target sentence.
 - `python nedtm.py input.src input.tgt train` trains a baseline neural translation model.
 - `python nedtm.py input.src input.tgt test test.src` runs the baseline neural translation model on a test dataset.  It is recommended that output be captured by redirecting to a file.

The `data/` directory contains the files needed to develop the MT system:

 - `data/train.*` the source and target training files.

 - `data/dev.*` the source and target development files.

 - `data/test.src` the source side of the blind test set.