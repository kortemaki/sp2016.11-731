Keith Maki
11-731 Machine Translation
Homework 3
4/1/16

In this homework, 


I submit results from the following configuration, which outperforms runs with longer
k=10, T=200, t=5

-------------- File Specifics --------------

There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - './anneal' a simulated annealing-based phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./anneal | ./grade

The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model

