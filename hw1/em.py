# -*- coding: utf-8 -*-
from nltk.stem.snowball import SnowballStemmer
import random
import pickle
import itertools
import pdb
import heapq
from aligntools import PosTagger
import math
import heapq
from collections import Counter
from heapq import *

POS_VERBS = set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

class Model1(object):
    german_stemmer = SnowballStemmer("german")
    english_stemmer = SnowballStemmer("english")

    null_val = ''.decode('utf-8').encode('utf-8')
    rare_val = "__RARE_TOKEN_"

    #Both vocabularies have a null
    german_vocab = set([null_val])
    english_vocab = set([null_val])

    #Conditional probabilities are initialized to a uniform distribution over all german words conditioned on an english
    #word. However, most words never co-occurr. To save space and avoid storing such parameters that are bound to be 0, we
    #only store those conditional probabilities involving word pairs that actually co-occur.

    translation_probs = {}

    #Expected count number of alignments between each german word - english word pair

    counts = {}

    #Expected number of all alignments involving an english word (sum counts over all german words while fixing the english
    #word

    totals = {}

    def __init__(self, parameter_file):
        (self.translation_probs,self.rare_tokens) = pickle.load(open(parameter_file,'rb'))
        self.INPUT_FILE = None

    def get_params(self):
        return(self.translation_probs)

    def get_german_stem(self, word):
        return(self.german_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8'))

    def get_english_stem(self, word):
        return(self.english_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8'))

    def get_translation_prob(self, german_stem, english_stem):
        if german_stem==self.rare_val and english_stem==self.rare_val:
            prior = 2
        else:
            prior = 1
        return prior*self.translation_probs.get((german_stem,english_stem),0.0)

    def get_parallel_instance(self, corpus_line):
        [german, english] = corpus_line.strip().split(' ||| ')
        german_stemmed_sentence,english_stemmed_sentence =\
               ([self.get_german_stem(word).lower() for word in german.split(' ')],
                [self.get_english_stem(word).lower() for word in english.split(' ')])
        return ([self.rare_val if tok in self.rare_tokens[0] else tok
                 for tok in german_stemmed_sentence],
                [self.rare_val if tok in self.rare_tokens[1] else tok 
                 for tok in english_stemmed_sentence])

    def get_counts(self, sent):
        """
        Returns a dicts mapping stemmed words to
        their respective counts in the sentence sent.
        Also, one null token count is added to sentence
        """
        ret = {self.null_val:1}
        for word in sent:
            ret[word] = ret.get(word, 0) + 1
        return ret

    def get_prior(self,**kwargs):
        return 1

    def get_alignment(self, german, english):
        """
        Returns model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies
        the best english word (or NULL) to align to
        """
        english.append(self.null_val)
        alignment = []
        for (i, g_i) in enumerate(german):
            best = -1
            bestscore = 0
            for (j, e_j) in enumerate(english):
                val = self.get_prior()*self.get_translation_prob(g_i,e_j)
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best < len(english)-1:
                yield (i,best) # don't yield anything for NULL alignment




class Bidirectional_decoder(Model1):
    def __init__(self,e_g_parameter_file,g_e_parameter_file="german_to_english_iter74.model"):
        super(Bidirectional_decoder,self).__init__(e_g_parameter_file)
        self.g_e_translation_probs = pickle.load(open(g_e_parameter_file,'rb'))


    def get_g_e_translation_prob(self, german_stem, english_stem):
        return self.g_e_translation_probs.get((german_stem,english_stem),0.0)


    def get_g_e_alignment(self, german, english):
        """
        Returns model1 alignment for a DE/EN parallel sentence pair.
        For each english word, identifies
        the best german word (or NULL) to align to
        """
        german.append(self.null_val)
        alignment = []
        for (j, e_j) in enumerate(english):
            best = -1
            bestscore = 0
            for (i, g_i) in enumerate(german):
                val = self.get_prior()*self.get_g_e_translation_prob(g_i,e_j)
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best < len(german)-1:
                yield (i,best) # don't yield anything for NULL alignment


    def get_alignment(self, german, english):
        """
        Returns model1 alignment for a DE/EN parallel sentence pair.
        Includes alignment decisions made in either direction by model1.
        """
        g_e_alignment = set(self.get_g_e_alignment(german, english))
        for el in super(Bidirectional_decoder, self).get_alignment(german,english):
            if el not in g_e_alignment: # only yield each alignment once
                yield el
        for el in g_e_alignment:
            yield el




class POS_decoder(Model1):
    TUNE_POS_WEIGHT = 2

    def __init__(self, parameter_file):
        super(POS_decoder,self).__init__(parameter_file)
        self.tagger = PosTagger()


    def get_parallel_instance(self, corpus_line):
        (german, english) = corpus_line.strip().split(' ||| ')
        return ([word for word in german.split(" ")],
                [word for word in english.split(" ")])


    def get_prior(self, **features):
        """
        returns 1+TUNE_POS_WEIGHT if the POS tags are aligned else 1
        """
        return 1+self.TUNE_POS_WEIGHT*(
            features.get("tag_german",self.null_val)==features.get("tag_english",self.null_val))


    def get_alignment(self, german, english):
        """
        Returns Model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies the best english word (or NULL) to align to
        Applies a prior which assigns higher probability to alignments which preserve POS tags.
        """
        alignment = []
        gtags = self.tagger.parse(german,"de")
        etags = self.tagger.parse(english,"en")
        (german,english) = ([self.get_german_stem(word).lower() for word in german],
                            [self.get_english_stem(word).lower() for word in english])
        etags.append(self.null_val)
        english.append(self.null_val)
        for (i, g_i) in enumerate(german):
            best = -1
            bestscore = 0
            for (j, e_j) in enumerate(english):
                val = self.get_prior(tag_german=gtags[i],tag_english=etags[j])*\
                      self.get_translation_prob(g_i,e_j)
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best < len(english)-1:
                yield (i,best) # don't yield anything for NULL alignment




class Bidirectional_POS_decoder(POS_decoder,Bidirectional_decoder):

    def __init__(self,e_g_parameter_file,g_e_parameter_file="german_to_english_iter74.model"):
        super(Bidirectional_POS_decoder,self).__init__(e_g_parameter_file)
        self.g_e_translation_probs = pickle.load(open(g_e_parameter_file,'rb'))


    def get_g_e_alignment(self, german, english):
        """
        Returns Model1 alignment for a DE/EN parallel sentence pair.
        For each english word, identifies the best german word (or NULL) to align to
        Applies a prior which assigns higher probability to alignments which preserve POS tags.
        """
        alignment = []
        gtags = self.tagger.parse(german,"de")
        etags = self.tagger.parse(english,"en")
        (german, english) = ([self.get_german_stem(word).lower() for word in german],
                             [self.get_english_stem(word).lower() for word in english])
        gtags.append(self.null_val)
        german.append(self.null_val)
        for (j, e_j) in enumerate(english):
            best = -1
            bestscore = 0
            for (i, g_i) in enumerate(german):
                val = self.get_prior(tag_german=gtags[i],tag_english=etags[j])*\
                      self.get_translation_prob(g_i,e_j)
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best < len(german)-1:
                yield (i,best) # don't yield anything for NULL alignment


    def get_alignment(self, german, english):
        """
        Returns model1 alignment for a DE/EN parallel sentence pair.
        Includes alignment decisions made in either direction by model1.
        """
        g_e_alignment = self.get_g_e_alignment(german, english)
        for el in super(Bidirectional_POS_decoder, self).get_alignment(german,english):
            if el not in g_e_alignment: # only yield each alignment once
                yield el
        for el in g_e_alignment:
            yield el




class DiagonalAligner(POS_decoder):
    """ Adds a diagonal prior to the POS prior. Uses Model 1 alignment """
    DIAG_WEIGHT = 1

    def __init__(self, parameter_file):
        super(DiagonalAligner,self).__init__(parameter_file)


    def get_prior(self, **features):
        """ 2 priors:
            * POS tags 1+TUNE_POS_WEIGHT if the POS tags are aligned else 1
            * diagonal prior
        """
        pos_prior = 1+self.TUNE_POS_WEIGHT*(
            features.get("tag_german",self.null_val)==features.get("tag_english",self.null_val))

        de_idx = float(features.get("position_german", self.null_val))
        de_len = float(features.get("length_german", self.null_val))
        en_idx = float(features.get("position_english", self.null_val))
        en_len = float(features.get("length_english", self.null_val))

        diag_prior = (1 - abs(de_idx/de_len - en_idx/en_len)) * self.DIAG_WEIGHT
        return pos_prior * diag_prior


    def get_alignment(self, german, english):
        """
        Returns Model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies the best english word (or NULL) to align to
        Applies a prior which assigns higher probability to alignments which preserve POS tags and are diagonally aligned
        """
        alignment = []
        gtags = self.tagger.parse(german,"de")
        etags = self.tagger.parse(english,"en")
        english.append(self.null_val)
        for (i, g_i) in enumerate(german):
            german_len = len(german)
            gs_i = self.get_german_stem(g_i).lower()
            best = -1
            bestscore = 0
            for (j, e_j) in enumerate(english):
                english_len = len(english)
                es_j = self.get_english_stem(e_j).lower()

                # handle null
                if e_j == self.null_val:
                    prior = 1.0
                else:
                    prior = self.get_prior(tag_german=gtags[i],position_german=i,\
                                           length_german=german_len,tag_english=etags[j],\
                                           position_english=j,length_english=english_len)
                val = prior * self.get_translation_prob(gs_i,es_j)
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best < len(english)-1:
                yield (i,best) # don't yield anything for NULL alignment
 
class EM_model1(Model1):

    ENGLISH_TO_GERMAN = 1
    GERMAN_TO_ENGLISH = 2

    english_to_german_translation_probs = {}
    german_to_english_translation_probs = {}

    german_totals = {}
    english_totals = {}

    RARE_THRESHOLD = 50

    def __init__(self, input_file, output_file, n_iterations):
        self.MAX_ITERS = n_iterations
        self.INPUT_FILE = input_file
        self.OUTPUT_FILE = output_file
        self.rare_tokens = (set(),set())
        self.rare_tokens = self._find_rares()

    def _find_rares(self):
        encounts = Counter()#{}
        decounts = Counter()
        with open(self.INPUT_FILE) as ip:
            for line in ip:
                [german_stemmed_sentence, english_stemmed_sentence] = super(EM_model1,self).get_parallel_instance(line)
                decounts.update(german_stemmed_sentence)
                encounts.update(english_stemmed_sentence)
                #for stem in english_stemmed_sentence:
                #    if stem not in counts:
                #        counts[stem] = Counter(german_stemmed_sentence)
                #    else:
                #        counts[stem].update(english_stemmed_sentence)        
        enrares = set([word for word in encounts if encounts[word]<self.RARE_THRESHOLD])
        derares = set([word for word in decounts if decounts[word]<self.RARE_THRESHOLD])
        return (derares,enrares)

    def preprocess(self, direction):
        assert direction == self.ENGLISH_TO_GERMAN or direction == self.GERMAN_TO_ENGLISH, "Invalid translation direction"
        ip_line_counter = 0
        corpus = []
        with open(self.INPUT_FILE) as ip:
            print("Starting to process corpus")
            for line in ip:
                ip_line_counter += 1
                if (ip_line_counter % 1000 == 0):
                    print("Processed %d lines" % (ip_line_counter))
                [german_stemmed_sentence, english_stemmed_sentence] = self.get_parallel_instance(line)
                ret_german = {}
                ret_english = {}
                for german_stemmed_word in german_stemmed_sentence:
                    ret_german[german_stemmed_word] = ret_german.get(german_stemmed_word, 0.0) + 1.0
                    self.german_vocab.add(german_stemmed_word)
                    if direction == self.GERMAN_TO_ENGLISH:
                        for english_stemmed_word in english_stemmed_sentence:
                            key = (english_stemmed_word, german_stemmed_word)
                            if not self.german_to_english_translation_probs.has_key(key):
                                self.german_totals[german_stemmed_word] = self.german_totals.get(
                                    german_stemmed_word, 0.0) + 1.0
                            self.german_to_english_translation_probs[key] = 1.0
                for english_stemmed_word in english_stemmed_sentence:
                    ret_english[english_stemmed_word] = ret_english.get(english_stemmed_word, 0.0) + 1.0
                    self.english_vocab.add(english_stemmed_word)
                    if direction == self.ENGLISH_TO_GERMAN:
                        for german_stemmed_word in german_stemmed_sentence:
                            key = (german_stemmed_word, english_stemmed_word)
                            if not self.english_to_german_translation_probs.has_key(key):
                                self.english_totals[english_stemmed_word] = self.english_totals.get(
                                    english_stemmed_word, 0.0) + 1.0
                            self.english_to_german_translation_probs[key] = 1.0
                ret_german[self.null_val] = 1  # null added to sentence
                ret_english[self.null_val] = 1  # null added to sentence
                corpus.append([ret_german, ret_english])
        return (corpus)

    def normalize(self, direction):
        for english_stemmed_word in self.english_vocab:
            for german_stemmed_word in self.german_vocab:
                if direction == self.ENGLISH_TO_GERMAN:
                    key = (german_stemmed_word, english_stemmed_word)
                    if self.english_to_german_translation_probs.has_key(
                            key):  # prevent populating entries unless they occur in parallel sentences
                        self.english_to_german_translation_probs[key] = self.english_to_german_translation_probs[key] / \
                                                                        self.english_totals[
                                                                            english_stemmed_word]  # english_totals of english word should NEVER be 0
                    elif english_stemmed_word == self.null_val:
                        self.english_to_german_translation_probs[key] = 1.0 / len(self.german_vocab)
                elif direction == self.GERMAN_TO_ENGLISH:
                    key = (english_stemmed_word, german_stemmed_word)
                    if self.german_to_english_translation_probs.has_key(
                            key):  # prevent populating entries unless they occur in parallel sentences
                        self.german_to_english_translation_probs[key] = self.german_to_english_translation_probs[key] / \
                                                                        self.german_totals[
                                                                            german_stemmed_word]  # german_totals of german word should NEVER be 0
                    elif german_stemmed_word == self.null_val:
                        self.german_to_english_translation_probs[key] = 1.0 / len(self.english_vocab)

    def stop_condition(self,
                       iter_count):  # Currently only checking for iteration limit. Ideally, we should also check for
        # convergence, i.e., when parameters change by value below a certain threshold
        if iter_count == self.MAX_ITERS:
            return(True)
        else:
            return(False)

    def estimate_params(self, direction, store_frequency):
        assert direction == self.GERMAN_TO_ENGLISH or direction == self.ENGLISH_TO_GERMAN, "Invalid direction specified"

        # May have to take care of last line being empty
        self.corpus = self.preprocess(direction)

        self.normalize(direction)

        iter_count = 0

        """
        EM algorithm for estimating the translation probablities
        See https://www.cl.cam.ac.uk/teaching/1011/L102/clark-lecture3.pdf for a good tutorial
        """

        while(True):  #until convergence or max_iters
            print("Iteration " + str(iter_count + 1))
            iter_count += 1
            self.counts = {}  # All counts default to 0. These are counts of (german, english) word pairs
            self.english_totals = {}  # All english_totals default to 0. These are sums of counts (marginalized over all foreign words), for each
            self.german_totals = {}  # totals for german words, used when estimating p(english word | german_word) instead of p(german_word | english_word)
            for parallel_instance in self.corpus:  # Stemmed parallel instances stored in memory to speed up EM
                german_sent_dict = parallel_instance[0]
                english_sent_dict = parallel_instance[1]
                if direction == self.ENGLISH_TO_GERMAN:
                    for german_word in german_sent_dict.keys():  # For each unique german word in the german sentence
                        german_word_count = german_sent_dict[german_word]  # Its count in the german sentence
                        total_s = 0.0  # Expected count of number of alignments for this german word with any english word
                        for english_word in english_sent_dict.keys():
                            total_s += self.english_to_german_translation_probs.get((german_word, english_word),
                                                                                    0.0) * german_word_count
                        for english_word in english_sent_dict.keys():
                            english_word_count = english_sent_dict[english_word]
                            if self.counts.has_key(english_word):
                                self.counts[english_word][german_word] = self.counts[english_word].get(german_word,
                                                                                                       0.0) + self.english_to_german_translation_probs.get(
                                    (german_word, english_word), 0.0) * german_word_count * english_word_count / total_s
                            else:
                                self.counts[english_word] = {}
                                self.counts[english_word][german_word] = self.counts[english_word].get(german_word,
                                                                                                       0.0) + self.english_to_german_translation_probs.get(
                                    (german_word, english_word), 0.0) * german_word_count * english_word_count / total_s
                            # Expected count of alignments between german word and this english word, divided by the expected count of all alignments of this german word
                            self.english_totals[english_word] = self.english_totals.get(english_word,
                                                                                        0.0) + self.english_to_german_translation_probs.get(
                                (german_word, english_word), 0.0) * german_word_count * english_word_count / total_s
                            # Aggregating the expected counts of all german words, for each english word. This will be used as a normalizing factor
                elif direction == self.GERMAN_TO_ENGLISH:
                    for english_word in english_sent_dict.keys():  # For each unique german word in the german sentence
                        english_word_count = english_sent_dict[english_word]  # Its count in the german sentence
                        total_s = 0.0  # Expected count of number of alignments for this german word with any english word
                        for german_word in german_sent_dict.keys():
                            total_s += self.german_to_english_translation_probs.get((english_word, german_word),
                                                                                    0.0) * english_word_count
                        for german_word in german_sent_dict.keys():
                            german_word_count = german_sent_dict[german_word]
                            if self.counts.has_key(german_word):
                                self.counts[german_word][english_word] = self.counts[german_word].get(english_word,
                                                                                                      0.0) + self.german_to_english_translation_probs.get(
                                    (english_word, german_word), 0.0) * english_word_count * german_word_count / total_s
                            else:
                                self.counts[german_word] = {}
                                self.counts[german_word][english_word] = self.counts[german_word].get(english_word,
                                                                                                      0.0) + self.german_to_english_translation_probs.get(
                                    (english_word, german_word), 0.0) * english_word_count * german_word_count / total_s
                            # Expected count of alignments between german word and this english word, divided by the expected count of all alignments of this german word
                            self.german_totals[german_word] = self.german_totals.get(german_word,
                                                                                     0.0) + self.german_to_english_translation_probs.get(
                                (english_word, german_word), 0.0) * english_word_count * german_word_count / total_s
                            # Aggregating the expected counts of all german words, for each english word. This will be used as a normalizing factor
            if direction == self.ENGLISH_TO_GERMAN:
                for english_word in self.english_totals.keys():  # restricting to domain total( . )
                    for german_word in self.counts[english_word].keys():
                        self.english_to_german_translation_probs[(german_word, english_word)] = self.counts[
                                                                                                    english_word].get(
                            german_word, 0.0) / self.english_totals.get(english_word, 0.0)
                        # Neither domain nor counts should never be 0 given our domain restriction
            elif direction == self.GERMAN_TO_ENGLISH:
                for german_word in self.german_totals.keys():  # restricting to domain total( . )
                    for english_word in self.counts[german_word].keys():
                        self.german_to_english_translation_probs[english_word, german_word] = self.counts[
                                                                                                  german_word].get(
                            english_word, 0.0) / self.german_totals.get(german_word, 0.0)
                        # Neither domain nor counts should never be 0 given our domain restriction

            if (iter_count % store_frequency == 0):  # Store the model at some frequency of iterations
                print("Storing model after %d iterations" % (iter_count))
                model_dump = open(self.OUTPUT_FILE, 'wb')
                if direction == self.ENGLISH_TO_GERMAN:
                    print("Spot checking on 5% of english vocabulary before storing!")
                    self.sanity_check(direction, int(len(self.english_vocab) * 0.05))
                    pickle.dump((self.english_to_german_translation_probs,self.rare_tokens), model_dump)
                    print("Storing english to german translation model after %d iterations" % (iter_count))
                elif direction == self.GERMAN_TO_ENGLISH:
                    print("Spot checking on 5% of german vocabulary before storing!")
                    self.sanity_check(direction, int(len(self.german_vocab) * 0.05))
                    pickle.dump(self.german_to_english_translation_probs, model_dump)
                    print("Storing german to english model after %d iterations" % (iter_count))
                model_dump.close()

            if (self.stop_condition(iter_count)):
                print("Storing model after %d iterations" % (iter_count))
                model_dump = open(self.OUTPUT_FILE, 'wb')
                if direction == self.ENGLISH_TO_GERMAN:
                    print("Spot checking on 5% of english vocabulary before storing!")
                    self.sanity_check(direction, int(len(self.english_vocab) * 0.05))
                    pickle.dump((self.english_to_german_translation_probs,self.rare_tokens), model_dump)
                    print("Storing english to german translation model after %d iterations" % (iter_count))
                elif direction == self.GERMAN_TO_ENGLISH:
                    print("Spot checking on 5% of german vocabulary before storing!")
                    self.sanity_check(direction, int(len(self.german_vocab) * 0.05))
                    pickle.dump(self.german_to_english_translation_probs, model_dump)
                    print("Storing german to english model after %d iterations" % (iter_count))
                model_dump.close()
                break

        print("Memory usage stats")
        print("German vocab length: ", len(self.german_vocab))
        print("English vocab length: ", len(self.english_vocab))
        print("No of cross product entries required: ", len(self.german_vocab) * len(self.english_vocab))

        if direction == self.ENGLISH_TO_GERMAN:
            print(
            "Num of conditional probabilities actually stored: ", len(self.english_to_german_translation_probs.keys()))
            print("Num of english_totals actually stored: ", len(self.english_totals.keys()))
        elif direction == self.GERMAN_TO_ENGLISH:
            print(
                "Num of conditional probabilities actually stored: ",
                len(self.german_to_english_translation_probs.keys()))
            print("Num of german_totals actually stored: ", len(self.german_totals.keys()))
        tot_counts = 0
        for key in self.counts.keys():
            tot_counts += len(self.counts[key].keys())
        print("Num of counts actually stored: ", tot_counts)

        self.sanity_check(direction)
        if direction == self.GERMAN_TO_ENGLISH:
            return(self.german_to_english_translation_probs)
        elif direction == self.ENGLISH_TO_GERMAN:
            return(self.english_to_german_translation_probs)

    def sanity_check(self, direction, n_sample=None):
        if direction == self.ENGLISH_TO_GERMAN:
            source_vocab = self.english_vocab
            target_vocab = self.german_vocab
            translation_probs = self.english_to_german_translation_probs
        elif direction == self.GERMAN_TO_ENGLISH:
            source_vocab = self.german_vocab
            target_vocab = self.english_vocab
            translation_probs = self.german_to_english_translation_probs
        if n_sample is None:
            test_source_words = [source_word for source_word in
                                 source_vocab]  # should not further stem words in english vocab
            print("Performing sanity check on full vocabulary")
        else:
            test_source_words = [
                source_word for
                source_word in random.sample(source_vocab, n_sample)]
            # print("Spot checking for the following words ")
            # print(test_source_words)

        for source_word in test_source_words:
            max_prob_word = None
            max_prob = 0.0
            tot_conditional_prob = 0.0
            for target_word in target_vocab:
                if translation_probs.get((target_word, source_word), 0.0) != 0.0:
                    tot_conditional_prob += translation_probs.get((target_word, source_word), 0.0)
                    if translation_probs.get((target_word, source_word), 0.0) > max_prob:
                        max_prob = translation_probs.get((target_word, source_word), 0.0)
                        max_prob_word = target_word
            assert abs(tot_conditional_prob - 1.0) < 0.000000000001, 'Tot conditional probability != 1 !!!'
            #if n_sample is not None:
            #    print("Most likely word for source word ", source_word, " is the target word ", max_prob_word, " with translation probability ", max_prob)
        print("Sanity check passed!")




class DE_Compound(Model1):

    def __init__(self,parameter_file):#,compounds_file="data/compound.dict"):
        super(DE_Compound,self).__init__(parameter_file)
        self.compounds = pickle.load(open("data/compound.dict",'rb'))#compounds_file)

    def get_parallel_instance(self, corpus_line):
        [german, english] = corpus_line.strip().split(' ||| ')
        german_stemmed_sentence,english_stemmed_sentence =\
                ([(i,self.get_german_stem(w).lower())
                 for (i,word) in [(j,(wt.strip("„") if "„" in wt else wt)) 
                                  for (j,wort) in enumerate(german.split(' '))
                                  for wt in ([wort] if wort=='-' else wort.split('-'))]
                 for w in ([word] if word not in self.compounds
                           else self.compounds[word])],
                [(i,self.get_english_stem(word).lower())
                 for (i,word) in [(j,wt) for (j,wort) in enumerate(english.split(' '))
                                  for wt in ([wort] if wort=='-' else wort.split('-'))]])
        return ([(i,self.rare_val) if word in self.rare_tokens[0] else (i,word)
                 for (i,word) in german_stemmed_sentence],
                [(i,self.rare_val) if word in self.rare_tokens[1] else (i,word) 
                 for (i,word) in english_stemmed_sentence])

    def get_alignment(self, german, english):
        """
        Returns model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies the best english word (or NULL) to align to
        """
        english.append((len(english),self.null_val))
        alignment = []
        for (i, g_i) in german:
            best = -1
            bestscore = 0
            for (j, e_j) in english:
                val = self.get_prior()*self.get_translation_prob(g_i,e_j)
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best < len(english)-1:
                yield (i,best) # don't yield anything for NULL alignment




class DE_Compound_POS_decoder(POS_decoder,DE_Compound):

    def __init__(self, parameter_file):
        super(DE_Compound_POS_decoder,self).__init__(parameter_file)
        self.compounds = pickle.load(open("data/compound.dict",'rb'))#compounds_file)


    def get_parallel_instance(self, corpus_line):
        [german, english] = corpus_line.strip().split(' ||| ')
        return ([word for word in german.split(' ')],
                [word for word in english.split(' ')])


    def stem(self, word):
        return word[0]


    def tag(self, word):
        return word[1]


    def get_alignment(self, german, english):
        """
        Returns Model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies the best english word (or NULL) to align to
        Applies a prior which assigns higher probability to alignments which preserve POS tags.
        """
        alignment = []
        (german,english) = self.tag_and_stem_compounds(german,english)
        english.append((english[-1][0]+1,(self.null_val,self.null_val)))
        for (i, g_i) in german:
            best = -1
            bestscore = 0
            for (j, e_j) in english:
                val = self.get_prior(tag_german=self.tag(g_i),tag_english=self.tag(e_j))*\
                      self.get_translation_prob(self.stem(g_i),self.stem(e_j))
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best < len(english)-1:
                yield (i,best) # don't yield anything for NULL alignment


    def tag_and_stem_compounds(self, german, english):
        gtags = self.tagger.parse(german,"de")
        etags = self.tagger.parse(english,"en")
 
        (german,english) = ([(i,(self.get_german_stem(self.stem(w)).lower(),self.tag(w)))
                             for (i,word) in [(j,((wt.strip("„") if wt[:3]=="„" else wt),
                                                  self.tag(wort))) 
                                              for (j,wort) in enumerate(zip(german,gtags))
                                              for wt in ([self.stem(wort)] if self.stem(wort)=='-'
                                                         else self.stem(wort).split('-'))]
                             for w in ([(self.stem(word),self.tag(word))]
                                       if self.stem(word) not in self.compounds
                                       else [(stem,self.tag(word))
                                             for stem in self.compounds[self.stem(word)]])],
                            [(i,(self.get_english_stem(self.stem(word)).lower(),self.tag(word)))
                             for (i,word) in [(j,(wt,self.tag(wort))) 
                                              for (j,wort) in enumerate(zip(english,etags))
                                              for wt in ([self.stem(wort)] if self.stem(wort)=='-'
                                                         else self.stem(wort).split('-'))]])

        return ([(i,(self.rare_val,self.tag(word))) 
                 if self.stem(word) in self.rare_tokens[0] else (i,word)
                 for (i,word) in german],
                [(i,(self.rare_val,self.tag(word))) 
                 if self.stem(word) in self.rare_tokens[1] else (i,word) 
                 for (i,word) in english])


class EM_DE_Compound(DE_Compound,EM_model1):

    def __init__(self, input_file, output_file, n_iterations):

        self.MAX_ITERS = n_iterations
        self.INPUT_FILE = input_file
        self.OUTPUT_FILE = output_file

        self.compounds = pickle.load(open("data/compound.dict",'rb'))#compounds_file)
        print("Accumulating word counts")
        self.rare_tokens = (set(), set())
        self.rare_tokens = self._find_rares()
        print("Done identifying rare words")


    def get_parallel_instance(self, corpus_line):
        """
        Removes the DE_Compound word indices for easier EM estimation
        """
        (german,english) = \
            super(EM_DE_Compound,self).get_parallel_instance(corpus_line)
        return ([w for (i,w) in german],[w for (i,w) in english])


class DiagonalCompoundDecoder(DiagonalAligner, DE_Compound_POS_decoder):
    """ Decoder that incorporates diagonal and POS priors
        and breaks up German compound words """

    def __init__(self, parameter_file):
        super(DiagonalCompoundDecoder,self).__init__(parameter_file)
        self.compounds = pickle.load(open("data/compound.dict",'rb'))#compounds_file)

    def get_alignment(self, german, english):
        """
        Returns Model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies the best english word (or NULL) to align to
        Applies a prior which assigns higher probability to alignments which preserve POS tags and are diagonally aligned
        Breaks up German compound words
        """
        alignment = []
        (german,english) = self.tag_and_stem_compounds(german,english)
        english.append((english[-1][0]+1,(self.null_val,self.null_val)))

        for (i, g_i) in german:
            german_len = len(german)
            best = -1
            bestscore = 0
            for (j, e_j) in english:
                english_len = len(english)

                if self.stem(e_j) == self.null_val: # handle null
                    prior = 1.0
                else:
                    prior = self.get_prior(tag_german=self.tag(g_i),position_german=i,\
                                           length_german=german_len,tag_english=self.tag(e_j),\
                                           position_english=j,length_english=english_len)
                val = prior * self.get_translation_prob(self.stem(g_i),self.stem(e_j))
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j

            if best < len(english)-1:
                yield (i,best) # don't yield anything for NULL alignment

class BeamDecoder(DiagonalCompoundDecoder):
    """ 
    Decoder that incorporates diagonal and POS priors
    and breaks up German compound words. 
    Runs a beam search based approach to dynamically build up the alignment
    """
    TUNE_JUMP_WEIGHT = 0.5
    TUNE_ALREADY_ALIGNED = 0.25
    TUNE_PREV_CONFIDENCE = 4.5
    TUNE_OVERRIDE_THRESH = 0.5
    BEAM_WIDTH = 3

    def get_prior(self, **features):
        wort = features.get("word_german",self.null_val)
        word = features.get("word_english",self.null_val)
        ortho_prior = 1+.1*(word[:min(3,len(wort))].lower()==wort[:min(3,len(word))].lower())

        pos_prior = 1+self.TUNE_POS_WEIGHT*(
           features.get("tag_german",self.null_val)==features.get("tag_english",self.null_val))
        if self.tagger.get_POS_class(features.get("tag_german",self.null_val))\
           ==self.tagger.get_POS_class(features.get("tag_english",self.null_val)):
            pos_prior+=1

        if features.get("tag_english",self.null_val)==self.null_val:
            null_prior = 2
        else:
            null_prior = 1

        de_idx = float(features.get("position_german", self.null_val))
        de_len = float(features.get("length_german", self.null_val))
        en_idx = float(features.get("position_english", self.null_val))
        en_len = float(features.get("length_english", self.null_val))

        diag_prior = (1 - abs(de_idx/de_len - en_idx/en_len)) * self.DIAG_WEIGHT

        duplicate_penalty = self.TUNE_ALREADY_ALIGNED if en_idx in features.get("en_aligned",[]) \
                            else 1
        return ortho_prior * pos_prior * diag_prior * duplicate_penalty

    def get_forward_prior(self, **features):
        prior = self.get_prior(**features)
        de_idx = float(features.get("position_german", self.null_val))
        de_len = float(features.get("length_german", self.null_val))
        en_idx = float(features.get("position_english", self.null_val))
        en_len = float(features.get("length_english", self.null_val))

        if features.get("last_align") and features.get("tag_german") not in POS_VERBS:
            return prior*(max(0,self.TUNE_PREV_CONFIDENCE*features.get("last_align")[1]*(2 - abs(en_idx-(features.get("last_align")[0][1]+1))))\
                          *self.TUNE_JUMP_WEIGHT + 0.1)
        elif features.get("tag_german") in POS_VERBS and features.get("tag_english") in POS_VERBS:
            return prior*(1+(en_len-en_idx)/en_len)
        else:
            return prior

    def get_backward_prior(self, **features):
        prior = self.get_prior(**features)
        de_idx = float(features.get("position_german", self.null_val))
        de_len = float(features.get("length_german", self.null_val))
        en_idx = float(features.get("position_english", self.null_val))
        en_len = float(features.get("length_english", self.null_val))

        if features.get("last_align") and features.get("tag_german") not in POS_VERBS:
            return prior*(max(0,self.TUNE_PREV_CONFIDENCE*features.get("last_align")[1]*(2 - abs(en_idx-(features.get("last_align")[0][1]-1))))\
                          *self.TUNE_JUMP_WEIGHT + 0.1)
        elif features.get("tag_german") in  POS_VERBS and features.get("tag_english") in POS_VERBS:
            return prior*(1+(en_len-en_idx)/en_len)
        else:
            return prior

    def alignment(self, al):
        return al[0]

    def score(self, al):
        return al[1]

    def get_alignment(self, german, english):
        """
        Returns Model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies the best english word (or NULL) to align to
        Applies a prior which assigns higher probability to alignments which preserve POS tags and are diagonally aligned
        Breaks up German compound words
        """
        german_words = german
        english_words = english+[self.null_val]
        alignments = [[[],0]]
        (german,english) = self.tag_and_stem_compounds(german,english)
        english.append((english[-1][0]+1,(self.null_val,self.null_val)))

        # Pick a word in each language and consider adding that to the alignments under consideration
        for (i, g_i) in german:
            old_alignments = alignments
            alignments = []

            german_len = len(german)
            for (j, e_j) in english:
                english_len = len(english)

                for alignment in old_alignments:
                    try:
                        last = self.alignment(alignment)[-1]
                    except:
                        last = None
                    en_prevs = set([l for (k,l) in self.alignment(alignment)])
                    prior = self.get_forward_prior(tag_german=self.tag(g_i),position_german=i,\
                                           length_german=german_len,tag_english=self.tag(e_j),\
                                           position_english=j,length_english=english_len,
                                           last_align=last,stem_german=self.stem(g_i),
                                           stem_english=self.stem(e_j),word_german=german_words[i],
                                           word_english=english_words[j],
                                                   en_aligned=en_prevs)

                    score = prior * self.get_translation_prob(self.stem(g_i),self.stem(e_j))
                    
                    #Decide whether to throw out old alignment
                    if j < len(english)-1 and j in en_prevs:
                        wl = min([al for al in self.alignment(alignment)],key=lambda(x):x[1])
                        if score < wl*self.TUNE_OVERRIDE_THRESH:
                            continue
                        if score*self.TUNE_OVERRIDE_THRESH > wl:
                            alignment.remove(wl)

                    # Here we add the possible alignment to our consideration
                    # and throw out the lowest scoring alignment if we have too many
                    if len(alignments)==self.BEAM_WIDTH:
                        if j < len(english)-1:
                            alignments.append([self.alignment(alignment)+[((i,j),score)],
                                               self.score(alignment)+score]) 
                        else: # only adjust score for NULL alignment
                            alignments.append([self.alignment(alignment),
                                               self.score(alignment)+score])
                        ##remove worst alignment
                        #alignments.remove(min(alignments,key=lambda(x):self.score(x)))
                    else:
                        if j < len(english)-1:
                            alignments.append([self.alignment(alignment)+[((i,j),score)],
                                               self.score(alignment)+score])
                        else:
                            alignments.append([self.alignment(alignment),
                                               self.score(alignment)+score])
            
                while len(alignments)>self.BEAM_WIDTH:
                    #remove worst alignment
                    alignments.remove(min(alignments,key=lambda(x):self.score(x)))
        forwards = alignments
        
        alignments = [[[],0]]
        # Pick a word in each language and consider adding that to the alignments under consideration
        for (i, g_i) in german[::-1]:
            old_alignments = alignments
            alignments = []

            german_len = len(german)
            for (j, e_j) in english:
                english_len = len(english)

                for alignment in old_alignments:

                    try:
                        last = self.alignment(alignment)[-1]
                    except:
                        last = None
                    prior = self.get_backward_prior(tag_german=self.tag(g_i),position_german=i,\
                                           length_german=german_len,tag_english=self.tag(e_j),\
                                           position_english=j,length_english=english_len,
                                           last_align=last,stem_german=self.stem(g_i),
                                           stem_english=self.stem(e_j),word_german=german_words[i],
                                           word_english=english_words[j],
                                           en_aligned=set([l for (k,l) in 
                                                           self.alignment(alignment)]))

                    score = prior * self.get_translation_prob(self.stem(g_i),self.stem(e_j))

                    #Decide whether to throw out old alignment
                    if j < len(english)-1 and j in en_prevs:
                        wl = min([al for al in self.alignment(alignment)],key=lambda(x):x[1])
                        if score < wl*self.TUNE_OVERRIDE_THRESH:
                            continue
                        if score*self.TUNE_OVERRIDE_THRESH > wl:
                            alignment.remove(wl)
                    
                    # Here we add the possible alignment to our consideration
                    # and throw out the lowest scoring alignment if we have too many
                    if len(alignments)==self.BEAM_WIDTH:
                        if j < len(english)-1:
                            alignments.append([self.alignment(alignment)+[((i,j),score)],
                                               self.score(alignment)+score]) 
                        else: # only adjust score for NULL alignment
                            alignments.append([self.alignment(alignment),
                                               self.score(alignment)+score])
                        ##remove worst alignment
                        #alignments.remove(min(alignments,key=lambda(x):self.score(x)))
                    else:
                        if j < len(english)-1:
                            alignments.append([self.alignment(alignment)+[((i,j),score)],
                                               self.score(alignment)+score])
                        else:
                            alignments.append([self.alignment(alignment),
                                               self.score(alignment)+score])
                
                while len(alignments)>self.BEAM_WIDTH:
                    #remove worst alignment
                    alignments.remove(min(alignments,key=lambda(x):self.score(x)))

        backwards = alignments

        # Ultimately, we return the highest scoring alignment
        alignment = self.alignment(max(forwards+backwards,key=lambda(x):self.score(x)))
        return [al for (al,score) in alignment if al[1]<len(english)-1]
