"""
exploration_utils.py
Utility functions for preliminary data exploration.
"""

from collections import Counter
import numpy as np
import random

def ngram_dataset_freq(dataset, key, n=1, verbose=False, top_n=50):
    """Frequency distribution of ngrams across the entire dataset."""
    fdist = Counter()
    num_descriptions = 0
    lens = []
    diversity = []
    untokenized = []
    
    for example in dataset:
        descriptions = []
        if isinstance(example[key][0], list):
            for description in example[key]:
                if description[0] == '<': # Remove start/end tokens
                    start, end = 1, -1
                else:
                    start, end = 0, len(description)
                descriptions += [description[start:end]]

        else:
            description = example[key]
            if description[0] == '<': # Remove start/end tokens
                start, end = 1, -1
            else:
                start, end = 0, len(description)
            descriptions += [description[start:end]]
        
        # n-grams on a per-example basis
        fdist_in_task = Counter()
        for description in descriptions:
            if len(description) > 0:
                for i in range(len(description) - n + 1):
                    fdist[tuple(description[i:i+n])] += 1
                    fdist_in_task[tuple(description[i:i+n])] += 1
                lens.append(len(description))
                num_descriptions += len(descriptions)
    
                untokenized.append(" ".join(description))
        if len(descriptions) > 1:
            diversity.append(float(len(fdist_in_task)) / np.sum(list(fdist_in_task.values())))
                
    if verbose:
        if top_n:
            common_ngrams = [(word, num) for (word, num) in fdist.most_common(100) if (len(word) > 1 or len(word[0]) > 1 )][:top_n]
        else:
            common_ngrams = [(word, num) for (word, num) in fdist.most_common() if (len(word) > 1 or len(word[0]) > 1 )]
        
        print("Printing for ngram, n=%d" % n)
        print("Num descriptions: %d" % num_descriptions)
        print("Description avg: %d, med: %d, min: %d, max: %d" % (np.mean(lens), np.median(lens), np.min(lens), np.max(lens)))
        if len(diversity) > 1:
            print("Ngram diversity within tasks w. multiple examples: avg: %f, med: %f, min: %f, max: %d " % (np.mean(diversity), np.median(diversity), np.min(diversity), np.max(diversity)))
        print("Vocabulary size: %d" % len(fdist) )
        print("Ngrams with freq > 10: %d" % len([word for word in fdist if fdist[word] > 10]))
        print("Total ngram in corpus: %d" % np.sum(list(fdist.values())))
        print("50 most common: (not including letters): " + str(common_ngrams))
        print("Sample descriptions: ")
        rand = random.sample(untokenized, 5)
        for description in rand:
            print(description)
        
    return fdist

def ngram_cross_dataset_freq(fdists, verbose=False):
    """Frequency distributions intersected across several fdists from disparate datasets."""
    summed_fdist = Counter()
    for fdist in fdists:
        summed_fdist += fdist
        
    # Only get the intersecting vocabulary
    intersect_vocab = set.intersection(*[set(fdist.keys()) for fdist in fdists])
    intersected_fdist=Counter()
    for vocab in intersect_vocab:
        intersected_fdist[vocab] = summed_fdist[vocab]
    
    if verbose:
        common_ngrams = intersected_fdist.most_common(50)
        print("Cross dataset frequency for %d datasets." % len(fdists))
        print("Original vocabulary sizes are %s" % str([len(fdist) for fdist in fdists]))
        print("Combined vocabulary size is %d; intersected vocab is: %d" %(len(summed_fdist), len(intersect_vocab)))
        
        print("Intersection ngrams with freq > 10: %d" % len([word for word in intersected_fdist if intersected_fdist[word] > 10]))
        print("50 most common: (not including letters): " + str(common_ngrams))
    return intersected_fdist