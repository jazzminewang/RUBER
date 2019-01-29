__author__ = 'liming-vie'

import os
from referenced_metric import Referenced
from unreferenced_metric import Unreferenced
import numpy as np
import sys

class Hybrid():
    def __init__(self,
            data_dir,
            frword2vec,
            fqembed,
            frembed,
            qmax_length=20,
            rmax_length=30,
            ref_method='max_min',
            gru_units=128, mlp_units=[256, 512, 128]
        ):
        print("Initializing referenced model")
        self.ref=Referenced(data_dir, frword2vec, ref_method)
        print("Initializing unreferenced model")
        self.unref=Unreferenced(qmax_length, rmax_length,
                os.path.join(data_dir,fqembed),
                os.path.join(data_dir,frembed),
                gru_units, mlp_units,
                train_dir=train_dir)

    def train_unref(self, data_dir, fquery, freply):
        print("training unreferenced metric")
        self.unref.train(data_dir, fquery, freply)

    def normalize(self, scores, smin=None, smax=None, coefficient=None):
        if not smin and not smax:
	    smin = min(scores)
            smax = max(scores)
            diff = smax - smin
	# normalize to [0-2] instead to fit RUBER human scores
        else:
	    smin = smin
	    diff = smax - smin
        if coefficient:
	    print("coefficient is " + str(coefficient))
	    ret = [2 * (s - smin) / diff for s in scores]
	else:
	    ret = [(s - smin) / diff for s in scores]
        return ret

    def scores(self, data_dir, fquery ,freply, fgenerated, fqvocab, frvocab):
	ref_scores = self.ref.scores(data_dir, freply, fgenerated)
	print("min, max unnorm ref")
	print(min(ref_scores), max(ref_scores))
        ref_scores = self.normalize(ref_scores, coefficient=2)
	print("min, max norm ref")
        print(min(ref_scores), max(ref_scores))
        unref_scores = self.unref.scores(data_dir, fquery, fgenerated,
                fqvocab, frvocab)
	print("min unnormalized unref metric score")
	print(min(unref_scores))
	print("max of same")
	print(max(unref_scores))
        unref_scores = self.normalize(unref_scores, coefficient=2)
        #heuristic used is minimum - changed to arithmetic mean
	print("max norm unref")
	print(min(unref_scores))
        print("max of same")
        print(max(unref_scores))
        return [np.mean([a,b]) for a,b in zip(ref_scores, unref_scores)]

if __name__ == '__main__':
    train_dir = 'data'
    data_dir = 'data'
    qmax_length, rmax_length = [20, 30]

    # fquery = [] 
    # freply = []
    """ for validation """
    # embedding matrix file for query and reply
    fquery = "personachat/validation_personachat/queries_validation.txt"
    freply = "personachat/validation_personachat/replies_validation.txt"
    """ for training """
    #fquery = "personachat/queries.txt"
    #freply = "personachat/replies.txt"
    # to do - insert word2vec txt file?
    frword2vec = 'GoogleNews-vectors-negative300.txt'

    print("Initializing Hybrid object")
    hybrid = Hybrid(data_dir, frword2vec, '%s.embed'%fquery, '%s.embed'%freply)
    """test"""
    
    print("Getting scores")
    scores = hybrid.unref.scores(data_dir, '%s.sub'%fquery, '%s.sub'%freply, "%s.vocab%d"%(fquery,qmax_length), "%s.vocab%d"%(freply, rmax_length))
    scores = hybrid.scores(data_dir, '%s.sub'%fquery, '%s.true.sub'%freply, '%s.sub'%freply, '%s.vocab%d'%(fquery, qmax_length),'%s.vocab%d'%(freply, rmax_length))
    min_score = sys.float_info.max
    max_score = 0
    for i, s in enumerate(scores):
        #print i,s
	min_score = min(s, min_score)
	max_score = max(s, max_score)
    print 'avg:%f'%(sum(scores)/len(scores))
    print 'min:%f'%min_score
    print 'max:%f'%max_score
    """train"""
    #hybrid.train_unref(data_dir, fquery, freply)


# 1. go to datahelpers, change their files. restructure data.
# 2. run train for unref metric --> checkpoint file. 
# 3. run inference. 
