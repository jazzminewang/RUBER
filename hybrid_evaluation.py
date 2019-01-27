__author__ = 'liming-vie'

import os
from referenced_metric import Referenced
from unreferenced_metric import Unreferenced
import numpy as np

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

    def normalize(self, scores):
        smin = min(scores)
        smax = max(scores)
        diff = smax - smin
	# normalize to [0-2] instead to fit RUBER human scores
        ret = [2*(s - smin) / diff for s in scores]
        return ret

    def scores(self, data_dir, fquery ,freply, fgenerated, fqvocab, frvocab):
        print("Getting referenced scores")
	ref_scores = self.ref.scores(data_dir, freply, fgenerated)
        ref_scores = self.normalize(ref_scores)

	print("Got referenced scores")
	print(ref_scores)
	print("Getting unreferenced scores")
        unref_scores = self.unref.scores(data_dir, fquery, fgenerated,
                fqvocab, frvocab)
        unref_scores = self.normalize(unref_scores)
	print("Got unreferenced scores")
        #heuristic used is minimum - changed to arithmetic mean
        return [np.mean([a,b]) for a,b in zip(ref_scores, unref_scores)]

if __name__ == '__main__':
    train_dir = 'data'
    data_dir = 'data'
    qmax_length, rmax_length = [20, 30]

    # fquery = [] 
    # freply = []

    # embedding matrix file for query and reply
    fquery = "personachat/queries.txt"
    freply = "personachat/replies.txt"

    # to do - insert word2vec txt file?
    frword2vec = 'GoogleNews-vectors-negative300.txt'

    print("Initializing Hybrid object")
    hybrid = Hybrid(data_dir, frword2vec, '%s.embed'%fquery, '%s.embed'%freply)
    """test"""
    
    print("Getting scores")
    scores = hybrid.unref.scores(data_dir, '%s.sub'%fquery, '%s.sub'%freply, "%s.vocab%d"%(fquery,qmax_length), "%s.vocab%d"%(freply, rmax_length))
    scores = hybrid.scores(data_dir, '%s.sub'%fquery, '%s.true.sub'%freply, '%s.sub'%freply, '%s.vocab%d'%(fquery, qmax_length),'%s.vocab%d'%(freply, rmax_length))
    for i, s in enumerate(scores):
        print i,s
    print 'avg:%f'%(sum(scores)/len(scores))
   
    """train"""
    #hybrid.train_unref(data_dir, fquery, freply)


# 1. go to datahelpers, change their files. restructure data.
# 2. run train for unref metric --> checkpoint file. 
# 3. run inference. 
