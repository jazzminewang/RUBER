__author__ = 'liming-vie'

import os
from referenced_metric import Referenced
from unreferenced_metric import Unreferenced
import numpy as np
import sys
import csv
from numpy import median, mean

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
	        ret = [coefficient * (s - smin) / diff for s in scores]
	else:
	    ret = [(s - smin) / diff for s in scores]
        return ret

    def scores(self, data_dir, fquery ,freply, fgenerated, fqvocab, frvocab):
        ref_scores = self.ref.scores(data_dir, freply, fgenerated)
        norm_ref_scores = self.normalize(ref_scores, coefficient=2)
        
        unref_scores = self.unref.scores(data_dir, fquery, fgenerated,
                fqvocab, frvocab)
        norm_unref_scores = self.normalize(unref_scores, coefficient=2)

        return [np.mean([a,b]) for a,b in zip(norm_ref_scores, norm_unref_scores)], ref_scores, norm_ref_scores, unref_scores, norm_unref_scores

if __name__ == '__main__':
    train_dir = 'data'
    data_dir = 'data'
    qmax_length, rmax_length = [20, 30]

    """ for validation """
    # embedding matrix file for query and reply
    fquery = "personachat/validation_personachat/queries_validation.txt"
    freply = "personachat/validation_personachat/replies_validation.txt"
    """ for training """
    #fquery = "personachat/queries.txt"
    #freply = "personachat/replies.txt"

    """word2vec file"""
    frword2vec = 'GoogleNews-vectors-negative300.txt'

    print("Initializing Hybrid object")
    hybrid = Hybrid(data_dir, frword2vec, '%s.embed'%fquery, '%s.embed'%freply)

    """test"""
    print("Getting scores")
    # scores = hybrid.unref.scores(data_dir, '%s.sub'%fquery, '%s.sub'%freply, "%s.vocab%d"%(fquery,qmax_length), "%s.vocab%d"%(freply, rmax_length))
    scores, ref_scores, norm_ref_scores, unref_scores, norm_unref_scores = hybrid.scores(data_dir, '%s.sub'%fquery, '%s.true.sub'%freply, '%s.sub'%freply, '%s.vocab%d'%(fquery, qmax_length),'%s.vocab%d'%(freply, rmax_length))
    
    """write results to CSV""""
    with open('results' + int(time.time()) + '.csv', 'wb') as csvfile:
         writer = csv.writer(csvfile, delimiter=',')
         # Name the columns
         column_titles = ["Query", "Scored reply", "Ground truth reply", "Score", "Ref score", "Normed ref score", "Unref score", "Normed unref score"]
         writer.writerow([col for col in column_titles])
         
         with open(data_dir + '%s.sub'%fquery, "r") as queries, open(data_dir + '%s.sub'%freply, "r") as scored_replies, open(data_dir + '%s.true.sub'%freply, "r") as true_replies:
            #Write rows
            for query, scored_reply, true_reply, score, ref_score, norm_ref_score, unref_score, norm_unref_score in \
                queries, scored_replies, true_replies, scores, ref_scores, norm_ref_scores, unref_scores, norm_unref_scores:
                writer.writerow([query, scored_reply, true_reply, score, ref_score, norm_ref_score, unref_score, norm_unref_score])
    csvfile.close()

    print("max score: {}, min score: {}, median score: {}, mean score: {}, median norm ref: {}, median norm unref: {}, min unnorm unref: {}, max unnorm unref: {}").format(
        max(scores), min(scores), median(scores), mean(scores), median(norm_ref_scores), median(norm_unref_scores), min(unref_scores), max(unref_scores)
    )

    """train"""
    #hybrid.train_unref(data_dir, fquery, freply)


# 1. go to datahelpers, change their files. restructure data.
# 2. run train for unref metric --> checkpoint file. 
# 3. run inference. 
