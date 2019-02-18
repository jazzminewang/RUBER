__author__ = 'liming-vie'

import os
from referenced_metric import Referenced
from unreferenced_metric import Unreferenced
import numpy as np
import sys
import csv
from numpy import median, mean
import argparse
import time

class Hybrid():
    def __init__(self,
            data_dir,
            frword2vec,
            fqembed,
            frembed,
            qmax_length=20,
            rmax_length=30,
            ref_method='max_min',
            gru_units=128, mlp_units=[256, 512, 128],
            is_training=True
        ):
        print("Initializing referenced model")
        self.ref=Referenced(data_dir, frword2vec, ref_method)
        print("Initializing unreferenced model")
        self.unref=Unreferenced(qmax_length, rmax_length,
                os.path.join(data_dir,fqembed),
                os.path.join(data_dir,frembed),
                gru_units, mlp_units,
                train_dir=train_dir, 
                is_training=is_training)

    def train_unref(self, data_dir, fquery, freply):
        print("training unreferenced metric")
        self.unref.train(data_dir, fquery, freply)

    def normalize(self, scores, smin=None, smax=None, coefficient=None, smallest_value=0):
        if not smin and not smax:
	    smin = min(scores)
            smax = max(scores)
            diff = smax - smin
	# normalize to [0-2] instead to fit RUBER human scores
        else:
	    smin = smin
	    diff = smax - smin
        if coefficient:
	        ret = [smallest_value + (coefficient * (s - smin) / diff) for s in scores]
	else:
	    ret = [smallest_value + ((s - smin) / diff) for s in scores]
        return ret

    def scores(self, data_dir, fquery ,freply, fgenerated, fqvocab, frvocab):
	print("training dir is ")
	print(train_dir)
        ref_scores = self.ref.scores(data_dir, freply, fgenerated, train_dir=train_dir)
	norm_ref_scores = self.normalize(ref_scores, coefficient=4, smallest_value=1)
        
        unref_scores = self.unref.scores(data_dir, fquery, fgenerated,
                fqvocab, frvocab, init=False, train_dir=train_dir)
        norm_unref_scores = self.normalize(unref_scores, coefficient=4, smallest_value=1)

        return [np.mean([a,b]) for a,b in zip(norm_ref_scores, norm_unref_scores)], ref_scores, norm_ref_scores, unref_scores, norm_unref_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode') 
    parser.add_argument('-reply_file')
    parser.add_argument('-dataset')
    args = parser.parse_args()

    train_dir = 'ADEM_data/data'
    data_dir = hybrid_dir = 'data'
    qmax_length, rmax_length = [20, 30]

    print("Mode: " + args.mode)
    is_training=True

    if args.mode == "eval_ADEM":
        data_dir = 'ADEM_data/data'
	if args.dataset == "twitter":
             hybrid_fquery = "twitter_data/train/queries.txt"
             hybrid_freply = "twitter_data/train/replies.txt"
	else:
        	hybrid_fquery = 'personachat/better_turns/queries.txt'
        	hybrid_freply = 'personachat/better_turns/replies.txt'
        fquery = "queries.txt"
        freply = args.reply_file
        is_training=False
    else:
        if args.mode == "eval_personachat":
            is_training=False
	    hybrid_dir = 'data'
            train_dir = 'data'
	if args.dataset == "twitter":
             fquery =  hybrid_fquery = "twitter_data/train/queries.txt"
             freply =  hybrid_freply = "twitter_data/train/replies.txt"
	     train_dir = "twitter_data/train"
        else:
	     fquery =  hybrid_fquery = "personachat/better_turns/queries.txt"
             freply =  hybrid_freply = "personachat/better_turns/replies.txt"

    """word2vec file"""
    frword2vec = 'GoogleNews-vectors-negative300.txt'

    print("Initializing Hybrid object with " + hybrid_fquery + " as training query file")
    hybrid = Hybrid(hybrid_dir, frword2vec, '%s.embed'%hybrid_fquery, '%s.embed'%hybrid_freply, is_training=is_training)

    if args.mode == "eval_personachat":
        # use validation queries and replies
        fquery = "personachat/validation/queries.txt"
        freply = "personachat/validation/replies.txt"

    """test"""
    if args.mode != "train":
        print("Getting scores")


        if args.mode == "eval_ADEM": 
            print("Scoring ADEM data")
	    print("training directory is " + data_dir)
            scores, ref_scores, norm_ref_scores, unref_scores, norm_unref_scores = hybrid.scores(hybrid_dir, fquery, 'true.txt' ,freply, '%s.vocab%d'%(hybrid_fquery, qmax_length),'%s.vocab%d'%(hybrid_freply, rmax_length))
	    csv_title = './results/' + freply + str(int(time.time())) + '.csv'
        elif args.mode == "eval_personachat":
            scores, ref_scores, norm_ref_scores, unref_scores, norm_unref_scores = hybrid.scores(hybrid_dir, '%s.sub'%fquery, '%s.true.sub'%freply, '%s.sub'%freply, '%s.vocab%d'%(fquery, qmax_length),'%s.vocab%d'%(freply, rmax_length))
            csv_title = './results/personachat/' +  str(int(time.time())) + '.csv'

        """write results to CSV"""
        with open(csv_title, 'w+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            # Name the columns
            column_titles = ["Query", "Scored reply", "Ground truth reply", "Score", "Ref score", "Normed ref score", "Unref score", "Normed unref score"]
            writer.writerow([col for col in column_titles])
            
            if args.mode != "eval_ADEM":
                fquery = '%s.sub'%fquery
		true = '%s.true.sub'%freply
                freply = '%s.sub'%freply
            else:
                true = "true.txt"

            with open(data_dir + "/" + fquery, "r") as queries, \
                    open(data_dir+ "/" + freply, "r") as scored_replies, \
                        open(data_dir+ "/"  + true, "r") as true_replies:
                for query, scored_reply, true_reply, score, ref_score, norm_ref_score, unref_score, norm_unref_score in zip(queries, scored_replies, true_replies, scores, ref_scores, norm_ref_scores, unref_scores, norm_unref_scores):
                    query = query.rstrip()
                    scored_reply = scored_reply.rstrip()
                    true_reply = true_reply.rstrip()
                    writer.writerow([query, scored_reply, true_reply, score, ref_score, norm_ref_score, unref_score, norm_unref_score])
        csvfile.close()

        print("max score: {}, min score: {}, median score: {}, mean score: {}, median norm ref: {}, min unnorm ref: {}, max unnorm ref: {}, median norm unref: {}, min unnorm unref: {}, max unnorm unref: {}").format(
            max(scores), min(scores), median(scores), mean(scores), median(norm_ref_scores), min(ref_scores), max(ref_scores), median(norm_unref_scores), min(unref_scores), max(unref_scores)
        )

        print("Wrote  model results to " + csv_title)
    else:
        """train"""
        print("Training")
	print("Data dir is " + data_dir)
        hybrid.train_unref(data_dir, fquery, freply)
