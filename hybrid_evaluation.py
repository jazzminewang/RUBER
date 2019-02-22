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

    # best logic:
    # - dataset (ADEM, personachat, or twitter)
    # - mode (training or validation)
    # --reply_file (optional)

    # File structure

    # data
    # -- word2vec embeddings
    # - ADEM
    #   - validation
        #   - [ assorted files for validation ]
    #   - train
        #   - [ assorted files for training ]
    # - personachat
    #   - validation
            #   - [ assorted files for validation ]
    #   - train [RENAME]
            #   - [ assorted files for training ]
    # - twitter
    #   - validation
            #   - [ assorted files for validation ]
    #   - train
            #   - [ assorted files for training ]

    data_dir = "./data"

    parser.add_argument('dataset')
    parser.add_argument('mode')
    args = parser.parse_args()

    dataset = args.dataset #ADEM, personachat, or twitter
    mode = args.mode # train or validate

    qmax_length, rmax_length = [20, 30]

    print("Mode: " + args.mode)

    training_fquery = dataset + "/train/queries.txt"
    training_freply = dataset + "/train/replies.txt"
    validation_fquery = dataset + "/validation/queries.txt"
    if args.reply_file and args.dataset=="ADEM":
        validation_freply_true = dataset + "validation/true.txt"
        validation_freply_generated = dataset + args.reply_file
    else:
        validation_freply_true = dataset + "validation/replies.txt.true.sub"
        validation_freply_generated = dataset + "/validation/replies.txt.sub"

    if args.mode == "train":
        is_training=True
    else: 
        is_training=False

    """word2vec file"""
    frword2vec = 'GoogleNews-vectors-negative300.txt'

    print("Initializing Hybrid object with " + training_fquery + " as training query file")
    hybrid = Hybrid(data_dir, frword2vec, '%s.embed'%training_fquery, '%s.embed'%training_freply, is_training=is_training)
    """test"""
    if args.mode == "validate":
        scores, ref_scores, norm_ref_scores, unref_scores, norm_unref_scores = hybrid.scores(data_dir, validation_fquery, validation_freply_true, validation_freply_generated, '%s.vocab%d'%(training_fquery, qmax_length),'%s.vocab%d'%(training_freply, rmax_length))
        csv_title = './results/' + dataset + validation_freply_generated + str(int(time.time())) + '.csv'

        """write results to CSV"""
        with open(csv_title, 'w+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
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
