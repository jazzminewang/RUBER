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
                is_training=is_training)

    def train_unref(self, data_dir, fquery, freply, validation_fquery, validation_freply_true):
        print("training unreferenced metric")
        self.unref.train(data_dir, fquery, freply, validation_fquery, validation_freply_true)
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

    def scores(self, data_dir, fquery ,freply, fgenerated, fqvocab, frvocab, checkpoint_dir):
        ref_scores = self.ref.scores(data_dir, freply, fgenerated)
	norm_ref_scores = self.normalize(ref_scores, coefficient=4, smallest_value=1)
        
        unref_scores = self.unref.scores(data_dir, fquery, fgenerated,
                fqvocab, frvocab, checkpoint_dir, init=False)
        norm_unref_scores = self.normalize(unref_scores, coefficient=4, smallest_value=1)

        return [np.mean([a,b]) for a,b in zip(norm_ref_scores, norm_unref_scores)], ref_scores, norm_ref_scores, unref_scores, norm_unref_scores

    def validate_to_csv(self, checkpoint_dir, data_dir, validation_fquery, validation_freply_generated, validation_freply_true, training_fquery, qmax_length, training_freply, rmax_length, validation_dataset):
        scores, ref_scores, norm_ref_scores, unref_scores, norm_unref_scores \
                = self.scores(data_dir, validation_fquery, validation_freply_true, validation_freply_generated, \
                    '%s.vocab%d'%(training_fquery, qmax_length),'%s.vocab%d'%(training_freply, rmax_length), checkpoint_dir)
	
        csv_dir = os.path.join('./results', validation_dataset, "validation", checkpoint_dir)

	print(csv_dir)
	reply_file_path = validation_freply_generated.split("/")
	reply_file = reply_file_path[len(reply_file_path) - 1]
	print(reply_file)
        csv_title = os.path.join(csv_dir, reply_file.rstrip(".txt") + ".csv")
	print("Csv title: ")
	print(csv_title)
	if not os.path.exists(csv_dir):
	    os.makedirs(csv_dir)
        
        """write results to CSV"""
        with open(csv_title, 'w+') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            column_titles = ["Query", "Scored reply", "Ground truth reply", "Score", "Ref score", "Normed ref score", "Unref score", "Normed unref score"]
            writer.writerow([col for col in column_titles])
            
            with open(os.path.join(data_dir, validation_fquery), "r") as queries, \
                    open(os.path.join(data_dir, validation_freply_generated), "r") as scored_replies, \
                        open(os.path.join(data_dir, validation_freply_true), "r") as true_replies:
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

    parser.add_argument('train_dataset')
    parser.add_argument('validation_dataset')
    parser.add_argument('mode')
    parser.add_argument('-reply_files', nargs='+')
    parser.add_argument('-checkpoint_dirs', nargs='+')
    args = parser.parse_args()

    train_dataset = args.train_dataset #personachat or twitter
    validation_dataset = args.validation_dataset #ADEM, personachat
    mode = args.mode # train or validate
    
    if args.reply_files and args.checkpoint_dirs:
        checkpoint_dirs = args.checkpoint_dirs[0].split(" ")
        reply_files = args.reply_files[0].split(" ")
        print("Checkpoint dirs: ")
        print(checkpoint_dirs)
	print("Example checkpoint dir: ")
	print(checkpoint_dirs[0])
        print("Reply files: ")
	print(reply_files)

    qmax_length, rmax_length = [20, 30]

    print("Mode: " + args.mode)

    training_fquery = train_dataset + "/train/queries.txt"
    training_freply = train_dataset + "/train/replies.txt"
    
    if args.validation_dataset =="ADEM":
        validation_fquery = validation_dataset + "/validation/queries.txt"
        validation_freply_true = validation_dataset + "/validation/true.txt"
        if reply_files:
             validation_freply_generated = validation_dataset + "/validation/" + reply_files[0]
        else:
             validation_freply_generated = validation_dataset + "/validation/hred_replies.txt"
    else:
        #personachat validation on test set
        validation_fquery = validation_dataset + "/test/queries.txt"
        validation_freply_true = validation_dataset + "/test/ground_truth.txt"
        if reply_files:
             validation_freply_generated = validation_dataset + "/test/" + reply_files[0]
        else:
             validation_freply_generated = validation_dataset + "/test/high_quality_responses.txt"

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
	print("First checkpoint dir " + checkpoint_dirs[0])
	print("First reply file " + reply_files[0])
        for checkpoint_dir in checkpoint_dirs:
	    for reply_file in reply_files: 
                print("Validating " + checkpoint_dir + " model with " + reply_file + " replies.")
                validation_freply_generated = os.path.join(validation_dataset, "validation", reply_file)

                hybrid.validate_to_csv(
                    checkpoint_dir, data_dir, validation_fquery, \
                        validation_freply_generated, validation_freply_true, \
                            training_fquery, qmax_length, training_freply, rmax_length, validation_dataset)

    else:
        """train"""
        print("Training")
	print("Data dir is " + data_dir)
        hybrid.train_unref(data_dir, training_fquery, training_freply, validation_fquery, validation_freply_true)
