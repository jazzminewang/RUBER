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
            gru_num_units=512, 
            mlp_units=[256, 512, 128],
            init_learning_rate=0.001,
            margin=0.5, 
            batch_norm=False,
            is_training=True,
            train_dataset='',
	        log_dir="training",
            scramble=False,
            additional_negative_samples='',
        ):
        print("Initializing referenced model")
        self.ref=Referenced(data_dir, frword2vec, ref_method)
        print("Initializing unreferenced model with log_dir " + log_dir + " and additional samples: " + additional_negative_samples)
        self.unref=Unreferenced(qmax_length, rmax_length,
                os.path.join(data_dir,fqembed),
                os.path.join(data_dir,frembed),
                gru_num_units=gru_num_units, 
                mlp_units=mlp_units,
                init_learning_rate=init_learning_rate,
                margin=margin,
                is_training=is_training,
                batch_norm=batch_norm,
                train_dataset=train_dataset,
		        log_dir=log_dir,
                scramble=scramble,
                additional_negative_samples=additional_negative_samples
                )

    def train_unref(self, data_dir, fquery, freply, validation_fquery, validation_freply_true, scramble=False):
        print("training unreferenced metric with scramble: " + str(scramble))
        self.unref.train(data_dir, fquery, freply, validation_fquery, validation_freply_true, scramble=scramble)
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

    def validate_to_csv(self, checkpoint_dir, data_dir, validation_fquery, validation_freply_generated, validation_freply_true, training_fquery, qmax_length, training_freply, rmax_length, train_dataset, validation_dataset):
	print("Starting validation")
        scores, ref_scores, norm_ref_scores, unref_scores, norm_unref_scores \
                = self.scores(data_dir, validation_fquery, validation_freply_true, validation_freply_generated, \
                    '%s.vocab%d'%(training_fquery, qmax_length),'%s.vocab%d'%(training_freply, rmax_length), checkpoint_dir)
	
        csv_dir = os.path.join('./results', checkpoint_dir, validation_dataset)

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

    data_dir = "./data"

    parser.add_argument('train_dataset')
    parser.add_argument('validation_dataset')
    parser.add_argument('mode')

    # Training
    parser.add_argument('-log_dir', default="experiments/")

    # Hyperparameters
    parser.add_argument('-gru_num_units', type=int)
    parser.add_argument('-init_learning_rate', type=float)
    parser.add_argument('-margin', type=float)
    parser.add_argument('-batch_norm', type=bool, default=False)
    parser.add_argument('-scramble', type=bool, default=False)
    parser.add_argument('-additional_negative_samples', type=bool, default=False)

    # Evaluation
    parser.add_argument('-checkpoint_dir')

    args = parser.parse_args()

    train_dataset = args.train_dataset #personachat or twitter
    validation_dataset = args.validation_dataset #ADEM, personachat
    mode = args.mode # train or validate

    log_dir = args.log_dir

    batch_norm = args.batch_norm 
    gru_num_units = args.gru_num_units

    if args.mode == "train":
        is_training=True
    else: 
        is_training=False
    
    if is_training:
        init_learning_rate = float(args.init_learning_rate) / 1000
        margin = float(args.margin) / 100
    else:
        # If evaluating, set arbitrary values
        init_learning_rate=0.001
        margin=0.5

    training_fquery = os.path.join(train_dataset, "train", "queries.txt")
    training_freply = os.path.join(train_dataset, "train", "replies.txt")

    # Choose ADEM or personachat validation
    if args.validation_dataset =="ADEM":
        sub_dir_validate = "validation"
        reply_files = ["human_replies.txt", "de_replies.txt", "tfidf_replies.txt", "hred_replies.txt"]
    else:
        sub_dir_validate = "test"
        reply_files = ["high_quality_responses.txt", "kevmemnn.txt", "language_model.txt", "random_response.txt", "seq_to_seq.txt", "tf_idf.txt"]

    validation_fquery = os.path.join(validation_dataset, sub_dir_validate, "queries.txt")
    validation_freply_true = os.path.join(validation_dataset, sub_dir_validate, "ground_truth.txt")

    """word2vec file"""
    frword2vec = 'GoogleNews-vectors-negative300.txt'
    qmax_length, rmax_length = [20, 30]
    
    if args.additional_negative_samples:
        additional_negative_samples = os.path.join("generated_responses", "personachat_train_responses.txt")
    else:
        additional_negative_samples = ''
    print("Mode: " + args.mode)
    print("Initializing Hybrid object with " + training_fquery + " as training query file")

    hybrid = Hybrid(
        data_dir, 
        frword2vec, 
        '%s.embed'%training_fquery, 
        '%s.embed'%training_freply, 
        gru_num_units=gru_num_units,
        init_learning_rate=init_learning_rate,
        margin=margin,
        batch_norm=batch_norm,
        is_training=is_training, 
        train_dataset=train_dataset,
	    log_dir=log_dir,
        scramble=args.scramble,
        additional_negative_samples=additional_negative_samples
        )
    
    if not is_training:
        """test"""
        experiment_folder = log_dir

        print("reply files: ")
        print(reply_files)
	for reply_file in reply_files: 
            checkpoint_dir = os.path.join(experiment_folder, args.checkpoint_dir)
            print("Validating " + checkpoint_dir + " model with " + reply_file + " replies.")
            validation_freply_generated = os.path.join(validation_dataset, sub_dir_validate, reply_file)

            hybrid.validate_to_csv(
                checkpoint_dir, data_dir, validation_fquery, \
                    validation_freply_generated, validation_freply_true, \
                        training_fquery, qmax_length, training_freply, rmax_length, train_dataset, validation_dataset)

    else:
        """train"""
        print("Training")
        print("hybrid scramble is: " + str(args.scramble))
        hybrid.train_unref(data_dir, training_fquery, training_freply, validation_fquery, validation_freply_true, scramble=args.scramble)
