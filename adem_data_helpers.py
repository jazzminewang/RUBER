__author__ = 'jasmine-wang'

import os
import numpy as np
from tensorflow.contrib import learn
from pathlib import Path
import time
from data_helpers import *
import pickle

# This file creates a CSV with columns: 
# Context ID, Context (Query), Ground truth reply, Reply1 ('human'), Reply2 ('hred'), Reply3 ('de'), Reply4 ('tfidf'), Score1, Score2, Score3, Score4
# as well as:
# queries.txt
# replies1.txt
# replies2.txt
# replies3.txt
# replies4.txt

# separate results.csv for each query/reply pair of files


def load_adem_data(data_dir):
    with open(data_dir + '/models.pkl', 'rb') as models, \
        open(data_dir + '/clean_data.pkl', 'rb') as scores, \
            open(data_dir + '/contexts.pkl', 'rb') as queries_and_replies, \
                open(data_dir + '/queries.txt', 'w+') as queries, \
                    open(data_dir + '/human_replies.txt', 'w+') as human_replies, \
                        open(data_dir + '/hred_replies.txt', 'w+') as hred_replies, \
                            open(data_dir + '/de_replies.txt', 'w+') as de_replies, \
                                open(data_dir + '/tfidf_replies.txt', 'w+') as tfidf_replies :


                def write_to_file(model_type, line):
                    line = line.encode('utf-8').strip()
                    line = line[line.index(':')+2:] + "\n"
                    if not model_type:
                        queries.write(line)
                    elif "human" in model_type:
                        human_replies.write(line)
                    elif "hred" in model_type:
                        hred_replies.write(line)
                    elif "de" in model_type:
                        de_replies.write(line)
                    elif "tfidf" in model_type:
                        tfidf_replies.write(line)


                replies_to_model_type = pickle.load(models)
                """
                type: dict
                    keys: context_id
                    values: list of length 4. The model type corresponds to which model 
                        generated the response for the corresponding score in 
                        clean_data*.pkl.
                """

                scores = pickle.load(scores)
                """
                type: dict
                    keys: AMT HIT ID
                    values: list of dict
                        keys: score names (e.g. overall1, overall2, overall3, overall4)
                        values: scores by AMT user
                """

                queries_and_replies = pickle.load(queries_and_replies)   
                """
                type: list
		            values: list [context_id, context, r1, r2, r3, r4]
                """
                
                # Step 1:
                # Write queries and replies (list of lists) into queries.txt and replies1-n.txt
                #
                for content in queries_and_replies:
                    context_id = content[0]
                    query = content[1]
                    r1 = content[2]
                    r2 = content[3]
                    r3 = content[4]
                    r4 = content[5]

                    model_order = replies_to_model_type[context_id]

                    write_to_file(model_order[0], r1)
                    write_to_file(model_order[1], r2)
                    write_to_file(model_order[2], r3)
                    write_to_file(model_order[3], r4)
                    write_to_file(None, line=query)
                print("Finished writing queries and replies files")


            
if __name__ == '__main__':
    data_dir = './ADEM_data/data'
    query_max_length, reply_max_length = [20, 30]
# fquery, freply1, freply2, freply3, freply4 = 
    load_adem_data(data_dir)

    fquery = "queries.txt"
    freply1 = "human_replies.txt"
    freply2 = "hred_replies.txt"
    freply3 = "de_replies.txt"
    freply4 = "tfidf_replies.txt"

    # # Path to word2vec weights
    fqword2vec = 'GoogleNews-vectors-negative300.txt'
    frword2vec = 'GoogleNews-vectors-negative300.txt'

    # print("Processing training files")
    process_train_file(data_dir, fquery, query_max_length)
    process_train_file(data_dir, freply1, reply_max_length)
    process_train_file(data_dir, freply2, reply_max_length)
    process_train_file(data_dir, freply3, reply_max_length)
    process_train_file(data_dir, freply4, reply_max_length)

    fqvocab = '%s.vocab%d'%(fquery, query_max_length)
    frvocab1 = '%s.vocab%d'%(freply1, reply_max_length)
    frvocab2 = '%s.vocab%d'%(freply2, reply_max_length)
    frvocab3 = '%s.vocab%d'%(freply3, reply_max_length)
    frvocab4 = '%s.vocab%d'%(freply4, reply_max_length)

    word2vec, vec_dim, _ = load_word2vec(data_dir, fqword2vec)
    make_embedding_matrix(data_dir, fquery, word2vec, vec_dim, fqvocab)

    word2vec, vec_dim, _ = load_word2vec(data_dir, frword2vec)
    make_embedding_matrix(data_dir, freply1, word2vec, vec_dim, frvocab)
    make_embedding_matrix(data_dir, freply2, word2vec, vec_dim, frvocab)
    make_embedding_matrix(data_dir, freply3, word2vec, vec_dim, frvocab)
    make_embedding_matrix(data_dir, freply4, word2vec, vec_dim, frvocab)
    pass
