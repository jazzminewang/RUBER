__author__ = 'jasmine-wang'

import os
import numpy as np
from tensorflow.contrib import learn
from pathlib import Path
import time
from data_helpers import process_train_file, load_word2vec, make_embedding_matrix
import pickle
import csv
from itertools import izip


# This file creates a CSV with columns: 
# Context ID, Context (Query), Ground truth reply, Reply1 ('human'), Reply2 ('hred'), Reply3 ('de'), Reply4 ('tfidf'), Score1, Score2, Score3, Score4
# as well as:
# queries.txt
# replies1.txt
# replies2.txt
# replies3.txt
# replies4.txt

# separate results.csv for each query/reply pair of files


def load_adem_data(raw_data_dir, processed_data_dir):
    with open(raw_data_dir + '/models.pkl', 'rb') as models, \
        open(raw_data_dir + '/clean_data.pkl', 'rb') as scores, \
            open(raw_data_dir + '/contexts.pkl', 'rb') as queries_and_replies_pickle, \
                open(processed_data_dir + '/queries.txt', 'w+') as queries, \
                    open(processed_data_dir + '/human_replies.txt', 'w+') as human_replies, \
                        open(processed_data_dir + '/hred_replies.txt', 'w+') as hred_replies, \
                            open(processed_data_dir + '/de_replies.txt', 'w+') as de_replies, \
                                open(processed_data_dir + '/tfidf_replies.txt', 'w+') as tfidf_replies,\
                                    open(processed_data_dir + '/context_ids.txt', 'w+') as context_ids, \
                                        open(processed_data_dir + '/human_scores.txt', 'w+') as human_scores, \
                                            open(processed_data_dir + '/hred_scores.txt', 'w+') as hred_scores, \
                                                open(processed_data_dir + '/de_scores.txt', 'w+') as de_scores, \
                                                    open(processed_data_dir + '/tfidf_scores.txt', 'w+') as tfidf_scores:
                def write_scores_to_file(context_id, model_type, index):
                    score_key = "overall" + str(index)
                    avg_score = []
                    
                    for list_scores in scores_dict.values():
                        for task in list_scores:
                            if task["c_id"] == context_id:
                                model_score = int(task[score_key])
                                avg_score.append(model_score)
                    
                    if avg_score == []:
                        score = -1
                    else:
                        score = sum(avg_score) / float(len(avg_score))

                    score = str(score) +"\n"

                    if "human" in model_type:
                        human_scores.write(score)
                    elif "hred" in model_type:
                        hred_scores.write(score)
                    elif "de" in model_type:
                        de_scores.write(score)
                    elif "tfidf" in model_type:
                        tfidf_scores.write(score)

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

                scores_dict = pickle.load(scores)
                """
                type: dict
                    keys: AMT HIT ID
                    values: list of dict
                        keys: score names (e.g. overall1, overall2, overall3, overall4)
                        values: scores by AMT user
                """

                queries_and_replies = pickle.load(queries_and_replies_pickle)   
                """
                type: list
		            values: list [context_id, context, r1, r2, r3, r4]
                """
                
                # Step 1:
                # Sort queries_and_replies.
                # Write queries and replies (list of lists) into queries.txt and replies1-n.txt
                # 
                queries_and_replies.sort(key=lambda x: x[0])

                for content in queries_and_replies:
                    context_id = content[0]
                    query = content[1]
                    r1 = content[2]
                    r2 = content[3]
                    r3 = content[4]
                    r4 = content[5]

                    # print("Example content: query: {}, r1: {}, r2: {}, r3: {}, r4: {}").format(query, r1, r2, r3, r4)

                    model_order = replies_to_model_type[context_id]

                    print("Context id: " + str(context_id))
                    print("r1 " + r1 + " " + model_order[0])
                    print("r2 " + r2 + " " + model_order[1])
                    print("r3 " + r3+ " " + model_order[2])
                    print("r4 " + r4+ " " + model_order[3])

                    write_to_file(model_order[0], r1)
                    write_to_file(model_order[1], r2)
                    write_to_file(model_order[2], r3)
                    write_to_file(model_order[3], r4)
                    write_to_file(None, line=query)
                    context_ids.write(str(context_id)+"\n")

                    # write all shit to scores - separate txt file for each
                    write_scores_to_file(context_id, model_order[0], 1)
                    write_scores_to_file(context_id, model_order[1], 2)
                    write_scores_to_file(context_id, model_order[2], 3)
                    write_scores_to_file(context_id, model_order[3], 4)
                    
                print("Finished writing queries, replies, and scores files")

    models.close()
    scores.close()
    queries_and_replies_pickle.close()
    queries.close()
    human_replies.close()
    hred_replies.close()
    de_replies.close()
    tfidf_replies.close()
    context_ids.close()
    human_scores.close()
    hred_scores.close()
    de_scores.close()
    tfidf_scores.close()


def write_adem_to_csv(data_dir):
    # Step 2:
    # Create CSV with human scores (can combine later results with pandas)
    """write results to CSV"""
    with  open(data_dir + '/queries.txt', 'r') as queries, \
            open(data_dir + '/human_replies.txt', 'r') as human_replies, \
                open(data_dir + '/hred_replies.txt', 'r') as hred_replies, \
                    open(data_dir + '/de_replies.txt', 'r') as de_replies, \
                        open(data_dir + '/tfidf_replies.txt', 'r') as tfidf_replies, \
                            open(data_dir + '/context_ids.txt', 'r') as context_ids, \
                                open(data_dir + '/true.txt', 'r') as true_replies, \
                                    open(data_dir + '/human_scores.txt', 'r') as human_scores, \
                                        open(data_dir + '/hred_scores.txt', 'r') as hred_scores, \
                                            open(data_dir + '/de_scores.txt', 'r') as de_scores, \
                                                open(data_dir + '/tfidf_scores.txt', 'r') as tfidf_scores:


        with open(data_dir + "/benchmark.csv", "w+") as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            column_titles = ["Context_ID", "Query", "Ground truth reply", "Human", "Human_score", "HRED", "HRED_score", "DE", "DE_score", "TF-IDF", "TF-IDF_score"]
            writer.writerow([col for col in column_titles])

            for context_id, query, true_reply, human, human_score, hred, hred_score, de, de_score, tf_idf, tf_idf_score in \
                izip(context_ids, queries, true_replies, human_replies, human_scores, hred_replies, hred_scores, de_replies, de_scores, tfidf_replies, tfidf_scores):
                    writer.writerow([context_id, query, true_reply, human, human_score, hred, hred_score, de, de_score, tf_idf, tf_idf_score])
        csvfile.close()


            
if __name__ == '__main__':
    raw_data_dir = './data/ADEM/raw'
    processed_data_dir = './data/ADEM/validation'
    word2vec_dir = './data'

    query_max_length, reply_max_length = [20, 30]
    print("Loading ADEM data")
    load_adem_data(raw_data_dir, processed_data_dir)

    print("Writing data to CSV")
    write_adem_to_csv(processed_data_dir)

    fquery = "queries.txt"
    freply1 = "human_replies.txt"
    freply2 = "hred_replies.txt"
    freply3 = "de_replies.txt"
    freply4 = "tfidf_replies.txt"
    freply5 = "true.txt"
    # # Path to word2vec weights
    fqword2vec = 'GoogleNews-vectors-negative300.txt'
    frword2vec = 'GoogleNews-vectors-negative300.txt'

    print("Processing training files")
    process_train_file(processed_data_dir, fquery, query_max_length)
    process_train_file(processed_data_dir, freply1, reply_max_length)
    process_train_file(processed_data_dir, freply2, reply_max_length)
    process_train_file(processed_data_dir, freply3, reply_max_length)
    process_train_file(processed_data_dir, freply4, reply_max_length)
    process_train_file(processed_data_dir, freply5, reply_max_length)

    fqvocab = '%s.vocab%d'%(fquery, query_max_length)
    frvocab1 = '%s.vocab%d'%(freply1, reply_max_length)
    frvocab2 = '%s.vocab%d'%(freply2, reply_max_length)
    frvocab3 = '%s.vocab%d'%(freply3, reply_max_length)
    frvocab4 = '%s.vocab%d'%(freply4, reply_max_length)
    frvocab5 = '%s.vocab%d'%(freply5, reply_max_length)
    word2vec, vec_dim, _ = load_word2vec(word2vec_dir, fqword2vec)
    make_embedding_matrix(processed_data_dir, fquery, word2vec, vec_dim, fqvocab)

    make_embedding_matrix(processed_data_dir, freply1, word2vec, vec_dim, frvocab1)
    make_embedding_matrix(processed_data_dir, freply2, word2vec, vec_dim, frvocab2)
    make_embedding_matrix(processed_data_dir, freply3, word2vec, vec_dim, frvocab3)
    make_embedding_matrix(processed_data_dir, freply4, word2vec, vec_dim, frvocab4)
    make_embedding_matrix(processed_data_dir, freply5, word2vec, vec_dim, frvocab5)
    pass
