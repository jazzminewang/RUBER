__author__ = 'liming-vie'

import os
import cPickle
import numpy as np
from tensorflow.contrib import learn
from pathlib import Path
import time

def tokenizer(iterator):
    for value in iterator:
        yield value.split()

def load_file(data_dir, fname):
    
   
  
 
    fname = os.path.join(data_dir, fname)
    print 'Loading file %s'%(fname)
    lines = open(fname).readlines()
    return [line.rstrip() for line in lines]

def process_train_file(data_dir, fname, max_length, min_frequency=10):
    """
    Make vocabulary and transform into id files

    Return:
        vocab_file_name
        vocab_dict: map vocab to id
        vocab_size
    """
    fvocab = '%s.vocab%d'%(fname, max_length)
    foutput = os.path.join(data_dir, fvocab)
    if os.path.exists(foutput):
        print 'Loading vocab from file %s'%foutput
        vocab = load_vocab(data_dir, fvocab)
        return fvocab, vocab, len(vocab)

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_length,
            tokenizer_fn = tokenizer, min_frequency=min_frequency)
    x_text = load_file(data_dir, fname)
    print 'Vocabulary transforming'
    # will pad 0 for length < max_length
    ids = list(vocab_processor.fit_transform(x_text))
    print "Vocabulary size %d"%len(vocab_processor.vocabulary_)
    fid = os.path.join(data_dir, fname+'.id%d'%max_length)
    print 'Saving %s ids file in %s'%(fname, fid)
    cPickle.dump(ids, open(fid, 'wb'), protocol=2)

    print 'Saving vocab file in %s'%foutput
    size = len(vocab_processor.vocabulary_)
    vocab_str = [vocab_processor.vocabulary_.reverse(i) for i in range(size)]
    with open(foutput, 'w') as fout:
        fout.write('\n'.join(vocab_str))

    vocab = load_vocab(data_dir, fvocab)
    return fvocab, vocab, len(vocab)

def load_data(data_dir, fname, max_length):
    """
    Read id file data

    Return:
        data list: [[length of vector (represents 1 sentence), [token_ids in that sentence]]]
    """
    fname = os.path.join(data_dir, "%s.id%d"%(fname, max_length))
    print 'Loading data from %s'%fname
    ids = cPickle.load(open(fname, 'rb'))
    data=[]
    for vec in ids:
        length = len(vec)
        if vec[-1] == 0:
            length = list(vec).index(0)
        data.append([length, vec])
    return data

def load_vocab(data_dir, fvocab):
    """
    Load vocab
    """
    fvocab = os.path.join(data_dir, fvocab)
    print 'Loading vocab from %s'%fvocab
    vocab={}
    with open(fvocab) as fin:
        for i, s in enumerate(fin):
            vocab[s.rstrip()] = i
    return vocab

def transform_to_id(vocab, sentence, max_length):
    """
    Transform a sentence into id vector using vocab dict
    Return:
        length, ids
    """
    words = sentence.split()
    ret = [vocab.get(word, 0) for word in words]
    l = len(ret)
    l = max_length if l > max_length else l
    if l < max_length:
        ret.extend([0 for _ in range(max_length - l)])
    return l, ret[:max_length]

def make_embedding_matrix(data_dir, fname, word2vec, vec_dim, fvocab):
    foutput = os.path.join(data_dir, '%s.embed'%fname)
    if os.path.exists(foutput):
        print 'Loading embedding matrix from %s'%foutput
        return cPickle.load(open(foutput, 'rb'))

    vocab_str = load_file(data_dir, fvocab)
    print 'Saving embedding matrix in %s'%foutput
    matrix=[]

    for vocab in vocab_str:
        vec = word2vec[vocab] if vocab in word2vec \
                else [0.0 for _ in range(vec_dim)]
        matrix.append(vec)
    cPickle.dump(matrix, open(foutput, 'wb'), protocol=2)
    return matrix

def load_word2vec(data_dir, fword2vec):
    """
    Return:
        word2vec dict
        vector dimension
        dict size
    """
    fword2vec = os.path.join(data_dir, fword2vec)
    print 'Loading word2vec dict from %s'%fword2vec
    vecs = {}
    vec_dim=0
    with open(fword2vec) as fin:
        size, vec_dim = map(int, fin.readline().split())
        for line in fin:
            ps = line.rstrip().split()
            # print(ps[1:])
            vecs[ps[0]] = map(float, ps[1:])
    return vecs, vec_dim, size

def parse_persona_chat_dataset(data_dir, persona_chat_dir):
    """
    Return:
        file path to file with all queries
        file path to file with all replies
    """
    print("Entering parse")
    fquery_filename = os.path.join(data_dir, persona_chat_dir, "queries.txt")
    freply_filename = os.path.join(data_dir, persona_chat_dir, "replies.txt")
    fquery_file = Path(fquery_filename)
    freply_file = Path(freply_filename)

    fquery_filename_short = os.path.join(persona_chat_dir, "queries.txt")
    freply_filename_short = os.path.join(persona_chat_dir, "replies.txt")

    if not fquery_file.exists() and not freply_file.exists():
        print("Creating queries and replies dataset")
        directory = os.path.join(data_dir, "personachat")
        
	for data_filename in os.listdir(directory):
            if "train" in data_filename and "no_cand" in data_filename and "revised" in data_filename:
                data_filename = os.path.join(data_dir, "personachat", data_filename)
                # parse files with 
                with open(data_filename, "r") as datafile, \
                    open(fquery_filename, "w+") as queries, \
                        open(freply_filename, "w+") as replies:

                    lines = datafile.readlines()
                    filtered_lines = [line for line in lines if 'persona:' not in line]
                    print("Example filtered line")
                    print(filtered_lines[0])

                    new_conversation = True

                    # change to new conversation if next line is a smaller number --> omitting last line edge case
                    for x in range(0, len(filtered_lines) - 2):
                        original = filtered_lines[x]
                        number = int(original[:2])
                        split = (original[2:]).split("\t")
                        next_number = int(filtered_lines[x + 1][:2])
                        
                        if new_conversation:
                            # add first part only as query
                            print("new conversation")
                            queries.write(split[0] + "\n")
                            print("wrote " + split[0] + " to queries")
                            if len(split) > 1:
                                replies.write(split[1])
                                queries.write(split[1])
                                print("wrote " + split[1] + " to queries and replies")
                            new_conversation = False
                        elif next_number < number:
                            print("ending conversation")
                            # next line is new conversation, so add last part as only a reply
                            new_conversation = True
                            if len(split) > 1:
                                replies.write(split[0])
                                queries.write(split[0])
                                replies.write(split[1])
                            else:
                                replies.write(split[0])
                        else:
                            print("continuing conversation")
                            #write both parts as queries and replies
                            if len(split) > 1:
                                replies.write(split[0])
                                queries.write(split[0])
                                replies.write(split[1])
                                queries.write(split[1])
                            else:
                                queries.write(split[0])
                                replies.write(split[0])
                    
                    replies.write(filtered_lines[len(filtered_lines) - 1])
                datafile.close()
                queries.close()
                replies.close()
    return fquery_filename_short, freply_filename_short

# Run this first to create the embedding matrix ? 
if __name__ == '__main__':
    data_dir = './data'
    query_max_length, reply_max_length = [20, 30]

    """
    PERSONA CHAT
    modes: create training dataset
    create embedding files for validation dataset
    """
    # fquery, freply = parse_persona_chat_dataset(data_dir, "personachat/better_turns/")
    fquery = "personachat/validation_personachat/queries.txt"
    freply = "personachat/validation_personachat/replies.txt"

    # Better turns
    # fquery = "personachat/better_turns/queries.txt"
    # freply = "personachat/better_turns/replies.txt"

    # Path to word2vec weights
    fqword2vec = 'GoogleNews-vectors-negative300.txt'
    frword2vec = 'GoogleNews-vectors-negative300.txt'
    print("Processing training files")
    process_train_file(data_dir, fquery, query_max_length)
    process_train_file(data_dir, freply, reply_max_length)

    fqvocab = '%s.vocab%d'%(fquery, query_max_length)
    frvocab = '%s.vocab%d'%(freply, reply_max_length)

    word2vec, vec_dim, _ = load_word2vec(data_dir, fqword2vec)
    make_embedding_matrix(data_dir, fquery, word2vec, vec_dim, fqvocab)

    word2vec, vec_dim, _ = load_word2vec(data_dir, frword2vec)
    make_embedding_matrix(data_dir, freply, word2vec, vec_dim, frvocab)
    pass
