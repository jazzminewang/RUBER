__author__ = 'liming-vie'

import os
import cPickle
import numpy as np
from tensorflow.contrib import learn
from pathlib import Path
import time
import argparse
import random

def tokenizer(iterator):
    for value in iterator:
        yield value.split()

def load_file(data_dir, fname):
    fname = os.path.join(data_dir, fname)
    print 'Loading file %s'%(fname)
    lines = open(fname).readlines()
    print("Num lines in file: ")
    print(len(lines))
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

def parse_twitter_dataset(raw_data_dir, processed_data_dir, filename="train.txt"):
    # end-of-utterance: </s>
    # end-of-dialogue: </d>
    # first speaker: <first_speaker>
    # second speaker: <second_speaker>
    # third speaker: <third_speaker>
    # minor speaker: <minor_speaker>
    # voice over: <voice_over>
    # off screen: <off_screen>
    # pause: <pause>
    
    # one dialogue per line

    fquery_filename = os.path.join(processed_data_dir, "queries.txt")
    freply_filename = os.path.join(processed_data_dir, "replies.txt")
    fquery_file = Path(fquery_filename)
    freply_file = Path(freply_filename)
    fquery_short = "queries.txt" 
    freply_short = "replies.txt"
    if not fquery_file.exists() and not freply_file.exists():
        print("Creating queries and replies dataset from twitter dataset")
        
	for data_filename in os.listdir(raw_data_dir):
		if data_filename == filename:
                    data_filename = os.path.join(raw_data_dir, data_filename)
		else:
		    continue
                with open(data_filename, "r") as datafile, \
                    open(fquery_filename, "w+") as queries, \
                        open(freply_filename, "w+") as replies:

                    lines = datafile.readlines()

                    for line in lines:
			filter_set = ("/s>", "</d>", " /d>", "/d>", "/d> <")
                        dialogue = filter(None, line.split("</s>"))
			dialogue = [x.strip() for x in dialogue]
			dialogue = filter(lambda x: (x not in filter_set), dialogue)
            		context = []
                        for i in range(0, len(dialogue) - 2):
                            more = get_most_recent_context(context)
                            query = dialogue[i].strip().lstrip("<first_speaker>").lstrip("<second_speaker>").strip().lstrip("<at>")
                            context.append(query)
                            query = query + more
                            reply = dialogue[i + 1].strip().lstrip("<first_speaker>").lstrip("<second_speaker>").strip().lstrip("<at>")
                            queries.write(query + "\n")
                            replies.write(reply + "\n")
    print("Wrote queries to " + fquery_filename)
    print("Wrote replies to " + freply_filename)
    return fquery_short, freply_short
      
def get_most_recent_context(context):
    context = [item.strip('\n').strip('\t').replace('\n','').replace('\t','') for item in context]

    if len(context) >= 3:
        return "{}{}{}".format(context[len(context) - 1], context[len(context) - 2], context[len(context) - 3])
    elif len(context) == 2:
        return "{}{}".format(context[len(context) - 1], context[len(context) - 2])
    elif len(context) == 1:
        return "{}".format(context[len(context) - 1])
    else:
        return ""

def parse_persona_chat_dataset(raw_data_dir, processed_data_dir, file_type="train"):
    """
    Return:
        file path to file with all queries
        file path to file with all replies
    """
    fquery_filename = os.path.join(processed_data_dir, "queries.txt")
    freply_filename = os.path.join(processed_data_dir, "replies.txt")
    fquery_file = Path(fquery_filename)
    freply_file = Path(freply_filename)
    fquery_short = "queries.txt" 
    freply_short = "replies.txt"
    print("Creating queries and replies dataset from personachat with context added (past three queries)")
        
    for data_filename in os.listdir(raw_data_dir):
        if file_type in data_filename and "no_cand" in data_filename and "none" in data_filename:
            data_filename = os.path.join(raw_data_dir, data_filename)
            print("parsing" + data_filename)
            with open(data_filename, "r") as datafile, \
                open(fquery_filename, "w+") as queries, \
                    open(freply_filename, "w+") as replies:
                
                lines = datafile.readlines()
                filtered_lines = [line for line in lines if 'persona:' not in line]
                new_conversation = True
                context = []

                # change to new conversation if next line is a smaller number --> omitting last line edge case
                for x in range(0, len(filtered_lines) - 2):
                    original = filtered_lines[x]
                    number = int(original[:2])
                    split = (original[2:].rstrip("\n")).split("\t")
                    print(split)
                    # time.sleep(1)
                    if len(split) > 2:
                        print(len(split))

                    next_number = int(filtered_lines[x + 1][:2])
                    
                    if new_conversation:
                        # add first part only as query
                        if "__SILENCE__" not in split[0]:
                            queries.write(split[0] + get_most_recent_context(context) + "\n")
                            context.append(split[0])
                            if len(split) > 1:
                                replies.write(split[1]+ "\n")
                                queries.write(split[1] + get_most_recent_context(context)+ "\n")
                                context.append(split[1])
                        else:
                            queries.write(split[1] + get_most_recent_context(context) + "\n")
                            context.append(split[1])
                        new_conversation = False
                    elif next_number < number:
                        # next line is new conversation, so add last part as only a reply. Also, clear context.
                        new_conversation = True
                        if len(split) > 1:
                            replies.write(split[0]+ "\n")
                            queries.write(split[0] + get_most_recent_context(context)+ "\n")
                            replies.write(split[1] + "\n")
                        else:
                            replies.write(split[0])
                        context = []
                    else:
                        #write both parts as queries and replies
                        if len(split) > 1:
                            replies.write(split[0]+ "\n")
                            queries.write(split[0]+ get_most_recent_context(context)+ "\n")
                            context.append(split[0])
                            replies.write(split[1]+ "\n")
                            queries.write(split[1] + get_most_recent_context(context)+ "\n")
                            context.append(split[1])
                            # time.sleep(3)
                        else:
                            queries.write(split[0]+ get_most_recent_context(context)+ "\n")
                            replies.write(split[0]+ "\n")
                            context.append(split[0])
                
                replies.write(filtered_lines[len(filtered_lines) - 1])
            datafile.close()
            print(fquery_filename)
            print(freply_filename)
            queries.close()
            replies.close()
    return fquery_short, freply_short

    def randomize(lines, proportion):
        new_lines = []
        for i, line in enumerate(lines):
            new_line = []
            for word in line.split():
                if random.randint(0, proportion) == 1:
                    rand_line = lines[random.randint(0, len(lines))-1].split()
                    if rand_line:
                                rand_word = rand_line[random.randint(0, len(rand_line)-1)]
                                word = rand_word
        new_line.append(word)
        string_line =" ".join(str(x) for x in new_line)
        new_lines.append(string_line) 
    return new_lines
    

def scramble(raw_data_dir, processed_data_dir):
    fquery_filename = os.path.join(processed_data_dir, "queries.txt")
    freply_filename = os.path.join(processed_data_dir, "replies.txt")
    fquery_file = Path(fquery_filename)
    freply_file = Path(freply_filename)
    fquery_short = "queries.txt" 
    freply_short = "replies.txt"

    with open(fquery_filename, "r") as fquery, \
        open(freply_filename, "r") as freply:
            fquery_lines = fquery.readlines()
            freply_lines = freply.readlines()
    print("Randomizing query file " + fquery_filename)
    new_fquery = randomize(fquery_lines, 10)
    print("Randomizing reply file " + freply_filename)
    new_freply = randomize(freply_lines, 10)

    scrambled_dir = os.path.join(raw_data_dir, "scramble_train")
    fquery_filename = os.path.join(scrambled_dir, "queries.txt")
    freply_filename = os.path.join(scrambled_dir, "replies.txt")

    with open(fquery_filename, "w+") as fquery, \
        open(freply_filename, "w+") as freply:
            for line in new_fquery:
                fquery.write(line + "\n")
            for line in new_freply:
                freply.write(line + "\n")

    print("Finished scrambling text - example sentence changes")
    print(fquery_lines[:5])
    print(new_fquery[:5])

    return fquery_short, freply_short
    
if __name__ == '__main__':
    # Argument is dataset. Twitter or Personachat 
    query_max_length, reply_max_length = [20, 30]

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help="Either Personachat or Twitter")
    parser.add_argument('-scramble', action='store_true', help="If true, 1/10 of the words will be randomly switched")

    args = parser.parse_args()
    if args.dataset=="twitter":
	    raw_data_dir = "./data/twitter"
            processed_train_dir = "./data/twitter/train/"
            processed_validation_dir = "./data/twitter/validation"

            print("Parsing twitter training data")
            fquery_train, freply_train = parse_twitter_dataset(raw_data_dir, processed_train_dir)

            if args.scramble:
                print("Scrambling twitter dataset")
                fquery_train, freply_train = scramble(raw_data_dir, processed_train_dir)
                processed_train_dir = os.path.join("data", "twitter", "scramble_train")

            print("Parsing twitter validation data")
            fquery_validate, freply_validate = parse_twitter_dataset(raw_data_dir, processed_validation_dir, filename="valid.txt")
    else:
        raw_data_dir = './data/personachat'
        processed_train_dir = "./data/personachat/train/"
        processed_validation_dir = "./data/personachat/validation"
        print("Parsing personachat training data")
        fquery_train, freply_train = parse_persona_chat_dataset(raw_data_dir, processed_train_dir)
        fgenerated_train = os.path.join("generated_responses", "personachat_train_responses.txt")

        print("Parsing personachat validation data")
        fquery_validate, freply_validate = parse_persona_chat_dataset(raw_data_dir, processed_validation_dir, file_type="valid")
    
        if args.scramble:
            print("Scrambling personachat dataset")
            fquery_train, freply_train = scramble(raw_data_dir, processed_train_dir)
	    processed_train_dir = os.path.join("data", "personachat", "scramble_train")

    # Path to word2vec weights
    fqword2vec = 'GoogleNews-vectors-negative300.txt'
    frword2vec = 'GoogleNews-vectors-negative300.txt'

    raw_data_dir = "./data"

    word2vec, vec_dim, _ = load_word2vec(raw_data_dir, fqword2vec)
    print("Generated data - negative sampling")
    processed_generated_dir = os.path.join(raw_data_dir, "generated_responses")
    freply_generated = "personachat_train_responses.txt"
    process_train_file(processed_generated_dir, freply_generated, reply_max_length)

    fgvocab = '%s.vocab%d'%(freply_generated, reply_max_length)

    make_embedding_matrix(processed_generated_dir, freply_generated, word2vec, vec_dim, fgvocab)

    #make sure embed and vocab file paths are correct
    process_train_file(processed_train_dir, fquery_train, query_max_length)
    process_train_file(processed_train_dir, freply_train, reply_max_length)
    

    fqvocab = '%s.vocab%d'%(fquery_train, query_max_length)
    frvocab = '%s.vocab%d'%(freply_train, reply_max_length)


    make_embedding_matrix(processed_train_dir, fquery_train, word2vec, vec_dim, fqvocab)

    make_embedding_matrix(processed_train_dir, freply_train, word2vec, vec_dim, frvocab)

    print("Validation data")
    process_train_file(processed_validation_dir, fquery_validate, query_max_length)
    process_train_file(processed_validation_dir, freply_validate, reply_max_length)

    fqvocab = '%s.vocab%d'%(fquery_validate, query_max_length)
    frvocab = '%s.vocab%d'%(freply_validate, reply_max_length)

    make_embedding_matrix(processed_validation_dir, fquery_validate, word2vec, vec_dim, fqvocab)
    make_embedding_matrix(processed_validation_dir, freply_validate, word2vec, vec_dim, frvocab)
	

    pass
