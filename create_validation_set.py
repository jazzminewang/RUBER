import os
import cPickle
import numpy as np
from tensorflow.contrib import learn
from pathlib import Path
import time
from shutil import copyfile
import random

def create_validation_set():
    print("Creating validation set")
    directory = os.path.join('data/', 'validation_original_personachat')
    fquery_filename = os.path.join('data/', 'personachat', 'validation', "queries.txt")
    freply = os.path.join('data/', 'personachat', 'validation', "replies.txt")

    for data_filename in os.listdir(directory):
        data_filename = os.path.join(directory, data_filename)

        if "no_cand" in data_filename and "revised" in data_filename:
            with open(data_filename, "r") as datafile, open(fquery_filename, "w+") as queries, open(freply, "w+") as replies:
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
	    copyfile(freply, freply + ".true.sub")
	    
	    reply_true = open(freply).readlines()
	    random.shuffle(reply_true)
            open(freply + ".sub", "w+").writelines(reply_true)

	    copyfile(fquery_filename, fquery_filename + ".sub")
if __name__ == '__main__':
    create_validation_set()
