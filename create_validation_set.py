import os
import cPickle
import numpy as np
from tensorflow.contrib import learn
from pathlib import Path
import time

def create_validation_set():
    print("Creating validation set")
    directory = os.path.join('./data/', 'personachat', 'validation')
    fquery_filename = os.path.join('./data/', 'personachat', 'validation', "queries.txt")
    freply = os.path.join('./data/', 'personachat', 'validation', "replies.txt")

    for data_filename in os.listdir(directory):
        data_filename = os.path.join(directory, data_filename)

        if "no_cand" in data_filename and "revised" in data_filename:
            with open(data_filename, "r") as datafile, open(fquery_filename, "w+") as queries, open(freply, "w+") as replies:
                new_conversation = True
                queries_line_count = 0
                replies_line_count = 0

                while True: 
                    l1 = datafile.readline()
                    l2 = datafile.readline()

                    if (queries_line_count - replies_line_count) > 1:
                        difference = queries_line_count - replies_line_count
                        print(str(difference) + " more query lines than reply" )

                    if l1 == "" or l2 == "":
                        if l1 != "" and l2 == "":
                            replies.write(l1) 
                        break

                    if "persona:" in l1 and "persona:" in l2:
                        continue
                    else:
                        # cases:
                        # 1. persona description, query
                        # 2. persona description, persona description (above)
                        # 3. reply, query. reply.
                            # a. if next sentence if a persona, then finish.
                            # b. if not, reply #2 is also a query.
                        # 4. reply, persona description.

                        if "persona:" in l1 and "persona:" not in l2:
                            #l2 is starting a new conversation! l2 is query only.
                            print("l2 is query")
                            new_conversation = False
                            queries.write(l2) 
                            queries_line_count += 1
                        elif "persona:" not in l1 and "persona:" not in l2 and new_conversation:
                            #l1 is starting a new conversation! 
                            print("l1 is query, l2 is response + query")
                            queries.write(l1)
                            replies.write(l2)

                            queries.write(l2)
                            queries_line_count += 2
                            replies_line_count += 1
                            new_conversation = False
                        elif "persona:" not in l1 and "persona:" not in l2:
                            #part of an old conversation, but check if it terminates
                            print("l1 is response + query, l2 is response")
                            replies.write(l1)
                            queries.write(l1)
                            replies.write(l2)
                            queries_line_count += 1
                            replies_line_count += 2

                            position = datafile.tell()
                            check_line = datafile.readline()
                            # if the next line doesn't start a new conversation, store l2 as query also
                            if "persona:" not in check_line:
                                print("l2 is also query")
                                queries.write(l2)
                                queries_line_count +=1
                            else:
                                print("l2 is last sentence in conversation")
                                new_conversation = True
                            datafile.seek(position)
                        elif "persona:" not in l1 and "persona:" in l2:
                            print("l1 is last sentence in conversation")
                            replies.write(l1)
                            replies_line_count +=1
                            new_conversation = True

if __name__ == '__main__':
    create_validation_set()
