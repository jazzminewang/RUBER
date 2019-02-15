import os
import cPickle
import numpy as np
from tensorflow.contrib import learn
from pathlib import Path
import time
from shutil import copyfile
from data_helpers import parse_persona_chat_dataset
import random

if __name__ == '__main__':
    raw_data_dir = "data/validation_raw_personachat"
    processed_data_dir = "data/personachat/validation"
    freply_short, fquery_short = parse_persona_chat_dataset(raw_data_dir, processed_data_dir, "valid")
    freply = os.path.join(processed_data_dir, freply_short)
    fquery = os.path.join(processed_data_dir, fquery_short)

    copyfile(freply, freply + ".true.sub")
    reply_true = open(freply).readlines()
    random.shuffle(reply_true)
    open(freply + ".sub", "w+").writelines(reply_true)
    copyfile(fquery, fquery + ".sub")
