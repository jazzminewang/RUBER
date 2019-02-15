# RUBER
Implementation of [RUBER: An Unsupervised Method for Automatic Evaluation of Open-Domain Dialog Systems](https://arxiv.org/abs/1701.03079)

Using pre-trained word2vec embeddings to initialize bidirectional RNN. 

Steps to run:

1. Download dataset of choice and word2vec file of choice (convert to .txt). In my case, I cloned ParlAI and downloaded the dataset with command, and the word2vec bin file [here](https://github.com/eyaler/word2vec-slim). 
```
python examples/display_data.py --task convai2 --datatype train
```

2. Run data_helpers.py to create queries.txt, replies.txt, vocab and embedding files for each. 
```
python data_helpers.py
```

3. Run hybrid_evaluation.py to train model for unreferenced metric. You'll need to comment out the code block after the "Getting scores" print statement in hybrid_evaluation.py.
```
python hybrid_evaluation.py
```

4. Create files with sentences to score, with format some_string_queries.txt.sub, some_string_replies.txt.sub, some_string_replies.txt.true.sub. Run hybrid_evaluation.py to score these metrics. You'll need to comment out the code block after the "train" print statement in hybrid_evaluation.py. 
```
python hybrid_evaluation.py
```

To create your synthesized replies, either use a dialogue generation model or scramble your replies.txt.true like so: 
```
import random
lines = open('replies.txt').readlines()
random.shuffle(lines)
open('replies_scrambled.txt', 'w').writelines(lines)
```

5. 

