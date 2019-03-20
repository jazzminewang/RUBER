from __future__ import print_function

import torch
import pandas
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import cPickle
import os
import data_helpers
import gensim

def get_mlp(input_dim, output_dim, num_layers=2, dropout=0.0):
    network_list = []
    assert num_layers > 0
    for _ in range(num_layers-1):
        network_list.append(nn.Linear(input_dim, input_dim))
        network_list.append(nn.ReLU())
        network_list.append(nn.BatchNorm1d(num_features=input_dim))
        network_list.append(nn.Dropout(dropout))
    network_list.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(
        *network_list
    )

class BiRNNEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, num_words=10, num_dim=100, pooling=None):
        super(BiRNNEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_words, num_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.rnn = nn.GRU(input_dim, output_dim,num_layers=num_layers, bidirectional=True, batch_first=True)
        self.pooling = None
        
    def forward(self, data, inp_len):
        data = self.embedding(data)
        data_pack = pack_padded_sequence(data, inp_len, batch_first=True)
        outp, hidden_rep = self.rnn(data_pack)
        outp, _ = pad_packed_sequence(outp, batch_first=True)
        outp = outp.contiguous()
        # pooling
        if self.pooling:
        raise NotImplementedError("pooling yet not implemented")
        else:
        outp = outp[:,-1,:]
        
        return outp

class UnreferencedMetric(nn.Module):
    def __init__(self, qmax_length,
                rmax_length,
                fqembed,
                frembed,
                gru_num_units,
                mlp_units,
                init_learning_rate=0.001,
                l2_regular=0.1,
                emb_dim=128,
                vocab_size=10,
                margin=0.5, 
                train_dir='train_data_batch_norm_128',
                is_training=True,
                batch_norm=False, 
                train_dataset='',
                log_dir="tmp/",
                scramble=False,
                additional_negative_samples='',
                ):
        """
            Initialize related variables and construct the neural network graph.
            Args:
                qmax_length, rmax_length: max sequence length for query and reply
                fqembed, frembed: embedding matrix file for query and reply
                gru_num_units: number of units in each GRU cell
                mlp_units: number of units for mlp, a list of length T,
                    indicating the output units for each perceptron layer.
                No need to specify the output layer size 1.
    """
        super(UnreferencedMetric, self).__init__()
        print("Log dir is ")
        print(log_dir)
        if batch_norm:
            if scramble:
                self.train_dir = os.path.join(log_dir, train_dataset + "_" + str(gru_num_units) + "_" + str(init_learning_rate) + "_" + str(margin) + "_batchnorm" + "_scramble" + additional_negative_samples.split("/")[0])
            else:  
                self.train_dir = os.path.join(log_dir, train_dataset + "_" + str(gru_num_units) + "_" + str(init_learning_rate) + "_" + str(margin) + "_batchnorm" + additional_negative_samples.split("/")[0])
        else:
            if scramble:
                self.train_dir = os.path.join(log_dir, train_dataset + "_" + str(gru_num_units) + "_" + str(init_learning_rate) + "_" + str(margin)+ "_scramble"  + additional_negative_samples.split("/")[0])
            else:
                self.train_dir = os.path.join(log_dir, train_dataset + "_" + str(gru_num_units) + "_" + str(init_learning_rate) + "_" + str(margin) + additional_negative_samples.split("/")[0])
        
        self.additional_negative_samples = additional_negative_samples
        
        print('Loading embedding matrix')
        qembed = cPickle.load(open(fqembed, 'rb'))
        rembed = cPickle.load(open(frembed, 'rb'))

        self.qmax_length = qmax_length
        self.rmax_length = rmax_length

        self.queryGRU = BiRNNEmbedding(emb_dim,emb_dim,num_words=vocab_size, num_dim=emb_dim)
        self.replyGRU = BiRNNEmbedding(emb_dim,emb_dim,num_words=10, num_dim=emb_dim)
        self.quadratic_M = nn.Parameter(torch.zeros((emb_dim * 2,emb_dim * 2)))

        self.mlp = get_mlp((emb_dim * 4 + 1),1,2)
    
    def forward(self, query_batch, query_length, reply_batch, reply_length):
        qout = self.queryGRU(query_batch, query_length) # B x dim * 2
        rout = self.replyGRU(reply_batch, reply_length) # B x dim * 2

        qTM = torch.tensordot(qout, self.quadratic_M, dims=1) # B x dim * 2
        quadratic = qTM.mul(rout).sum(1).unsqueeze(1) # B x 1
        xin = torch.cat([qout, rout, quadratic],dim=1)
        print(xin.shape)
        out = self.mlp(xin) # B x (dim * 4 + 1)
        # out = B x 1
        return out

    def get_batch(self, data, data_size, batch_size, idx=None):
        """
        Get a random batch with size batch_size

        Args:
            data: [[length, [ids]], each with a line of segmented sentence
            data_size: size of data
            batch_size: returned batch size
            idx: [batch_size], randomly get batch if idx None, or get with idx

        Return:
            batched vectors [batch_size, max_length]
            sequence length [batch_size]
            idx [batch_size]
        """
        if not idx:
            idx=[random.randint(0, data_size-1) for _ in range(batch_size)]
        ids = [data[i][1] for i in idx]
        lens = [data[i][0] for i in idx]
        return ids, lens, idx

    #TODO: how to train w batches? do I need to make a custom data loader?

    def train(self, args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        
        # Save model after specified # epochs
        torch.save(model.state_dict(), PATH)
    
    def scores(self, data_dir, fquery, freply, fqvocab, frvocab, checkpoint_dir, init=False):
        # to do - how to init model?
        if not model:
           # model = UnreferencedMetric(??)
           model.load_state_dict(torch.load(PATH))
        model.eval()
        queries = data_helpers.load_file(data_dir, fquery)
        replies = data_helpers.load_file(data_dir, freply)
	    data_size = len(queries)

        qvocab = data_helpers.load_vocab(data_dir, fqvocab)
        rvocab = data_helpers.load_vocab(data_dir, frvocab)
        scores=[]

        with torch.no_grad():
            for query, reply in zip(queries, replies):
                ql, qids = data_helpers.transform_to_id(qvocab, query,
                        self.qmax_length)
                rl, rids = data_helpers.transform_to_id(rvocab, reply,
                        self.rmax_length)
                
                #TODO: what is the logic here to run inference?

                output = model(data) # how to pass in this data?
                scores.append(score[0])




    # TODO: add this to hybrid
    # um = UnreferencedMetric(10,10,None,None,2,2,emb_dim=10, vocab_size=10)