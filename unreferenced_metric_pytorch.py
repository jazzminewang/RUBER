from __future__ import print_function

import torch
import pandas
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import cPickle
import os
import data_helpers


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
  def __init__(self, input_dim, output_dim, inputs_length, inputs_indices, num_layers=2, num_words=10, num_dim=100, pooling=None):
    super(BiRNNEmbedding, self).__init__()
    # self.embedding = nn.Embedding.from_pretrained
    #TODO: load shit into BIRNN (specific qembed, look into nn.embedding)
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
    
    def forward(self, query, query_length, reply, reply_length):
        qout = self.queryGRU(query, query_length) # B x dim * 2
        rout = self.replyGRU(reply, reply_length) # B x dim * 2
        qTM = torch.tensordot(qout, self.quadratic_M, dims=1) # B x dim * 2
        quadratic = qTM.mul(rout).sum(1).unsqueeze(1) # B x 1
        xin = torch.cat([qout, rout, quadratic],dim=1)
        print(xin.shape)
        out = self.mlp(xin) # B x (dim * 4 + 1)
        # out = B x 1
        return out
    
    
    um = UnreferencedMetric(10,10,None,None,2,2,emb_dim=10, vocab_size=10)