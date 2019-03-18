
from __future__ import print_function

import os
import random
import cPickle
import tensorflow as tf
import data_helpers


class Unreferenced():
    """Unreferenced Metric
    Measure the relatedness between the generated reply and its query using
    neural network
    """
    def __init__(self,
            qmax_length,
            rmax_length,
            fqembed,
            frembed,
            gru_num_units,
            mlp_units,
            init_learning_rate=1e-4,
            l2_regular=0.1,
            margin=0.5,
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
        # initialize varialbes
	print("Log dir is {}, Learning rate {}".format(log_dir, init_learning_rate))
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
        self.qmax_length = qmax_length
        self.rmax_length = rmax_length
        random.seed()

        print('Loading embedding matrix')
        qembed = cPickle.load(open(fqembed, 'rb'))
        rembed = cPickle.load(open(frembed, 'rb'))

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        """graph"""

        with self.session.as_default():
            # build bidirectional gru rnn and get final state as embedding
            def get_birnn_embedding(sizes, inputs, embed, scope):
                embedding = tf.Variable(embed, dtype=tf.float32,
                                        name="embedding_matrix")
                with tf.variable_scope('forward'):
                    fw_cell = tf.contrib.rnn.GRUCell(gru_num_units)
                with tf.variable_scope('backward'):
                    bw_cell = tf.contrib.rnn.GRUCell(gru_num_units)
                inputs = tf.nn.embedding_lookup(embedding, inputs)
                # outputs, state_fw, state_bw
                _, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(
                    fw_cell, bw_cell,
                    # make inputs as [max_length, batch_size=1, vec_dim]
                    tf.unstack(tf.transpose(inputs, perm=[1, 0, 2])),
                    sequence_length=sizes,
                    dtype=tf.float32,
                    scope=scope)
                # [batch_size, gru_num_units * 2]
                return tf.concat([state_fw, state_bw], 1)

            # query GRU bidirectional RNN
            with tf.variable_scope('query_bidirectional_rnn'):
                self.query_sizes = tf.placeholder(tf.int32,
                                                  # batch_size
                                                  shape=[None], name="query_sizes")
                self.query_inputs = tf.placeholder(tf.int32,
                                                   # [batch_size, sequence_length]
                                                   shape=[None, self.qmax_length],
                                                   name="query_inputs")
                with tf.device('/gpu:1'):
                    query_embedding = get_birnn_embedding(
                        self.query_sizes, self.query_inputs,
                        qembed, 'query_rgu_birnn')

            # reply GRU bidirectional RNN
            with tf.variable_scope('reply_bidirectional_rnn'):
                self.reply_sizes = tf.placeholder(tf.int32,
                                                  shape=[None], name="reply_sizes")
                self.reply_inputs = tf.placeholder(tf.int32,
                                                   shape=[None, self.rmax_length],
                                                   name="reply_inputs")
                with tf.device('/gpu:1'):
                    reply_embedding = get_birnn_embedding(
                        self.reply_sizes, self.reply_inputs,
                        rembed, 'reply_gru_birnn')

            # quadratic feature as qT*M*r
            with tf.variable_scope('quadratic_feature'):
                matrix_size = gru_num_units * 2
                M = tf.get_variable('quadratic_M',
                                    shape=[matrix_size, matrix_size],
                                    initializer=tf.zeros_initializer())
                # [batch_size, matrix_size]
                qTM = tf.tensordot(query_embedding, M, 1)
                quadratic = tf.reduce_sum(qTM * reply_embedding,
                                          axis=1, keep_dims=True)

            # multi-layer perceptron
            with tf.variable_scope('multi_layer_perceptron'):
                # input layer
                mlp_input = tf.concat(
                    [query_embedding, reply_embedding, quadratic], 1)
                mlp_input = tf.reshape(mlp_input, [-1, gru_num_units * 4 + 1])
                # hidden layers
                inputs = mlp_input
                for i in range(len(mlp_units)):
                    with tf.variable_scope('mlp_layer_%d' % i):
                        inputs = tf.contrib.layers.legacy_fully_connected(
                            inputs, mlp_units[i],
                            activation_fn=tf.tanh,
                            weight_regularizer=tf.contrib.layers. \
                                l2_regularizer(l2_regular))
                        if batch_norm:
                            inputs = tf.contrib.layers.batch_norm(
                                inputs,
                                center=True, scale=True,
                                is_training=is_training)
                self.test = inputs
                # dropout layer
                self.training = tf.placeholder(tf.bool, name='training')
                inputs_dropout = tf.layers.dropout(inputs, training=self.training)
                # output layer
                self.score = tf.contrib.layers.legacy_fully_connected(
                    inputs_dropout, 1, activation_fn=tf.sigmoid,
                    weight_regularizer=tf.contrib.layers.l2_regularizer(l2_regular))
                self.score = tf.reshape(self.score, [-1])  # [batch_size]

        with tf.variable_scope('train'):
            if not is_training:
                self.pos_score = self.score  # for inference, only need the positive score (no negative sampling needed)
            # calculate losses
            else:
                self.pos_score, self.neg_score = tf.split(self.score, 2)
                losses = margin - self.pos_score + self.neg_score
                # make loss >= 0
                losses = tf.clip_by_value(losses, 0.0, 100.0)
                self.loss = tf.reduce_mean(losses)  # self.loss = tensor
                # optimizer
                self.learning_rate = tf.Variable(init_learning_rate, trainable=False, name="learning_rate")
                self.learning_rate_decay_op = \
                    self.learning_rate.assign(self.learning_rate * 0.99)

                # adam backprop as mentioned in paper
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.global_step = tf.Variable(0, trainable=False, name="global_step")
                # training op
                with tf.device('/gpu:1'):
                    self.train_op = optimizer.minimize(self.loss,
                                                       self.global_step)  # 'magic' tensor that updates the model. updates model to minimize loss.
            # global step is just a count of how many times the variables have been updated

            # checkpoint saver
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            # write summary
            self.log_writer = tf.summary.FileWriter(os.path.join(self.train_dir, 'logs/'), self.session.graph)
            self.summary = tf.Summary()

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
            idx = [random.randint(0, data_size - 1) for _ in range(batch_size)]
        ids = [data[i][1] for i in idx]
        lens = [data[i][0] for i in idx]
        return ids, lens, idx

    def make_input_feed(self, query_batch, qsizes, reply_batch, rsizes,
                        neg_batch=None, neg_sizes=None, training=True):
        if neg_batch:
            reply_batch += neg_batch
            rsizes += neg_sizes
            query_batch += query_batch
            qsizes += qsizes
        return {self.query_sizes: qsizes,
                self.query_inputs: query_batch,
                self.reply_sizes: rsizes,
                self.reply_inputs: reply_batch,
                self.training: training}

    def get_validation_loss(self, queries, replies, data_size, batch_size):
        # data_size = # of queries
        query_batch, query_sizes, idx = self.get_batch(queries, data_size, batch_size)
        reply_batch, reply_sizes, _ = self.get_batch(replies, data_size,
                                                     batch_size, idx)
        negative_reply_batch, neg_reply_sizes, _ = self.get_batch(replies,
                                                                  data_size, batch_size)

        # compute sample loss and do optimize
        feed_dict = self.make_input_feed(query_batch, query_sizes,
                                         reply_batch, reply_sizes, negative_reply_batch, neg_reply_sizes)

        output_feed = [self.global_step, self.loss]
        step, loss = self.session.run(output_feed, feed_dict)

        return step, loss

    def train_step(self, queries, replies, replies_scrambled, data_size, batch_size, generated_responses=None):
        # data_size = # of queries
        query_batch, query_sizes, idx = self.get_batch(queries, data_size, batch_size)
        reply_batch, reply_sizes, _ = self.get_batch(replies, data_size,
                batch_size, idx)

        if not generated_responses:
            if replies_scrambled:
                negative_reply_batch, neg_reply_sizes, _ = self.get_batch(replies_scrambled,
                    data_size, batch_size)
            else: 
                negative_reply_batch, neg_reply_sizes, _ = self.get_batch(replies,
                    data_size, batch_size)
        else:
            if replies_scrambled:
                negative_reply_batch, neg_reply_sizes, _ = self.get_batch(replies_scrambled,
                    data_size, batch_size / 2)
            else: 
                negative_reply_batch, neg_reply_sizes, _ = self.get_batch(replies,
                    data_size, batch_size / 2)
            # Add noisy responses from HRED models for half of the dataset
            generated_reply_batch, generated_reply_sizes, _ = self.get_batch(replies,
                    data_size, batch_size / 2)
            negative_reply_batch += generated_reply_batch
        
            neg_reply_sizes += generated_reply_sizes

        # compute sample loss and do optimize
        feed_dict = self.make_input_feed(query_batch, query_sizes,
                                         reply_batch, reply_sizes, negative_reply_batch, neg_reply_sizes)

        output_feed = [self.global_step, self.train_op, self.loss]
        step, _, loss = self.session.run(output_feed, feed_dict)
        #

        return step, loss

    def init_model(self, checkpoint_dir=None):
        """
        Initialize all variables or load model from checkpoint
        """
        if not checkpoint_dir:
            checkpoint_dir = self.train_dir            

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	print("Loading checkpoint from training dir: " + checkpoint_dir)

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print ('Restoring model from %s'%ckpt.model_checkpoint_path)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print ('Initializing model variables')
            self.session.run(tf.global_variables_initializer())

    def train(self, data_dir, fquery, freply, fquery_scrambled, validation_fquery, validation_freply_true, batch_size=128, steps_per_checkpoint=100):
        queries = data_helpers.load_data(data_dir, fquery, self.qmax_length)
        replies = data_helpers.load_data(data_dir, freply, self.rmax_length)
        if fquery_scrambled:
            fquery_scrambled = data_helpers.load_data(data_dir, fquery_scrambled, self.rmax_length)
        data_size = len(queries)
        validation_queries = data_helpers.load_data(data_dir, validation_fquery, self.qmax_length)
        validation_replies = data_helpers.load_data(data_dir, validation_freply_true, self.rmax_length)
	print("Writing validation + loss to " + self.train_dir)
        if self.additional_negative_samples:
            print("Adding additional samples")
            additional_negative_samples = data_helpers.load_data(data_dir, self.additional_negative_samples, self.rmax_length)
        else:
            additional_negative_samples = ''
            print("Not adding additional samples")
        with self.session.as_default():
            self.init_model()

            checkpoint_path = os.path.join(self.train_dir, "unref.model")
            loss = 0.0
            validation_loss = 0.0
            if os.path.isfile(self.train_dir  + "best_checkpoint.txt"):
                with open(self.train_dir  + "best_checkpoint.txt", "r") as best_file:
                    best_validation_loss = float(best_file.readlines()[1])

            else:
                best_validation_loss = 1000.0
            prev_losses = [1.0]
            impatience = 0.0
            while True:
                step, l = self.train_step(queries, replies, fquery_scrambled, data_size, batch_size, additional_negative_samples)
                _, validation_l = self.get_validation_loss(validation_queries, validation_replies,
                                                           len(validation_queries), batch_size)

                loss += l
                validation_loss += validation_l
                # save checkpoint
                if step % steps_per_checkpoint == 0:
                    loss /= steps_per_checkpoint
                    validation_loss /= steps_per_checkpoint
                    print("global_step %d, loss %f, learning rate %f" \
                          % (step, loss, self.learning_rate.eval()))
                    print("Best validation loss " + str(best_validation_loss))
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        impatience = 0.0
                        print("Saving checkpoint to " + self.train_dir)
                        with open(self.train_dir + "/best_checkpoint.txt", "w+") as best_file:
                            best_file.write(str(step) + "\n")
                            best_file.write(str(validation_loss) + "\n")
                        self.saver.save(self.session, checkpoint_path, global_step=self.global_step)
                    else:
                        impatience += 1
                    print("validation loss %f, best loss %f, impatience %f" % (
                    validation_loss, best_validation_loss, impatience))
                    with open(self.train_dir + "/validation_loss.txt", "a") as validation_file, \
                            open(self.train_dir + "/training_loss.txt", "a") as training_file:
                        training_file.write(str(step) + ":" + str(loss) + "\n")
                        validation_file.write(
                            str(step) + ": " + str(validation_loss) + ", impatience: " + str(impatience) + "\n")

                    if loss > max(prev_losses):
                        self.session.run(self.learning_rate_decay_op)
                    prev_losses = (prev_losses + [loss])[-5:]
                    loss = 0.0
                    validation_loss = 0.0
                    self.log_writer.add_summary(self.summary, step)

                    #                    """ Debug
    #                 query_batch, query_sizes, idx = self.get_batch(queries, data_size, 10)
    #                 reply_batch, reply_sizes, idx = self.get_batch(replies, data_size, 10, idx)
    #                 input_feed = self.make_input_feed(query_batch, query_sizes, reply_batch, reply_sizes,
    #                                                   training=False)
    #                 score, tests = self.session.run([self.pos_score, self.test], input_feed)
    #                 print('-------------')
    #                 for s, t in zip(score[:10], tests[:10]):
    #                     print(s, t)
    #
    # #                   """


    def scores(self, data_dir, fquery, freply, fqvocab, frvocab, checkpoint_dir, init=False):
        if not init:
	    print("not inited, initing now")
            self.init_model(checkpoint_dir=checkpoint_dir)

        queries = data_helpers.load_file(data_dir, fquery)
        replies = data_helpers.load_file(data_dir, freply)
        data_size = len(queries)

        qvocab = data_helpers.load_vocab(data_dir, fqvocab)
        rvocab = data_helpers.load_vocab(data_dir, frvocab)
        scores = []
        with self.session.as_default():
            for query, reply in zip(queries, replies):
                ql, qids = data_helpers.transform_to_id(qvocab, query,
                                                        self.qmax_length)
                rl, rids = data_helpers.transform_to_id(rvocab, reply,
                                                        self.rmax_length)
                feed_dict = self.make_input_feed([qids], [ql], [rids], [rl], training=False)
                score = self.session.run(self.pos_score, feed_dict)
                scores.append(score[0])
            """ Debug
            for i, s in enumerate(scores):
                print i,s
            """
        return scores
