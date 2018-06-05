import tensorflow as tf
import numpy as np
import tflearn
import random


import data_utils


train_dir = 'dataset/programs_800/'
query_dir = 'dataset/programs_200/'
model_dir = 'saved_model/model_parameter'


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 32, 'size of training batch')
tf.flags.DEFINE_integer('hidden_units', 128, 'number of LSTM hidden units')
tf.flags.DEFINE_integer('input_units', 64, 'number of LSTM input units')
tf.flags.DEFINE_integer('num_epoch', 1, 'number of epoch')


class RNN(object):

    def __init__(self, train_data):
        self.data = data_utils.data_processing(train_data)#将用字典表示的token转换为用string表示
        self.token_set = data_utils.get_token_set(train_data)
        self.num_token = len(self.token_set)
        self.token2int = {t:i for i, t in enumerate(self.token_set)}
        self.int2token = {i:t for i, t in enumerate(self.token_set)}

    def one_hot_encoding(self, string):
        vector = [0] * self.tokens_size
        vector[self.string_to_index[string]] = 1
        return vector

    def create_RNN(self):
        self.graph = tf.Graph()

        with graph.as_default():
            self.input_x = tf.placeholder(tf.float32, [None, self.num_token], name='input_x')
            self.output_y = tf.placeholder(tf.float32, [None, self.num_token], name='output_y')

            weight = {
                'in':tf.Variable(tf.truncated_normal([self.num_token, FLAGS.input_units]), name='in_weight'),
                'out':tf.Variable(tf.truncated_normal([self.hidden_units, self.num_token]), name='out_weight')
            }
            bias = {
                'in':tf.Variable(tf.zeros([FLAGS.input_units]), name='in_bias'),
                'out':tf.Variable(tf.zero([self.num_token]), name='out_bias')
            }

            input_lstm = tf.matmul(input_x, weight['in']) + bias['in']

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_units, forget_bias=1.0, state_is_tuple=True)
            init_state = lstm_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)

            output_lstm, final_state = tf.nn.dynamic_rnn(lstm_cell, input_x, initial_state=init_state)

            self.prediction = tf.matmul(output_lstm, weight['out']) + bias['out']
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_y)
            self.optimizer = tf.train.AdagradDAOptimizer(FLAGS.learning_rate).minimize(loss)
            accuracy = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_y, 1))
            self.accuracy = tf.mean(tf.cast(accuracy, tf.float32))


    def train(self, data):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(tf.global_variables_initializer())
        num_batch = len(data) // FLAGS.batch_size

        for epoch in range(FLAGS.num_epoch):
            generator = self.get_batch(data)
            for i in range(num_batch):
                batch_x, batch_y = next(generator)
                feed={self.input_x:batch_x, self.output_y:batch_y}
                self.sess.run(self.optimizer, feed_dict=feed)






    def get_batch(self, data):

        for i in range(0, len(data), FLAGS.batch_size):
            batch_x = data[i:i+FLAGS.batch_size]
            batch_y = data[i+1:i+FLAGS.batch_size+1]
            yield batch_x, batch_y







if __name__ == '__main':
    train_data = data_utils.load_tokens(query_dir, is_simplify=True)
    model = RNN(train_data)
