import json
import random
import tensorflow as tf
import numpy as np
import time
import data_utils


'''
Using CNN with sliding windows to predict token.
'''




x_train_data_path = 'processed_data/num_train_data.p'
y_train_data_path = 'processed_data/y_train_data.p'
train_data_parameter = 'processed_data/x_y_parameter.p'

tensorboard_log_path = './logs/CNN'

query_dir = 'dataset/programs_200/'

epoch_num = 1
batch_size = 64
learning_rate = 0.001
test_epoch = 3
embed_dim = 32
sliding_window = [2,3,4,5]
filter_num = 4
hidden_size = 128
context_size = 5

rnn_num_layers = 1
rnn_num_nuits = 128

class Code_Completion_Model:

    def __init__(self, x_data, y_data, token_set, string2int, int2string):
        '''
        :param x_data: a numerical representation of token sequence,  mapping dict is int2string
        :param y_data: a one_hot encoding representation of label y
        :param token_set: set(token)
        :param string2int: mapping dict
        :param int2string: mapping dict
        '''
        self.x_data, self.y_data = x_data, y_data
        self.data_size = len(self.x_data)
        self.index_to_string = int2string
        self.string_to_index = string2int
        self.tokens_set = token_set
        self.tokens_size = len(token_set)



    # neural network functions
    def create_CNN(self):
        tf.reset_default_graph()
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, context_size], name='input_x')
        self.output_y = tf.placeholder(dtype=tf.float32, shape=[None, self.tokens_size], name='output_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        self.embedding_matrix = tf.Variable(tf.truncated_normal(
            [self.tokens_size, embed_dim]), name='embedding_matrix')
        self.embedding_represent = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x, name='embedding_represent')
       # self.embedding_represent = tf.expend_dims(self.embedding_represent, -1)

        conv_layers_list = []
        weights = {}
        biases = {}
        for window in sliding_window:
            with tf.name_scope('convolution_layer_{}'.format(window)):
                conv_weight = tf.Variable(tf.truncated_normal(
                    shape=[window, embed_dim, 1, filter_num]), name='conv_weight')
                conv_bias = tf.Variable(tf.truncated_normal(
                    shape=[window, embed_dim, 1, filter_num]), name='conv_bias')
                weights['conv%d' % window] = conv_weight
                biases['conv%d' % window] = conv_bias
                conv_layer = tf.nn.conv2d(
                    self.embedding_represent, conv_weight, strides=[1,1,1,1],padding='SAME', name='conv_layer_1')
                conv_layer = tf.nn.relu(conv_layer, name='relu_layer')
                avgpool_layer = tf.nn.avg_pool(
                    conv_layer, [1, 2, 2, 1], [1,1,1,1], padding='VALID', name='avgpool_layer')
                # maxpooling or avgpooling? parameter adjust？
                conv_layers_list.append(avgpool_layer)

        representation_layer = tf.concat(conv_layers_list, 3, name='concat_conv_layers')
        weights['h1'] = tf.Variable(tf.truncated_normal(
            shape=[]), name='h1_weight')
        biases['h1'] = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[hidden_size]))
        self.representation_layer = tf.matmul(weights['h1'], representation_layer) + biases['h1']


    def create_RNN(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_num_nuits, forget_bias=1.0, state_is_tuple=True)
        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        cells = tf.contrib.rnn.MultiRNNCell([drop] * rnn_num_layers)
        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(
            cells, self.representation_layer, initial_state=init_state, time_major=False)
        output_weight = tf.Variable(tf.random_uniform(shape=[], dtype=tf.float32))
        output_bias = tf.Variable(tf.constant([0.1], dtype=tf.float32, shape=[self.tokens_size]))
        self.prediction = outputs * output_weight + output_bias


    def train(self):
        self.create_CNN()
        self.create_RNN()
        self.sess = tf.Session()
        writer = tf.summary.FileWriter(tensorboard_log_path, self.sess.graph)
        time_begin = time.time()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            batch_generator = self.get_batch()
            for i in range(0, len(self.x_data), batch_size):
                batch_x, batch_y = next(batch_generator)
                feed = {self.input_x: batch_x, self.output_y: batch_y, self.keep_prob: 0.5}
                _, summary_str = self.sess.run([self.optimizer, self.merged], feed_dict=feed)
                writer.add_summary(summary_str, i)
                writer.flush()
                if (i // batch_size) % 2000 == 0:
                    show_loss, show_acc = self.sess.run([self.loss, self.accuarcy], feed_dict=feed)
                    print('epoch: %d, training_step: %d, loss: %.2f, accuracy:%.3f' % (epoch, i, show_loss, show_acc))
        time_end = time.time()

        print('training time cost: %.3f ms' %(time_end - time_begin))


    def get_batch(self):
        for i in range(0, self.data_size, batch_size):
            if i >= context_size-1:
                batch_x = np.zeros(shape=[batch_size, context_size], dtype=np.int32)
                batch_y = np.array(self.y_data[i:i+batch_size])
                for b in range(batch_size):
                    batch_x[i] = self.x_data[i+b:i+b+context_size]
                yield batch_x, batch_y



    # query test
    def query_test(self, prefix, suffix):
        '''
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        In this function, use only one token before hole token to predict
        return: the most probable token
        '''
        prev_token_string = data_utils.token_to_string(prefix[-1])
        pre_token_x = data_utils.one_hot_encoding(prev_token_string, self.string_to_index)
        feed = {self.input_x: [pre_token_x], self.keep_prob:1}
        prediction = self.sess.run(self.prediction_index, feed)[0]
        best_string = self.index_to_string[prediction]
        best_token = data_utils.string_to_token(best_string)
        return [best_token]

    def test_model(self, query_test_data):
        correct = 0.0
        correct_token_list = []
        incorrect_token_list = []
        for token_sequence in query_test_data:
            prefix, expection, suffix = data_utils.create_hole(token_sequence)
            prediction = self.query_test(prefix, suffix)[0]
            if data_utils.token_equals([prediction], expection):
                correct += 1
                correct_token_list.append({'expection': expection, 'prediction': prediction})
            else:
                incorrect_token_list.append({'expection': expection, 'prediction': prediction})
        accuracy = correct / len(query_test_data)
        return accuracy



if __name__ == '__main__':


    x_data = data_utils.load_data_with_pickle(x_train_data_path)
    y_data = data_utils.load_data_with_pickle(y_train_data_path)
    token_set, string2int, int2string = data_utils.load_data_with_pickle(train_data_parameter)

    #model train
    model = Code_Completion_Model(x_data, y_data, token_set, string2int, int2string)
    model.train()

    # test model
    query_test_data = data_utils.load_data_with_file(query_dir)
    test_accuracy = 0.0
    for i in range(test_epoch):
        accuracy = model.test_model(query_test_data)
        print('test epoch: %d, query test accuracy: %.3f'%(i, accuracy))
        test_accuracy += accuracy
    print('total test accuracy: %.3f'%(test_accuracy/test_epoch))