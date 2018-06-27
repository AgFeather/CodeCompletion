import json
import random
import tensorflow as tf
import numpy as np
import tflearn
import os
import pickle
import time

train_dir = 'dataset/programs_800/'
query_dir = 'dataset/programs_200/'
model_file = 'trained_model_parameter'

epoch_num = 1
batch_size = 64
learning_rate = 0.002




class Code_Completion_Model:


    def __init__(self):
        pass


    # neural network functions
    def create_NN(self):
        tf.reset_default_graph()
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.tokens_size],name='input_x')
        self.output_y = tf.placeholder(dtype=tf.float32, shape=[None, self.tokens_size],name='output_y')
        self.nn = tf.layers.dense(inputs=self.input_x, units=128, activation=tf.nn.relu,name='hidden_1')
        self.output = tf.layers.dense(inputs=self.nn, units=self.tokens_size, activation=None,name='prediction')
        self.prediction_index = tf.argmax(self.ouput, 1)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.output_y,name='loss')
        self.loss = tf.reduce_sum(self.loss)
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        self.equal = tf.equal(tf.argmax(self.output_y, 1), tf.argmax(self.output, 1))
        self.accuarcy = tf.reduce_mean(tf.cast(self.equal, tf.float32),name='accuracy')


    def load_x_y_data(self):
        x_data = pickle.load(open('processed_data/x_train_data.p', 'rb'))
        y_data = pickle.load(open('processed_data/y_train_data.p', 'rb'))
        token_set, string2int, int2string = pickle.load(open('processed_data/x_y_parameter.p', 'rb'))
        self.index_to_string = int2string
        self.string_to_index = string2int
        self.tokens_set = token_set
        self.tokens_size = len(token_set)
        return x_data, y_data
    # training ML model
    def train(self):
        time_begin = time.time()

        x_data, y_data = self.load_x_y_data()

        time_end = time.time()
        print('data processing time cost: ', time_end - time_begin)
        self.create_NN()
        time_begin = time.time()
        self.sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            for i in range(0, len(x_data), batch_size):
                batch_x = x_data[i:i + batch_size]
                batch_y = y_data[i:i + batch_size]
                feed = {self.input_x: batch_x, self.output_y: batch_y}
                sess.run(self.optimizer, feed_dict=feed)
                if (i // batch_size) % 500 == 0:
                    show_acc = sess.run(self.accuarcy, feed_dict=feed)
                    print('epoch: %d, training_step: %d, accuracy:%.3f' % (epoch, i, show_acc))

        time_end = time.time()
        print('training time cost: ', time_end - time_begin)
        return time_end - time_begin

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
        feed = {self.input_x:pre_token_x}
        prediction = self.sess.run(self.prediction_index, feed)[0]

        # if type(predicted_seq) is np.ndarray:
        #     predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.index_to_string[prediction]
        best_token = data_utils.string_to_token(best_string)
        return [best_token]

    def test_model(self, query_test_data):
        correct = 0
        correct_token_list = []
        incorrect_token_list = []

        for token_sequence in query_test_data:
            prefix, expection, suffix = data_utils.create_hole(token_sequence)
            prediction = self.query_test(prefix, suffix)

            if data_utils.token_equals(prediction, expection):
                correct += 1
                correct_token_list.append({'expection': expection, 'prediction': prediction})
            else:
                incorrect_token_list.append({'expection': expection, 'prediction': prediction})
        accuracy = correct / len(query_test_data)
        return accuracy





def create_hole(tokens, max_hole_size = 2):
    '''
    input: a tokens sequence of source code and max_hole_size
    return: hole token to be predicted (expection)
            token sequence before the hole(prefix)
            token sequence after the hole(suffix)
    '''
    hole_size = min(random.randint(1, max_hole_size), len(tokens) - 1)
    hole_start_index = random.randint(1, len(tokens) - hole_size)
    hole_end_index = hole_start_index + hole_size
    prefix = tokens[0 : hole_start_index]
    expection = tokens[hole_start_index : hole_end_index]
    suffix = tokens[hole_end_index : 0]
    return prefix, expection, suffix

def token_equals(token1, token2):
    '''
    Determining whether input two tokens are equal or not
    '''
    if len(token1) != len(token2):
        return False
    for index, t1 in enumerate(token1):
        t2 = token2[index]
        if t1['type'] != t2['type'] or t1['value'] != t2['value']:
            return False
    return True


if __name__ == '__main__':


    code_completion = Code_Completion_Model()

    train_time = code_completion.train()


    # test model
    query_test_data = data_utils.load_data_with_file(query_dir)
    accuracy = cc_model.test_model(query_test_data)
    print('query test accuracy: ', accuracy)