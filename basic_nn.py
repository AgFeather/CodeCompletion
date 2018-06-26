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
learning_rate = 0.01


def load_tokens(token_dir, is_simplify=True):
    '''
    load token sequence data from input path: token_dir.
    is_simplify: whether or not simplify the value of some variable type(see function for detail)
    return a list whose elements are lists of a token sequence
    '''
    token_files = []  # stored the file's path which ends with 'tokens.json'
    for f in os.listdir(token_dir):
        file_path = os.path.join(token_dir, f)
        if os.path.isfile(file_path) and f.endswith('_tokens.json'):
            token_files.append(file_path)

    # load to a list, element is a token sequence of source code
    token_lists = [json.load(open(f, encoding='utf-8')) for f in token_files]

    def simplify_token(token):
        '''
        Because there are too many values for type: "Identifier", "String", "Numeric",
        NN may be diffcult to train because of these different value.
        So this function can transform these types of variables to a common value
        '''
        if token['type'] == 'Identifier':
            token['value'] = 'id'
        elif token['type'] == 'Numeric':
            token['value'] = '1'
        elif token['type'] == 'String':
            token['value'] = 'string'
        else:
            pass

    if is_simplify:
        for token_sequence in token_lists:
            for token in token_sequence:
                simplify_token(token)
    else:
        pass

    return token_lists


class Code_Completion_Model:
    '''
    Machine Learning model class, including data processing, encoding, model_building,
    training, query_testing, model_save, model_load
    '''

    def __init__(self, token_lists):
        '''
        Initialize ML model with training data
        token_lists: [[{type:.., value:..},{..},{..}], [..], [..]]
        '''
        time_begin = time.time()
        self.token_lists = token_lists
        self.tokens_set = set()
        for token_sequence in token_lists:
            for token in token_sequence:
                self.tokens_set.add(self.token_to_string(token))
        self.tokens_list = list(self.tokens_set)
        self.tokens_list.sort()
        self.tokens_size = len(self.tokens_set)  # 213
        self.index_to_string = {i: s for i, s in enumerate(self.tokens_list)}
        self.string_to_index = {s: i for i, s in enumerate(self.tokens_list)}
        time_end = time.time()
        print('model initialization time cost: ', time_end - time_begin)

    # data processing functions
    def token_to_string(self, token):
        return token['type'] + '~$$~' + token['value']

    def string_to_token(self, string):
        tokens = string.split('~$$~')
        return {'type': tokens[0], 'value': tokens[1]}

    # encoding token sequence as one_hot_encoding
    def one_hot_encoding(self, string):
        vector = [0] * self.tokens_size
        vector[self.string_to_index[string]] = 1
        return vector

    # generate X_train data and y_label for ML model
    def data_processing(self):
        '''
        first, transform a token in dict form to a type-value string
        x_data is a token, y_label is the previous token of x_data
        '''
        x_data = []
        y_data = []
        print('data processing is begining...')
        for token_sequence in self.token_lists:  # token_sequence of each source code
            for index, token in enumerate(token_sequence):  # each token(type_value) in source code
                if index > 0:
                    token_string = self.token_to_string(token)
                    prev_token = self.token_to_string(token_sequence[index - 1])
                    x_data.append(self.one_hot_encoding(prev_token))
                    y_data.append(self.one_hot_encoding(token_string))
        print('data processing is finished..')
        pickle.dump(x_data, open('processed_data/saved_x_data_for_basic.p', 'wb'))
        pickle.dump(y_data, open('processed_data/saved_y_data_for_basic.p', 'wb'))
        return x_data, y_data

    # neural network functions
    def create_NN(self):
        tf.reset_default_graph()
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.tokens_size],name='input_x')
        self.output_y = tf.placeholder(dtype=tf.float32, shape=[None, self.tokens_size],name='output_y')
        self.nn = tf.layers.dense(inputs=self.input_x, units=128, activation=tf.nn.relu,name='hidden_1')
        self.output = tf.layers.dense(inputs=self.nn, units=self.tokens_size, activation=None,name='prediction')
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
    def train(self, use_saved_data=True):
        time_begin = time.time()
        if use_saved_data:
            x_data, y_data = self.load_x_y_data()
        else:
            x_data, y_data = self.data_processing()

        time_end = time.time()
        print('data processing time cost: ', time_end - time_begin)
        self.create_NN()
        time_begin = time.time()
        with tf.Session() as sess:
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
        prev_token_string = self.token_to_string(prefix[-1])
        x = self.one_hot_encoding(prev_token_string)
        with tf.Session() as sess:
            feed = {self.input_x: x}
            predict_list = sess.run(self.output, feed_dict=feed)
            prediction = tf.argmax(predict_list, 1)
            best_string = self.index_to_string[prediction]
            best_token = self.string_to_token(best_string)
        return [best_token]





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
    start_time = time.time()
    dataset = load_tokens(train_dir)
    code_completion = Code_Completion_Model(dataset)

    train_time = code_completion.train()

    end_time = time.time()
    print('total time cost: %.2f s, model training cost: %.2f s' % (end_time - start_time, train_time))

    # query_test_data = load_tokens(query_dir)
    # correct = 0
    # correct_token_list = []
    # incorrect_token_list = []
    # for tokens in query_test_data:
    #     prefix, expection, suffix = create_hole(tokens)
    #     prediction = code_completion.query_test(prefix, suffix)
    #     if token_equals(prediction, expection):
    #         correct += 1
    #         correct_token_list.append({'expection': expection, 'prediction': prediction})
    #     else:
    #         incorrect_token_list.append({'expection': expection, 'prediction': prediction})
    # accuracy = correct / len(query_test_data)
    # print('query test accuracy: ', accuracy)