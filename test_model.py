import json
import random
import tensorflow as tf
import numpy as np
import tflearn
import os

train_dir = 'dataset/programs_800/'
query_dir = 'dataset/programs_200/'
model_file = 'trained_model_parameter'


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


import time


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
        for token_sequence in self.token_lists:  # token_sequence of each source code
            for index, token in enumerate(token_sequence):  # each token(type_value) in source code
                if index > 0:
                    token_string = self.token_to_string(token)
                    prev_token = self.token_to_string(token_sequence[index - 1])
                    x_data.append(self.one_hot_encoding(prev_token))
                    y_data.append(self.one_hot_encoding(token_string))
        print('data processing is finished..')
        return x_data, y_data

    # neural network functions
    def create_NN(self):
        tf.reset_default_graph()
        self.nn = tflearn.input_data(shape=[None, self.tokens_size])
        self.nn = tflearn.fully_connected(self.nn, 128)
        self.nn = tflearn.fully_connected(self.nn, 128)
        self.nn = tflearn.fully_connected(self.nn, self.tokens_size, activation='softmax')
        self.nn = tflearn.regression(self.nn)
        self.model = tflearn.DNN(self.nn)

    # load trained model into object
    def load_model(self, model_file):
        self.create_NN()
        self.model.load(model_file)

    # training ML model
    def train(self):
        start_time = time.time()
        x_data, y_data = self.data_processing()
        self.create_NN()
        self.model.fit(x_data, y_data, n_epoch=1, batch_size=500, show_metric=True)
        end_time = time.time()
        return end_time - start_time

    # save trained model to model path
    def save_model(self, model_file):
        self.model.save(model_file)

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
        y = self.model.predict([x])
        predicted_seq = y[0]
        if type(predicted_seq) is np.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.index_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]




if __name__ == '__main__':
    start_time = time.time()
    dataset = load_tokens(train_dir)
    load_time = time.time()
    print('time cost of loading data: %.2f s'%(load_time - start_time))

    code_completion = Code_Completion_Model(dataset)
    x, y = code_completion.data_processing()
    end_time = time.time()

    print('time cost of processing data: %.2f s, size of training data: %d'%(end_time-load_time, len(x)))