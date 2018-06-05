import tensorflow as tf
import numpy as np
import tflearn
import random


import data_utils


train_dir = 'dataset/programs_800/'
query_dir = 'dataset/programs_200/'
model_dir = 'saved_model/model_parameter'


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
        self.dataset = token_lists
        self.tokens_set = data_utils.get_token_set(self.dataset)
        self.tokens_size = len(self.tokens_set)  # 213
        # 构建映射字典
        self.index_to_string = {i: s for i, s in enumerate(self.tokens_set)}
        self.string_to_index = {s: i for i, s in enumerate(self.tokens_set)}


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
        for index, token in enumerate(self.dataset):
            if index > 0:
                token_string = data_utils.token_to_string(token)
                prev_token = data_utils.token_to_string(self.dataset[index - 1])
                x_data.append(self.one_hot_encoding(prev_token))
                y_data.append(self.one_hot_encoding(token_string))
        return x_data, y_data

    # neural network functions
    def create_NN(self):
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
        print('model training...')
        x_data, y_data = self.data_processing()
        self.create_NN()
        self.model.fit(x_data, y_data, n_epoch=1, batch_size=500, show_metric=True)

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
        prev_token_string = data_utils.token_to_string(prefix[-1])
        x = self.one_hot_encoding(prev_token_string)
        y = self.model.predict([x])
        predicted_seq = y[0]
        if type(predicted_seq) is np.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.index_to_string[best_number]
        best_token = data_utils.string_to_token(best_string)
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

    #data load and model create
    train_data = data_utils.load_tokens(train_flag=False, is_simplify=True)
    cc_model = Code_Completion_Model(train_data)

    #training model
    use_stored_model = False
    if use_stored_model:
        cc_model.load_model(model_dir)
    else:
        cc_model.train()
        cc_model.save_model(model_dir)


    #test model
    query_test_data = data_utils.load_tokens(query_dir)
    correct = 0
    correct_token_list = []
    incorrect_token_list = []
    for tokens in query_test_data:
        prefix, expection, suffix = create_hole(tokens)
        prediction = cc_model.query_test(prefix, suffix)
        if token_equals(prediction, expection):
            correct += 1
            correct_token_list.append({'expection': expection, 'prediction': prediction})
        else:
            incorrect_token_list.append({'expection': expection, 'prediction': prediction})
    accuracy = correct / len(query_test_data)
    print('query test accuracy: ', accuracy)
