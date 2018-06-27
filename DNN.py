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

    def __init__(self):
        self.string_to_index, self.index_to_string, token_set = \
            data_utils.load_data_with_pickle('processed_data/train_parameter.p')
        self.num_token = len(token_set)

    def init_with_orig_data(self, token_lists):
        '''
        Initialize ML model with training data
        token_lists: [[{type:.., value:..},{..},{..}], [..], [..]]
        '''
        self.dataset = token_lists
        self.tokens_set = data_utils.get_token_set(self.dataset)
        self.num_tokens = len(self.tokens_set)  # 74 经过简化后只有74种token
        print(self.num_tokens)
        # 构建映射字典
        self.index_to_string = {i: s for i, s in enumerate(self.tokens_set)}
        self.string_to_index = {s: i for i, s in enumerate(self.tokens_set)}


    # def one_hot_encoding(self, string):
    #     vector = [0] * self.num_token
    #     vector[self.string_to_index[string]] = 1
    #     return vector

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

    def vector_data_process(self,dataset):
        '''
        读取已经被处理成one_hot_vector的token data，该函数会根据该dataset
        构造x_data and y_data
        :param dataset:
        :return:
        todo：修改x_data和y_data的index
        '''
        x_data = []
        y_data = []
        for index, token in enumerate(dataset):
            if index > 0:
                x_data.append(dataset[index])
                y_data.append(token)
        return x_data, y_data



    # neural network functions
    def create_NN(self):
        self.nn = tflearn.input_data(shape=[None, self.num_token])
        self.nn = tflearn.fully_connected(self.nn, 128)
        self.nn = tflearn.fully_connected(self.nn, 128)
        self.nn = tflearn.fully_connected(self.nn, self.num_token, activation='softmax')
        self.nn = tflearn.regression(self.nn)
        self.model = tflearn.DNN(self.nn)

    # load trained model into object
    def load_model(self, model_file):
        self.create_NN()
        self.model.load(model_file)

    # training ML model
    def train(self, train_data, with_original_data=False):
        print('model training...')
        if with_original_data:
            self.init_with_orig_data(train_data)
            x_data, y_data = self.data_processing()
            self.create_NN()
            self.model.fit(x_data, y_data, n_epoch=1, batch_size=500, show_metric=True)
        else:
            x_data, y_data = self.vector_data_process(train_data)
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
        x = data_utils.one_hot_encoding(prev_token_string, self.string_to_index)
        y = self.model.predict([x])
        predicted_seq = y[0]
        if type(predicted_seq) is np.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.index_to_string[best_number]
        best_token = data_utils.string_to_token(best_string)
        return [best_token]


    def test_model(self, query_test_data):
        correct = 0
        correct_token_list = []
        incorrect_token_list = []

        # token_set = set()
        # for token_sequence in query_test_data:
        #     data_utils.get_token_set(token_sequence)
        #     token_set.update()
        #
        # print(len(token_set))

        for tokens in query_test_data:
            prefix, expection, suffix = data_utils.create_hole(tokens)
            prediction = self.query_test(prefix, suffix)

            if data_utils.token_equals(prediction, expection):
                correct += 1
                correct_token_list.append({'expection': expection, 'prediction': prediction})
            else:
                incorrect_token_list.append({'expection': expection, 'prediction': prediction})
        accuracy = correct / len(query_test_data)
        return accuracy



if __name__ == '__main__':

    #data load and model create

    cc_model = Code_Completion_Model()
    processed_data = data_utils.load_data_with_pickle('processed_data/vec_train_data.p')


    #training model
    use_stored_model = False
    if use_stored_model:
        cc_model.load_model(model_dir)
    else:
        cc_model.train(processed_data, with_original_data=False)



    #test model
    query_test_data = data_utils.load_data_with_file(query_dir)

    accuracy = cc_model.test_model(query_test_data)
    print('query test accuracy: ', accuracy)
