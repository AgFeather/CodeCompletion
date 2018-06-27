import tensorflow as tf
import numpy as np
import tflearn
import random

import data_utils

train_dir = 'dataset/programs_800/'
query_dir = 'dataset/programs_200/'
model_dir = 'saved_model/sliding_window'

slide_windows = [1, 2]
hidden_units = 128
epoch_num = 3
learning_rate = 0.001
batch_size = 256


class Code_Completion_Model:
    def __init__(self):
        self.string_to_index, self.index_to_string, token_set = \
            data_utils.load_data_with_pickle('mapping_dict.p')
        self.num_token = len(token_set)


    def split_with_windows(self, token_list, window_size):
        # 给定train_x, train_y list，元素由
        train_x = []
        train_y = []
        for index, token_vec in enumerate(token_list):
            if index > window_size - 1:
                prev_token_list = []
                for i in range(window_size):
                    prev_token = token_list[index - i - 1]
                    prev_token_list.extend(prev_token)
                train_x.append(prev_token_list)
                train_y.append(token_vec)
        return train_x, train_y

    # neural network functions
    def create_NN(self, window_size):
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            input_x = tf.placeholder(tf.float32, [None, window_size * self.num_token], name='input_x')
            output_y = tf.placeholder(tf.float32, [None, self.num_token], name='output_y')

            fc1 = tf.layers.dense(
                inputs=input_x, units=window_size * hidden_units,
                activation=tf.nn.relu)
            fc2 = tf.layers.dense(
                inputs=fc1, units=window_size * hidden_units,
                activation=tf.nn.relu)
            fc3 = tf.layers.dense(
                inputs=fc2, units=window_size * hidden_units,
                activation=tf.nn.relu)
            output_layer = tf.layers.dense(
                inputs=fc3, units=self.tokens_num, activation=None, name='output_layer')

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=output_y)
            loss = tf.reduce_mean(loss, name='loss')
            optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer').minimize(loss)
            accuracy = tf.equal(tf.argmax(output_layer, 1), tf.argmax(output_y, 1))
            accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32), name='accuracy')

        return graph

    #  return input_x, output_y, output_layer, loss, optimizer, accuracy

    # load trained model into object
    def load_model(self, model_file):
        self.create_NN()
        self.model.load(model_file)

    # training ML model
    def train(self):
        print('model training...')
        model_list = []

        def get_batch(x_data, y_data):
            for i in range(0, len(x_data), batch_size):
                batch_x = x_data[i:i + batch_size]
                batch_y = y_data[i:i + batch_size]
                yield i // batch_size, batch_x, batch_y

        for window_size in slide_windows:
            graph = self.create_NN(window_size)
            model_list.append(graph)

        token_list = self.data_processing()
        for window_size, graph in zip(slide_windows, model_list):
            print('with window_size: %d' % window_size)
            x_data, y_data = self.split_with_windows(token_list, window_size)
            with tf.Session(graph=graph) as sess:
                saver = tf.train.Saver()
                input_x = graph.get_tensor_by_name('input_x:0')
                output_y = graph.get_tensor_by_name('output_y:0')
                loss = graph.get_tensor_by_name('loss:0')
                optimizer = graph.get_operation_by_name('optimizer')
                accuracy = graph.get_tensor_by_name('accuracy:0')
                sess.run(tf.global_variables_initializer())

                for epoch in range(epoch_num):
                    geneator = get_batch(x_data, y_data)
                    for i, batch_x, batch_y in geneator:
                        #     batch_x, batch_y = next(geneator)
                        feed = {input_x: batch_x, output_y: batch_y}
                        sess.run(optimizer, feed)
                        if (i % 300 == 0):
                            s_loss, s_accu = sess.run([loss, accuracy], feed)
                            print('epoch: %d, step: %d, loss: %.2f, accuracy:%.2f' %
                                  (epoch, i, s_loss, s_accu))
                saver.save(sess, model_dir + '_' + str(window_size) + '.ckpt.meta')

    # query test
    def query_test(self, prefix, suffix, window_size):
        '''
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        In this function, use only one token before hole token to predict
        return: the most probable token
        '''
        test_tokens = []
        new_saver = tf.train.import_meta_graph(model_dir + '.meta')
        for i in range(window_size):
            prev_token_string = self.token_to_string(prefix[-i - 1])
            x = self.one_hot_encoding(prev_token_string)
            test_tokens.extend(x)
        with tf.Session() as sess:
            new_saver.restore(sess, model_dir)
            graph = tf.get_default_graph()
            input_x = graph.get_tensor_by_name('input_x:0')
            accuracy = graph.get_tensor_by_name('accuracy:0')
            output_layer = graph.get_tensor_by_name('output_layer:0')
            feed = {input_x: test_tokens}
            show_accu, prediction = sess.run([accuracy, output_layer], feed)
            print('window_size: %d, accuracy:%.3f' % (window_size, show_accu))
        return show_accu



if __name__ == '__main__':

    #data load and model create
    train_data = data_utils.load_tokens(query_dir, is_simplify=True)
    cc_model = Code_Completion_Model(train_data)


    #training model
    cc_model.train()


    #test model
    test_data = data_utils.load_tokens(query_dir)
    correct = 0

    for token_sequence in test_data:
        prefix, expection, suffix = create_hole(token_sequence)
        for window_size in range(slide_windows):
            prediction = cc_model.query_test(prefix, expection, suffix, window_size)


