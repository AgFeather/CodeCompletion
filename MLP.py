import tensorflow as tf
import numpy as np
import time
from sklearn.model_selection import train_test_split

import data_utils


'''
Using multiply layers perceptrons to predict token
'''
x_train_data_path = 'processed_data/x_train_data.p'
y_train_data_path = 'processed_data/y_train_data.p'
train_data_parameter = 'processed_data/x_y_parameter.p'

tensorboard_log_path = './logs/MLP/'

query_dir = 'dataset/programs_200/'

epoch_num = 2
batch_size = 128
learning_rate = 0.005
test_epoch = 3
hidden_size = 128



class Code_Completion_Model(object):

    def __init__(self, x_data, y_data, token_set, string2int, int2string):
        self.x_data = x_data
        self.y_data = y_data
        self.x_data, self.valid_x, self.y_data, self.valid_y = \
                train_test_split(x_data, y_data, train_size=0.9, random_state=100)
        self.index_to_string = int2string
        self.string_to_index = string2int
        self.tokens_set = token_set
        self.tokens_size = len(token_set)
        self.data_size = len(self.x_data)

    # neural network functions
    def create_NN(self):
        tf.reset_default_graph()
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.tokens_size], name='input_x')
        self.output_y = tf.placeholder(dtype=tf.float32, shape=[None, self.tokens_size], name='output_y')
        weights = {'h1':tf.Variable(tf.truncated_normal(shape=[self.tokens_size, hidden_size])),
                   'h2':tf.Variable(tf.truncated_normal(shape=[hidden_size, hidden_size])),
                   'output':tf.Variable(tf.truncated_normal(shape=[hidden_size, self.tokens_size]))}
        biases = {'h1':tf.Variable(tf.constant(0.1, shape=[hidden_size], dtype=tf.float32)),
                  'h2':tf.Variable(tf.constant(0.1, shape=[hidden_size], dtype=tf.float32)),
                  'output':tf.Variable(tf.constant(0.1, shape=[self.tokens_size], dtype=tf.float32))}

        h1_layer = tf.matmul(self.input_x, weights['h1']) + biases['h1']
        h1_layer = tf.nn.relu(h1_layer)
        h2_layer = tf.matmul(h1_layer, weights['h2']) + biases['h2']
        h2_layer = tf.nn.relu(h2_layer)
        output_layer = tf.matmul(h2_layer, weights['output']) + biases['output']
        self.prediction = tf.argmax(output_layer, 1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=self.output_y)
        self.loss = tf.reduce_mean(loss)
        self.optimizer_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        equal = tf.equal(tf.argmax(output_layer, 1), tf.argmax(self.output_y, 1))
        accuracy = tf.cast(equal, tf.float32)
        self.accuracy = tf.reduce_mean(accuracy)

        tf.summary.histogram('weight1', weights['h1'])
        tf.summary.histogram('weight2', weights['h2'])
        tf.summary.histogram('output_weight', weights['output'])
        tf.summary.histogram('bias1', biases['h1'])
        tf.summary.histogram('bias2', biases['h2'])
        tf.summary.histogram('output_bias', biases['output'])
        tf.summary.scalar('train_loss', self.loss)
        tf.summary.scalar('train_accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()


    def train(self):
        self.create_NN()
        self.sess = tf.Session()
        writer = tf.summary.FileWriter(tensorboard_log_path, self.sess.graph)
        time_begin = time.time()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            for i in range(0, self.data_size, batch_size):
                batch_x = self.x_data[i:i + batch_size]
                batch_y = self.y_data[i:i + batch_size]
                feed = {self.input_x: batch_x, self.output_y: batch_y}
                _, summary_str = self.sess.run([self.optimizer_op, self.merged], feed_dict=feed)
                writer.add_summary(summary_str, epoch*self.data_size + i)
                writer.flush()
                if (i // batch_size) % 2000 == 0:
                    valid_feed = {self.input_x:self.valid_x, self.output_y:self.valid_y}
                    valid_loss, valid_acc = self.sess.run([self.loss, self.accuracy], feed_dict=valid_feed)
                    show_loss, show_acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed)
                    print('epoch: %d, training_step: %d, train_loss: %.2f, train_accuracy:%.3f'
                          % (epoch, i, show_loss, show_acc))
                    print('epoch: %d, training_step: %d, valid_loss: %.2f, valid_accuracy:%.3f'
                          %(epoch, i, valid_loss, valid_acc))
        time_end = time.time()

        print('training time cost: %.3f ms' %(time_end - time_begin))

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
        feed = {self.input_x: [pre_token_x]}
        prediction = self.sess.run(self.prediction, feed)[0]
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

    # model train
    model = Code_Completion_Model(x_data, y_data, token_set, string2int, int2string)
    model.train()

    # test model
    query_test_data = data_utils.load_data_with_file(query_dir)
    test_accuracy = 0.0
    for i in range(test_epoch):
        accuracy = model.test_model(query_test_data)
        print('test epoch: %d, query test accuracy: %.3f' % (i, accuracy))
        test_accuracy += accuracy
    print('total test accuracy: %.3f' % (test_accuracy / test_epoch))