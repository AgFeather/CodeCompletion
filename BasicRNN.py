import tensorflow as tf
import numpy as np
import pickle
import data_utils


'''
Using MultiRNN to pridect token. with LSTM cell
'''



processed_data_path = 'processed_data/vec_train_data.p'
data_parameter_path = 'processed_data/train_parameter.p'
tensorboard_log_path = 'logs/MultiRNN'

time_step = 1
unit_num = 128
batch_size = 64
epoch_num = 1
learning_rate = 0.001
num_layers = 2




class RNN_model(object):
    def __init__(self, string2int, int2string, token_set, dataset):
        self.string2int, self.int2string, self.token_set = string2int, int2string, token_set
        self.dataset = dataset
        self.token_size = len(self.token_set)
        self.dataset_size = len(self.dataset)


    def create_NN(self):
        tf.reset_default_graph()
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.token_size])
        self.output_y = tf.placeholder(tf.float32, shape=[None, self.token_size])
        self.keep_prob = tf.Variable(1.0, dtype=tf.float32, name='keep_prob')

        self.weights = {
            'in': tf.Variable(tf.random_normal([self.token_size, unit_num])),
            'out': tf.Variable(tf.random_normal([unit_num, self.token_size]))
        }
        self.biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[unit_num])),
            'out': tf.Variable(tf.constant(0.1, shape=[self.token_size]))
        }

        #    reshape_input = tf.reshape(self.input_x, [-1, self.token_size])
        reshape_input = tf.matmul(self.input_x, self.weights['in']) + self.biases['in']
        reshape_input = tf.reshape(reshape_input, [-1, time_step, unit_num])

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(unit_num, forget_bias=1.0, state_is_tuple=True)


        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        cells = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(
            lstm_cell, reshape_input, initial_state=init_state, time_major=False)

     #   print(outputs.get_shape())
        outputs = tf.unstack(outputs, axis=1)  # 沿着time step方向切割tensor为一个list，list长度为time step
        self.prediction = tf.matmul(outputs[0], self.weights['out']) + self.biases['out']

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.output_y)
        self.loss = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.accuracy = tf.equal(tf.argmax(self.output_y, 1), tf.argmax(self.prediction, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

    def train(self):
        print('training begin...')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            for i in range(0, self.dataset_size, batch_size):
                batch_x = self.dataset[i:i + batch_size]
                batch_y = self.dataset[i + 1:i + batch_size + 1]
                feed = {self.input_x: batch_x, self.output_y: batch_y}
                self.sess.run(self.optimizer, feed_dict=feed)

                if ((i / batch_size) % 500 == 0):
                    show_accu, show_loss = self.sess.run([self.accuracy, self.loss], feed)
                    print('epoch: %d, training step: %d, loss: %.3f, accuracy: %.3f' % (epoch, i, show_loss, show_accu))



    # query test
    def query_test(self, prefix, suffix):
        '''
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        '''
        test_previous_tokens = []
        for token in prefix:
            prev_token_string = data_utils.token_to_string(token)
            pre_token_x = data_utils.one_hot_encoding(prev_token_string, self.string_to_index)
            test_previous_tokens.append(pre_token_x)

        feed = {self.input_x: test_previous_tokens, self.keep_prob:1}
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
    train_data = data_utils.load_data_with_pickle(processed_data_path)
    string2int, int2string, token_set = data_utils.load_data_with_pickle(data_parameter_path)

    model = RNN_model(string2int, int2string, token_set, train_data)

    # model train
    model.create_NN()
    model.train()

    # model test
    query_test_data = data_utils.load_data_with_file(query_dir)
    test_accuracy = 0.0
    for i in range(test_epoch):
        accuracy = model.test_model(query_test_data)
        print('test epoch: %d, query test accuracy: %.3f' % (i, accuracy))
        test_accuracy += accuracy
    print('total test accuracy: %.3f' % (test_accuracy / test_epoch))