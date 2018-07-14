import tensorflow as tf
import numpy as np
import pickle
import data_utils


'''
使用基本的RNN对token进行预测
'''






processed_data_path = 'processed_data/vec_train_data.p'
data_parameter_path = 'processed_data/train_parameter.p'

time_step = 1
unit_num = 128
batch_size = 64
epoch_num = 1
learning_rate = 0.002
num_layers = 2




class RNN_model(object):
    def __init__(self, string2int, int2string, token_set, dataset):
        self.string2int, self.int2string, self.token_set = string2int, int2string, token_set
        self.dataset = dataset
        self.token_size = len(self.token_set)
        self.dataset_size = len(self.dataset)

    # def load_data(self, dataset):
    #     #   self.dataset = pickle.load(open(processed_data_path, 'rb'))
    #     self.dataset = dataset
    #     self.string2int, self.int2string, self.token_set = \
    #         pickle.load(open(data_parameter_path, 'rb'))
    #     self.token_size = len(self.token_set)
    #     self.dataset_size = len(self.dataset)
    #     print('data loading...')

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

        '''
        # Todo:添加dorpout，Multiple LSTM
        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, outputs_keep_prob=self.keep_prob)
        cells = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
        '''

        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(
            lstm_cell, reshape_input, initial_state=init_state, time_major=False)

        print(outputs.get_shape())
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


if __name__ == '__main__':
    train_data = data_utils.load_data_with_pickle(processed_data_path)
    string2int, int2string, token_set = data_utils.load_data_with_pickle(data_parameter_path)

    model = RNN_model(string2int, int2string, token_set)
#    model.load_data(train_data)
    model.create_NN()
    model.train()