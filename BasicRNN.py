import tensorflow
import numpy
import pickle
import data_utils


processed_data_path = 'processed_data/vec_train_data.p'
data_parameter_path = 'processed_data/train_parameter.p'

time_step = 20
unit_num = 128
batch_size = 64
epoch_num = 1
learning_rate = 0.01

class RNN_model(object):
    def __init__(self):
        self.load_data()
    def load_data(self):
        self.dataset = pickle.load(open(processed_data_path, 'rb'))
        self.string2int, self.int2string, self.token_set = pickle(open(data_parameter_path), 'rb')
        self.token_size = len(self.token_set)
        self.dataset_size = len(self.dataset)

    def create_NN(self):
        self.input_x = tf.placeholder(tf.float32, [None, time_step, self.token_size])
        self.output_y = tf.placeholder(tf.float32, [None, time_step, self.token_size])
        self.weights = {
            'in':tf.Variable(tf.random_nromal([self.token_size, unit_num])),
            'out':tf.Variable(tf.random_nromal(unit_num, self.token_size))
        }
        self.biases = {
            'in':tf.Variable(tf.constant(0.1, shape=[unit_num])),
            'out':tf.Variable(tf.constant(0.1, shape=[self.token_size]))
        }
        reshape_input = tf.reshape(self.input_x, [-1, self.token_size])
        reshape_input = tf.matmul(reshape_input, self.weights['in']) + self.biases['in']
        reshape_input = tf.reshape(reshape_input, [-1, time_step, unit_num])
        lstm_cell = tf.contrib.rnn.BiasicLSTMCell(unit_num, forget_bais=1.0)
        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, reshape_input, initial_state=init_state)
        self.prediction = tf.matmul(outputs[0], self.weights['out']) + self.biases['out']

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs[0], labels=self.output_y)
        self.loss = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.accuracy = tf.equal(tf.argmax(self.output_y, 1), tf.argmax(outpus[0], 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epoch_num):
                for i in range(0, range(self.dataset_size), batch_size):
                    batch_x = self.dataset[i:i+batch_size]
                    batch_y = self.dataset[i+1:i+batch_size+1]
                    feed = {self.input_x:batch_x, self.output_y:batch_y}
                    sess.run(self.optimizer, feed_dict=feed)

                    if ((i/batch_size) % 500 == 0):
                        show_accu, show_loss = sess.run([self.accuracy, self.loss], feed)
                        print('epoch: %d, training step: %d, loss: %.3f, accuracy: %.3f'%(epoch, i, show_loss, show_accu))
