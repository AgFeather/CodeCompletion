import tensorflow as tf
import numpy as np
import pickle
import data_utils
import time


'''
Using MultiRNN to pridect token. with LSTM cell
'''



processed_data_path = 'processed_data/vec_train_data.p'
data_parameter_path = 'processed_data/train_parameter.p'
tensorboard_log_path = 'logs/MultiRNN'

test_dir = 'dataset/programs_200'


num_epoches = 1
show_every_n = 50
save_every_n = 200





class LSTM_Model(object):
    def __init__(self,
                 token_set, time_steps=100,
                 batch_size=64,
                 num_layers=2,
                 n_units=128,
                 learning_rate=0.003,
                 grad_clip=5,
                 keep_prob=0.5,
                 is_training=True):
        self.string2int, self.int2string, self.token_set = string2int, int2string, token_set
        if is_training:
            self.time_steps = time_steps
            self.batch_size = batch_size
        else:
            self.time_steps = 1 # todo: modify
            self.batch_size = 1
        self.num_classes = len(self.token_set)
        self.num_layers = num_layers
        self.n_units = n_units
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.keep_prob = keep_prob

        self.bulid_model()


    def get_batch(self, data_seq, n_seq, n_steps):
        '''

        :param n_seq: 一个batch中序列的个数
        :param n_steps: 单个序列中包含字符的个数
        :return:
        '''
        batch_size = n_steps * n_seq
        n_batches = len(data_seq) // batch_size
        data_seq = data_seq[:batch_size * n_batches] #仅保留完整的batch，舍去末尾

        data_seq = data_seq.reshape((n_seq, -1))

        for n in range(0, data_seq.shape[1], n_steps):
            x = data_seq[:, n:n+n_steps]
            y = np.zeros_like(x)
            y[:, -1], y[:, -1] = x[:, 1:], x[:,0]
            yield x, y


    def build_input(self):
        input_x = tf.placeholder(tf.float32, [self.batch_size, self.time_steps], name='input_x')
        target_y = tf.placeholder(tf.float32, [self.batch_size, self.time_steps], name='target_y')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        return input_x, target_y, keep_prob

    def bulid_lstm(self, keep_prob):
        cell_list = []
        for i in range(self.num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_units, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cell_list.append(cell)
        cells = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)
        init_state = cells.zero_state(self.batch_size, dtype=tf.float32)

        return cells, init_state

    def bulid_output(self, lstm_output):

        # 将lstm_output的形状由[batch_size, time_steps, n_units] 转换为 [batch_size*time_steps, n_units]
        seq_output = tf.concat(lstm_output, axis=1)
        seq_output = tf.reshape(seq_output, [-1, self.n_units])

        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([self.n_units, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))

        logits = tf.matmul(seq_output, softmax_w) + softmax_b
        softmax_output = tf.nn.softmax(logits=logits, name='softmax_output')

        return softmax_output, logits

    def bulid_loss(self, logits, targets):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, targets=targets)
        loss = tf.reduce_mean(loss)

        return loss

    def bulid_optimizer(self,loss):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.apply_gradients(zip(grads, tvars))

        return optimizer


    def bulid_model(self):
        tf.reset_default_graph()
        self.input_x, self.target_y, self.keep_prob = self.build_input()
        self.cell, self.init_state = self.bulid_lstm(self.keep_prob)

        one_hot_x = tf.one_hot(self.input_x, self.num_classes)

        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, one_hot_x, initial_state=self.init_state)

        self.softmax_output, logits = self.bulid_output(lstm_outputs)

        self.loss = self.bulid_loss(logits,self.target_y)

        self.optimizer = self.bulid_optimizer(self.loss)





    def train(self, data):
        print('training begin...')
        saver = tf.train.Saver(max_to_keep=100)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            global_step = 0
            for epoch in range(num_epoches):
                new_state = sess.run(self.init_state)
                batch_generator = self.get_batch(data, self.batch_size, self.time_steps)
                batch_step = 0
                start_time = time.time()
                for x, y in batch_generator:
                    global_step += 1
                    batch_step += 1
                    feed = {self.input_x:x,
                            self.target_y:y,
                            self.keep_prob:self.keep_prob,
                            self.init_state:new_state}
                    show_loss, new_state, _ = sess.run(
                        [self.loss, self.final_state, self.optimizer], feed_dict=feed)
                    end_time = time.time()
                    if global_step % show_every_n == 0:
                        print('epoch: {}/{}...'.format(epoch+1, num_epoches),
                              'global_step: {}...'.format(global_step),
                              ' train_loss: {:.4f}...'.format(show_loss),
                              'time cost in per_batch: {:.4f}...'.format(end_time-start_time))

                    if global_step % save_every_n == 0:
                        saver.save(sess, 'checkpoints/epoch{}_batch_step{}'.format(epoch, batch_step))

            saver.save(sess, 'checkpoints/last_check')






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

    model = LSTM_Model(token_set)

    # model train
    model.train(train_data)

    # model test
    query_test_data = data_utils.load_data_with_file(test_dir)
    test_accuracy = 0.0
    test_epoches = 10
    for i in range(test_epoches):
        accuracy = model.test_model(query_test_data)
        print('test epoch: %d, query test accuracy: %.3f' % (i, accuracy))
        test_accuracy += accuracy
    print('total test accuracy: %.3f' % (test_accuracy / test_epoches))