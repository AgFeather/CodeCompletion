import tensorflow as tf
import numpy as np
import pickle
import data_utils
import time
import json
import os

'''
Using MultiRNN to pridect token. with LSTM cell
'''



processed_data_path = 'processed_data/rnn_train_data.p'
data_parameter_path = 'processed_data/rnn_train_parameter.p'
tensorboard_log_path = 'logs/MultiRNN'

train_dir = 'dataset/programs_800'
test_dir = 'dataset/programs_200'
checkpoint_dir = 'checkpoints/'


show_every_n = 100
save_every_n = 1000



def load_tokens(train_flag=True, is_simplify=True):
    if train_flag:
        token_dir = train_dir
    else:
        token_dir = test_dir
    token_list = []
    for f in os.listdir(token_dir):
        file_path = os.path.join(token_dir, f)
        if os.path.isfile(file_path) and f.endswith('_tokens.json'):
            token_seq = json.load(open(file_path, encoding='utf-8'))
            token_list.extend(token_seq)
    string_token_list = []
    for token in token_list:
        if is_simplify:
            data_utils.simplify_token(token)
        string_token = data_utils.token_to_string(token)
        string_token_list.append(string_token)
    token_set = list(set(string_token_list))
    string2int = {c:i for i,c in enumerate(token_set)}
    int2string = {i:c for i,c in enumerate(token_set)}
    int_token_list = [string2int[c] for c in string_token_list]
    pickle.dump((int_token_list), open(processed_data_path, 'wb'))
    pickle.dump((string2int, int2string, token_set), open(data_parameter_path, 'wb'))


'''
Using MultiRNN to pridect token. with LSTM cell
'''


class LSTM_Model(object):
    def __init__(self,
                 token_set, time_steps=100,
                 batch_size=64,
                 num_layers=2,
                 n_units=128,
                 learning_rate=0.003,
                 grad_clip=5,
                 keep_prob=0.5,
                 num_epoches=20,
                 is_training=True):

        if is_training:
            self.time_steps = time_steps
            self.batch_size = batch_size
        else:
            self.time_steps = 1
            self.batch_size = 1

        self.token_set = token_set
        self.num_classes = len(self.token_set)
        self.num_layers = num_layers
        self.n_units = n_units
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.keep_prob = keep_prob
        self.num_epoches = num_epoches

        self.bulid_model()

    def get_batch(self, data_seq, n_seq, n_steps):
        '''
        :param n_seq: 一个batch中序列的个数
        :param n_steps: 单个序列中包含字符的个数
        '''
        data_seq = np.array(data_seq)
        batch_size = n_steps * n_seq
        n_batches = len(data_seq) // batch_size
        data_seq = data_seq[:batch_size * n_batches]  # 仅保留完整的batch，舍去末尾
        data_seq = data_seq.reshape((n_seq, -1))
        for n in range(0, data_seq.shape[1], n_steps):
            x = data_seq[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

    def build_input(self):
        input_x = tf.placeholder(tf.int32, [self.batch_size, self.time_steps], name='input_x')
        target_y = tf.placeholder(tf.int32, [self.batch_size, self.time_steps], name='target_y')
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
        one_hot_y = tf.one_hot(targets, self.num_classes)
        one_hot_y = tf.reshape(one_hot_y, logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
        loss = tf.reduce_mean(loss)
        return loss

    def bulid_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradient_pairs = optimizer.compute_gradients(loss)
        clip_gradient_pairs = []
        for grad, var in gradient_pairs:
            grad = tf.clip_by_value(grad, -2, 2)
            clip_gradient_pairs.append((grad, var))
        optimizer = optimizer.apply_gradients(clip_gradient_pairs)
        return optimizer

    def build_accuracy(self, logits, targets):
        self.show_logits = tf.argmax(logits, axis=1)
        show_targets = tf.one_hot(targets, self.num_classes)
        show_targets = tf.reshape(show_targets, logits.get_shape())
        self.show_targets = tf.argmax(show_targets, axis=1)
        self.aaa = tf.equal(self.show_logits, self.show_targets)
        accu = tf.cast(self.aaa, tf.float32)
        accu = tf.reduce_mean(accu)
        return accu

    def bulid_model(self):
        tf.reset_default_graph()
        self.input_x, self.target_y, self.keep_prob = self.build_input()
        self.cell, self.init_state = self.bulid_lstm(self.keep_prob)
        one_hot_x = tf.one_hot(self.input_x, self.num_classes)
        # print(one_hot_x.get_shape()) # (64, 100, 86)
        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
            self.cell, one_hot_x, initial_state=self.init_state)
        # print(1, lstm_outputs.get_shape()) # (64, 100, 128)
        self.softmax_output, logits = self.bulid_output(lstm_outputs)
        # print(self.softmax_output.get_shape()) # (6400, 86)
        # print(logits.get_shape()) #(6400, 86)
        self.loss = self.bulid_loss(logits, self.target_y)
        self.accuracy = self.build_accuracy(self.softmax_output, self.target_y)
        self.optimizer = self.bulid_optimizer(self.loss)
        
        tf.summary.scalar('train_loss', self.loss)
        tf.summary.scalar('train_accu', self.accuracy)
        self.merged_op = tf.summary.merge_all()

    def train(self, data, string2int, int2string):
        print('training begin...')
        self.string2int = string2int
        self.int2string = int2string
        saver = tf.train.Saver(max_to_keep=100)
        keep_prob = 0.5
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(tensorboard_log_path, sess.graph)
            global_step = 0
            for epoch in range(self.num_epoches):
                new_state = sess.run(self.init_state)
                batch_generator = self.get_batch(data, self.batch_size, self.time_steps)
                batch_step = 0
                start_time = time.time()
                for x, y in batch_generator:
                    global_step += 1
                    batch_step += 1
                    feed = {self.input_x: x,
                            self.target_y: y,
                            self.keep_prob: keep_prob,
                            self.init_state: new_state}
                    show_accu, show_loss, new_state, summary_str, _= sess.run(
                        [self.accuracy, self.loss, self.final_state, self.merged_op, self.optimizer], feed_dict=feed)
                    end_time = time.time()
                    writer.add_summary(summary_str, global_step)
                    writer.flush()
                    if global_step % show_every_n == 0:
                        print('epoch: {}/{}..'.format(epoch + 1, self.num_epoches),
                              'global_step: {}..'.format(global_step),
                              'train_loss: {:.2f}..'.format(show_loss),
                              'train_accuracy: {:.2f}..'.format(show_accu),
                              'time cost in per_batch: {:.2f}..'.format(end_time - start_time))

                    if global_step % save_every_n == 0:
                        saver.save(sess, 'checkpoints/epoch{}_batch_step{}'.format(epoch, batch_step))
            saver.save(sess, 'checkpoints/last_check')


class TestModel(object):
    def __init__(self, token_set, string2int, int2string):
        self.model = LSTM_Model(token_set, is_training=False)
        self.string2int = string2int
        self.int2string = int2string
        self.last_chackpoints = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self.last_chackpoints)

    # query test
    def query_test(self, prefix, suffix):
        '''
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        '''
        # saver = tf.train.Saver()
        # with tf.Session() as sess:
        #     saver.restore(sess, self.last_chackpoints)
        new_state = self.sess.run(self.model.init_state)
        prediction = None
        for i, token in enumerate(prefix):
            x = np.zeros((1, 1), dtype=np.int32)
            x[0, 0] = token
            feed = {self.model.input_x: x,
                    self.model.keep_prob: 1.,
                    self.model.init_state: new_state}
            prediction, new_state = self.sess.run(
                [self.model.softmax_output, self.model.final_state], feed_dict=feed)
        prediction = self.int2string[np.argmax(prediction)]
        return prediction

    def test(self, query_test_data):
        print('test step is beginning..')
        start_time = time.time()
        correct = 0.0
        for token_sequence in query_test_data:
            prefix, expection, suffix = data_utils.create_hole(token_sequence)
            prefix = self.token_to_int(prefix)
            prediction = self.query_test(prefix, suffix)
            prediction = data_utils.string_to_token(prediction)
            if data_utils.token_equals([prediction], expection):
                correct += 1
        accuracy = correct / len(query_test_data)
        end_time =time.time()
        print('test finished, time cost:{:.2f}..'.format(end_time-start_time))

        return accuracy

    def token_to_int(self, token_seq):
        int_token_seq = []
        for token in token_seq:
            int_token = self.string2int[data_utils.token_to_string(token)]
            int_token_seq.append(int_token)
        return int_token_seq



if __name__ == '__main__':
    # train_data = data_utils.load_data_with_pickle(processed_data_path)
    string2int, int2string, token_set = data_utils.load_data_with_pickle(data_parameter_path)

    # model = LSTM_Model(token_set)
    # model.train(train_data, string2int, int2string)

    test_data = data_utils.load_data_with_file(test_dir)
    accuracy = 0.0
    test_epoches = 1
    test_model = TestModel(token_set, string2int, int2string)
    for epoch in range(test_epoches):
        accuracy += test_model.test(test_data)
    accuracy /= test_epoches
    print(accuracy) # 0.615  -> 0.68