"""
用两个lstm来实现模型，一个负责分类NT种类（sigmoid），一个根据分类结果进行进一步的预测
"""


import tensorflow as tf
import pickle
import numpy as np
import time

import utils
from setting import Setting


base_setting = Setting()
sub_int_train_dir = base_setting.sub_int_train_dir
sub_int_valid_dir = base_setting.sub_int_valid_dir
model_save_dir = base_setting.lstm_model_save_dir
tensorboard_log_dir = base_setting.lstm_tb_log_dir
training_log_dir = base_setting.lstm_train_log_dir

num_subset_train_data = base_setting.num_sub_train_data
num_subset_test_data = base_setting.num_sub_test_data
show_every_n = base_setting.show_every_n
save_every_n = base_setting.save_every_n
num_terminal = base_setting.num_terminal


class BiRnnModel(object):
    def __init__(self,
                 num_ntoken, num_ttoken, is_training=True, saved_model=False,
                 batch_size=64,
                 n_embed_dim=64,
                 t_embed_dim=200,
                 num_hidden_units=256,
                 num_hidden_layers=2,
                 learning_rate=0.001,
                 num_epoches=12,
                 time_steps=50,
                 grad_clip=2,):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.n_embed_dim = n_embed_dim
        self.num_ntoken = num_ntoken
        self.num_ttoken = num_ttoken
        self.t_embed_dim = t_embed_dim
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        self.num_epoches = num_epoches
        self.grad_clip = grad_clip
        self.saved_model = saved_model

        if not is_training:
            self.batch_size = 1
            self.time_steps = 1

        self.build_model()

    def build_input(self):
        input_x = tf.placeholder(
            tf.int32, [None, None], name='input_x')
        target_y = tf.placeholder(
            tf.int32, [None, None], name='target_y')
        target_cate = tf.placeholder(
            tf.int32, [None, None], name='target_cate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return input_x, target_y, target_cate, keep_prob

    def build_input_embed(self, input_x):
        embed_matrix = tf.Variable(tf.truncated_normal(
            [self.num_ntoken, self.n_embed_dim]), name='n_embed_matrix')
        input_embedding = tf.nn.embedding_lookup(embed_matrix, input_x)
        return input_embedding

    def build_token_lstm(self, keep_prob):
        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell
        cell_list = [lstm_cell() for _ in range(self.num_hidden_layers)]
        cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        init_state = cells.zero_state(self.batch_size, dtype=tf.float32)
        return cells, init_state

    def build_category_lstm(self, keep_prob):
        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell
        cell_list = [lstm_cell() for _ in range(self.num_hidden_layers)]
        cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        init_state = cells.zero_state(self.batch_size, dtype=tf.float32)
        return cells, init_state

    def build_dynamic_output(self, lstm_input, token_cells, category_cells, token_init_state, category_init_state):
        token_lstm_output = tf.nn.dynamic_rnn(token_cells, lstm_input, token_init_state)
        category_lstm_output = tf.nn.dynamic_rnn(category_cells, lstm_input, category_init_state)
        return token_lstm_output, category_lstm_output

    def build_category_output(self, category_lstm_output):
        category_output_weight = tf.Variable(tf.truncated_normal([self.num_hidden_units, 1]))
        category_output_bias = tf.Variable(tf.constant(0.1))
        category_logits = tf.matmul(category_lstm_output, category_output_weight) + category_output_bias
        category_output = tf.nn.sigmoid(category_logits)
        category_output = tf.greater(category_output, 0.5)
        return category_output

    def build_token_output(self, category_logits, lstm_output):
        non_terminal = tf.boolean_mask(lstm_output,category_logits)
        inv_category_logits = tf.logical_not(category_logits)
        terminal = tf.boolean_mask(lstm_output, inv_category_logits)
        return non_terminal, terminal

    def build_non_terminal_prediction(self, nt_lstm_output):
        nt_output_weight = tf.Variable(tf.truncated_normal([self.num_hidden_units, self.num_ntoken]))
        nt_output_bias = tf.Variable(tf.constant(0.1, [self.num_ntoken]))
        nt_prediction = tf.matmul(nt_lstm_output, nt_output_weight) + nt_output_bias
        return nt_prediction

    def build_terminal_prediction(self, tt_lstm_output):
        tt_output_weight = tf.Variable(tf.truncated_normal([self.num_hidden_units, self.num_ttoken]))
        tt_output_bias = tf.Variable(tf.constant(0.1, [self.num_ttoken]))
        tt_prediction = tf.matmul(tt_lstm_output, tt_output_weight) + tt_output_bias
        return tt_prediction

    def build_category_loss(self, category_output, target_cate):
        pass

    def build_nt_loss(self):
        pass

    def build_category_optimizer(self, category_loss):
        pass

    def build_token_optimizer(self):
        pass

    def build_accuracy(self):
        pass

    def build_model(self):
        self.input_x, self.target_y, self.target_cate, self.keep_prob = self.build_input()
        input_embedding = self.build_input_embed(self.input_x)
        self.token_cells, self.token_init_state = self.build_token_lstm(self.keep_prob)
        self.category_cells, self.category_init_state = self.build_category_lstm(self.keep_prob)
        token_lstm_output, category_lstm_output = self.build_dynamic_output(
            self.token_cells, self.category_cells, self.token_init_state, self.category_init_state)
        category_output = self.build_category_output(category_lstm_output)
        self.category_loss = self.build_category_loss(category_output, self.target_cate)
        self.category_optimizer = self.build_category_optimizer(self.category_loss)
        non_terminal_output, terminal_output = self.build_token_output(category_output, token_lstm_output)

    def get_subset_data(self):
        for i in range(1, num_subset_train_data + 1):
            data_path = sub_int_train_dir + 'int_part{}.json'.format(i)
            with open(data_path, 'rb') as file:
                data = pickle.load(file)
                yield data

    def get_batch(self, data_seq):
        data_seq = np.array(data_seq)  # 是否可以注释掉节省时间
        total_length = self.time_steps * self.batch_size
        n_batches = len(data_seq) // total_length
        data_seq = data_seq[:total_length * n_batches]
        nt_x = data_seq[:, 0]
        tt_x = data_seq[:, 1]
        nt_y = np.zeros_like(nt_x)
        tt_y = np.zeros_like(tt_x)
        nt_y[:-1], nt_y[-1] = nt_x[1:], nt_x[0]
        tt_y[:-1], tt_y[-1] = tt_x[1:], tt_x[0]

        nt_x = nt_x.reshape((self.batch_size, -1))
        tt_x = tt_x.reshape((self.batch_size, -1))
        nt_y = nt_y.reshape((self.batch_size, -1))
        tt_y = tt_y.reshape((self.batch_size, -1))
        data_seq = data_seq.reshape((self.batch_size, -1))
        for n in range(0, data_seq.shape[1], self.time_steps):
            batch_nt_x = nt_x[:, n:n + self.time_steps]
            batch_tt_x = tt_x[:, n:n + self.time_steps]
            batch_nt_y = nt_y[:, n:n + self.time_steps]
            batch_tt_y = tt_y[:, n:n + self.time_steps]
            if batch_nt_x.shape[1] == 0:
                break
            yield batch_nt_x, batch_nt_y, batch_tt_x, batch_tt_y

    def train(self):
        self.print_and_log('model training...')
        saver = tf.train.Saver()
        session = tf.Session()
        tb_writer = tf.summary.FileWriter(tensorboard_log_dir, session.graph)
        global_step = 0
        session.run(tf.global_variables_initializer())

        for epoch in range(self.num_epoches):
            epoch_start_time = time.time()
            batch_step = 0
            loss_per_epoch = 0.0
            n_accu_per_epoch = 0.0
            t_accu_per_epoch = 0.0

            subset_generator = self.get_subset_data()
            for data in subset_generator:
                batch_generator = self.get_batch(data)
                for b_x, b_y, b_cate in batch_generator:
                    batch_step += 1
                    global_step += 1
                    feed = {self.input_x: b_x,
                            self.target_y: b_y,
                            self.target_cate: b_cate,
                            self.keep_prob: 0.5,}
                    batch_start_time = time.time()
                    show_loss, show_n_accu, show_t_accu, _, summary_str = session.run(
                        [self.loss, self.n_accu, self.t_accu, self.optimizer, self.merged_op], feed_dict=feed)

                    tb_writer.add_summary(summary_str, global_step)
                    tb_writer.flush()

                    loss_per_epoch += show_loss
                    n_accu_per_epoch += show_n_accu
                    t_accu_per_epoch += show_t_accu
                    batch_end_time = time.time()

                    if global_step % show_every_n == 0:
                        log_info = 'epoch:{}/{}  '.format(epoch + 1, self.num_epoches) + \
                            'global_step:{}  '.format(global_step) + \
                            'loss:{:.2f}  '.format(show_loss) + \
                            'nt_accu:{:.2f}%  '.format(show_n_accu * 100) + \
                            'tt_accu:{:.2f}%  '.format(show_t_accu * 100) + \
                            'time cost per batch: {:.2f}/s'.format(
                            batch_end_time - batch_start_time)
                        self.print_and_log(log_info)

                    if self.saved_model and global_step % save_every_n == 0:
                        saver.save(session, model_save_dir + 'e{}_b{}.ckpt'.format(epoch, batch_step))
            epoch_end_time = time.time()
            epoch_cost_time = epoch_end_time - epoch_start_time
            epoch_log = 'EPOCH:{}/{}  '.format(epoch + 1, self.num_epoches) + \
                        'time cost this epoch:{:.2f}/s  '.format(epoch_cost_time) + \
                        'epoch average loss:{:.2f}  '.format(loss_per_epoch / batch_step) + \
                        'epoch average nt_accu:{:.2f}%  '.format(100*n_accu_per_epoch / batch_step) + \
                        'epoch average tt_accu:{:.2f}%  '.format(100*t_accu_per_epoch / batch_step)
            self.print_and_log(epoch_log)

        if self.saved_model:
            saver.save(session, model_save_dir + 'lastest_model.ckpt')
        self.print_and_log('model training finished...')
        session.close()

    def valid(self, session, epoch, global_step):
        valid_dir = sub_int_valid_dir + 'int_part1.json'
        with open(valid_dir, 'rb') as f:
            valid_data = pickle.load(f)
        batch_generator = self.get_batch(valid_data)
        valid_step = 0
        valid_n_accuracy = 0.0
        valid_t_accuracy = 0.0
        valid_start_time = time.time()
        for b_nt_x, b_nt_y, b_t_x, b_t_y in batch_generator:
            valid_step += 1
            feed = {self.t_input: b_t_x,
                    self.n_input: b_nt_x,
                    self.n_target: b_nt_y,
                    self.t_target: b_t_y,
                    self.keep_prob: 0.5,
                    self.global_step: global_step}
            n_accuracy, t_accuracy = session.run([self.n_accu, self.t_accu], feed)
            valid_n_accuracy += n_accuracy
            valid_t_accuracy += t_accuracy

        valid_n_accuracy /= valid_step
        valid_t_accuracy /= valid_step
        valid_end_time = time.time()
        valid_log = "VALID epoch:{}/{}  ".format(epoch, self.num_epoches) + \
                    "global step:{}  ".format(global_step) + \
                    "valid_nt_accu:{:.2f}%  ".format(valid_n_accuracy * 100) + \
                    "valid_tt_accu:{:.2f}%  ".format(valid_t_accuracy * 100) + \
                    "valid time cost:{:.2f}s".format(valid_end_time - valid_start_time)
        self.print_and_log(valid_log)

    def print_and_log(self, info):
        try:
            self.log_file.write(info)
            self.log_file.write('\n')
        except BaseException:
            self.log_file = open(training_log_dir, 'w')
            self.log_file.write(info)
            self.log_file.write('\n')
        print(info)




if __name__ == '__main__':
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = utils.load_dict_parameter()
    n_ntoken = len(nt_int_to_token)
    n_ttoken = len(tt_int_to_token)
    model = BiRnnModel(n_ntoken, n_ttoken)
    model.train()
