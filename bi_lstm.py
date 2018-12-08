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
subset_int_data_dir = base_setting.sub_int_train_dir
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
            tf.int32, [self.batch_size, self.time_steps], name='input_x')
        target_y = tf.placeholder(
            tf.int32, [self.batch_size, self.time_steps], name='target_y')
        target_cate = tf.placeholder(
            tf.int32, [self.batch_size, self.time_steps], name='target_cate')
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

    def train(self):
        pass





if __name__ == '__main__':
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = utils.load_dict_parameter()
    n_ntoken = len(nt_int_to_token)
    n_ttoken = len(tt_int_to_token)
    model = BiRnnModel(n_ntoken, n_ttoken)
    model.train()
