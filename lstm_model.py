import tensorflow as tf
import pickle
import numpy as np
import time

import utils
from setting import Setting
from data_generator import DataGenerator


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
valid_every_n = base_setting.valid_every_n
num_terminal = base_setting.num_terminal


class RnnModel(object):
    def __init__(self,
                 num_ntoken, num_ttoken, is_training=True, saved_model=False, kernel='LSTM',
                 batch_size=64,
                 n_embed_dim=1000,
                 t_embed_dim=1000,
                 num_hidden_units=1000,
                 num_hidden_layers=2,
                 learning_rate=0.001,
                 num_epoches=10,
                 time_steps=50,
                 grad_clip=5,):
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
        self.kernel = kernel
        self.is_training = is_training

        if not self.is_training:
            self.batch_size = 1
            self.time_steps = 1

        self.build_model()

    def build_input(self):
        n_input = tf.placeholder(
            tf.int32, [None, None], name='n_input')
        t_input = tf.placeholder(
            tf.int32, [None, None], name='t_input')
        n_target = tf.placeholder(
            tf.int32, [None, None], name='n_target')
        t_target = tf.placeholder(
            tf.int32, [None, None], name='t_target')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return n_input, t_input, n_target, t_target, keep_prob

    def build_input_embed(self, n_input, t_input):
        n_embed_matrix = tf.Variable(tf.truncated_normal(
            [self.num_ntoken, self.n_embed_dim]), name='n_embed_matrix')
        t_embed_matrix = tf.Variable(tf.truncated_normal(
            [self.num_ttoken, self.t_embed_dim]), name='t_embed_matrix')
        n_input_embedding = tf.nn.embedding_lookup(n_embed_matrix, n_input)
        t_input_embedding = tf.nn.embedding_lookup(t_embed_matrix, t_input)
        return n_input_embedding, t_input_embedding

    def build_rnn(self, keep_prob):
        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell
        def gru_cell():
            cell = tf.contrib.rnn.GRUCell(self.num_hidden_units)
        #    cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=50)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell
        if self.kernel == 'LSTM':
            cell_list = [lstm_cell() for _ in range(self.num_hidden_layers)]
        else:
            assert self.kernel == 'GRU'
            cell_list = [gru_cell() for _ in range(self.num_hidden_layers)]
        cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        init_state = cells.zero_state(self.batch_size, dtype=tf.float32)
        return cells, init_state

    def build_dynamic_rnn(self, cells, lstm_input, lstm_state):
        lstm_output, final_state = tf.nn.dynamic_rnn(
            cells, lstm_input, initial_state=lstm_state)
        return lstm_output, final_state

    def build_attention(self, inputs, attention_size):
        with tf.name_scope('attention'):
            w_omega = tf.Variable(tf.random_normal([self.num_hidden_units, attention_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)  # 不同维度tensor进行矩阵乘法
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')
        attention_output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        return attention_output, alphas

    def build_n_output(self, lstm_output):
        # 将output的形状由[batch, time_steps, n_units] 转换为 [batch*time_steps, n_units]
        seq_output = tf.concat(lstm_output, axis=1)
        seq_output = tf.reshape(seq_output, [-1, self.num_hidden_units])

        with tf.variable_scope('non_terminal_softmax'):
            nt_weight = tf.Variable(tf.truncated_normal(
                [self.num_hidden_units, self.num_ntoken], stddev=0.1))
            nt_bias = tf.Variable(tf.zeros(self.num_ntoken))

        nonterminal_logits = tf.matmul(seq_output, nt_weight) + nt_bias
        nonterminal_output = tf.nn.softmax(logits=nonterminal_logits, name='nonterminal_output')
        return nonterminal_logits, nonterminal_output

    def build_t_output(self, lstm_output):
        # 将lstm_output的形状由[batch_size, time_steps, n_units] 转换为
        # [batch_size*time_steps, n_units]
        seq_output = tf.concat(lstm_output, axis=1)
        seq_output = tf.reshape(seq_output, [-1, self.num_hidden_units])
        with tf.variable_scope('terminal_softmax'):
            t_weight = tf.Variable(tf.truncated_normal([self.num_hidden_units, self.num_ttoken], stddev=0.1))
            t_bias = tf.Variable(tf.zeros(self.num_ttoken))

        terminal_logits = tf.matmul(seq_output, t_weight) + t_bias
        termnial_output = tf.nn.softmax(
            logits=terminal_logits, name='terminal_output')
        return terminal_logits, termnial_output

    def build_loss(self, n_logits, n_target, t_logits, t_target):
        # todo: 使用负采样方法进行训练加快训练速度？
        n_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=n_logits, labels=n_target)
        t_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=t_logits, labels=t_target)
        loss = tf.add(n_loss, t_loss)
        loss = tf.reduce_mean(loss)
        n_loss = tf.reduce_mean(n_loss)
        t_loss = tf.reduce_mean(t_loss)
        return loss, n_loss, t_loss

    def bulid_accuracy(self, n_output, n_target, t_output, t_target):
        n_equal = tf.equal(
            tf.argmax(n_output, axis=1), tf.argmax(n_target, axis=1))
        t_equal = tf.equal(
            tf.argmax(t_output, axis=1), tf.argmax(t_target, axis=1))
        n_accuracy = tf.reduce_mean(tf.cast(n_equal, tf.float32))
        t_accuracy = tf.reduce_mean(tf.cast(t_equal, tf.float32))
        return n_accuracy, t_accuracy

    def bulid_optimizer(self, loss):
        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradient_pair = optimizer.compute_gradients(loss)
        clip_gradient_pair = []
        for grad, var in gradient_pair:
            grad = tf.clip_by_value(grad, -self.grad_clip, self.grad_clip)
            clip_gradient_pair.append((grad, var))
        optimizer = optimizer.apply_gradients(clip_gradient_pair)
        return optimizer

    def bulid_onehot_target(self, n_target, t_target):
        onehot_n_target = tf.one_hot(n_target, self.num_ntoken)
        n_shape = (self.batch_size * self.time_steps, self.num_ntoken)
        t_shape = (self.batch_size * self.time_steps, self.num_ttoken)
        onehot_n_target = tf.reshape(onehot_n_target, n_shape)
        onehot_t_target = tf.one_hot(t_target, self.num_ttoken)
        onehot_t_target = tf.reshape(onehot_t_target, t_shape)
        return onehot_n_target, onehot_t_target

    def build_summary(self):
        tf.summary.scalar('train_loss', self.loss)
        tf.summary.scalar('t_loss', self.t_loss)
        tf.summary.scalar('n_loss', self.n_loss)
        tf.summary.scalar('n_accuracy', self.n_accu)
        tf.summary.scalar('t_accuracy', self.t_accu)
        self.merged_op = tf.summary.merge_all()

    def build_model(self):
        tf.reset_default_graph()
        self.n_input, self.t_input, self.n_target, self.t_target, self.keep_prob = self.build_input()
        n_input_embedding, t_input_embedding = self.build_input_embed(
            self.n_input, self.t_input)
        lstm_input = tf.add(n_input_embedding, t_input_embedding)
        cells, self.init_state = self.build_rnn(self.keep_prob)
        self.lstm_state = self.init_state
        lstm_output, self.final_state = self.build_dynamic_rnn(cells, lstm_input, self.lstm_state)
        t_logits, self.t_output = self.build_t_output(lstm_output)
        n_logits, self.n_output = self. build_n_output(lstm_output)

        onehot_n_target, onehot_t_target = self.bulid_onehot_target(
            self.n_target, self.t_target)

        self.loss, self.n_loss, self.t_loss = self.build_loss(
            n_logits, onehot_n_target, t_logits, onehot_t_target)
        self.n_accu, self.t_accu = self.bulid_accuracy(
            self.n_output, onehot_n_target, self.t_output, onehot_t_target)
        self.optimizer = self.bulid_optimizer(self.loss)

        if self.is_training:
            self.build_summary()

        print('lstm model has been created...')

    def train(self):
        self.print_and_log('model training...')
        saver = tf.train.Saver()
        session = tf.Session()
        self.generator = DataGenerator(self.batch_size, self.time_steps)
        tb_writer = tf.summary.FileWriter(tensorboard_log_dir, session.graph)
        global_step = 0
        session.run(tf.global_variables_initializer())

        for epoch in range(1, self.num_epoches+1):
            epoch_start_time = time.time()
            batch_step = 0
            loss_per_epoch = 0.0
            n_accu_per_epoch = 0.0
            t_accu_per_epoch = 0.0

            subset_generator = self.generator.get_subset_data()
            for data in subset_generator:
                batch_generator = self.generator.get_batch(data_seq=data)
                lstm_state = session.run(self.init_state)
                for b_nt_x, b_nt_y, b_t_x, b_t_y in batch_generator:
                    batch_step += 1
                    global_step += 1
                    feed = {self.t_input: b_t_x,
                            self.n_input: b_nt_x,
                            self.n_target: b_nt_y,
                            self.t_target: b_t_y,
                            self.keep_prob: 0.5,
                            self.lstm_state:lstm_state,
                            self.global_step:global_step}
                    batch_start_time = time.time()
                    loss, n_loss, t_loss, n_accu, t_accu, _, summary_str = \
                        session.run([
                            self.loss,
                            self.n_loss,
                            self.t_loss,
                            self.n_accu,
                            self.t_accu,
                            self.optimizer,
                            self.merged_op], feed_dict=feed)

                    tb_writer.add_summary(summary_str, global_step)
                    tb_writer.flush()

                    loss_per_epoch += loss
                    n_accu_per_epoch += n_accu
                    t_accu_per_epoch += t_accu
                    batch_end_time = time.time()

                    if global_step % show_every_n == 0:
                        log_info = 'epoch:{}/{}  '.format(epoch, self.num_epoches) + \
                                   'global_step:{}  '.format(global_step) + \
                                   'loss:{:.2f}(n_loss:{:.2f} + t_loss:{:.2f})  '.format(loss, n_loss, t_loss) + \
                                   'nt_accu:{:.2f}%  '.format(n_accu * 100) + \
                                   'tt_accu:{:.2f}%  '.format(t_accu * 100) + \
                                   'time cost per batch:{:.2f}/s'.format(batch_end_time - batch_start_time)
                        self.print_and_log(log_info)

                    if global_step % valid_every_n == 0:
                        self.valid(session, epoch, global_step)

                    if self.saved_model and global_step % save_every_n == 0:
                        saver.save(session, model_save_dir + 'e{}_b{}.ckpt'.format(epoch, batch_step))
                        print('model saved: epoch:{} global_step:{}'.format(epoch, global_step))
            epoch_end_time = time.time()
            epoch_cost_time = epoch_end_time - epoch_start_time
            epoch_log = 'EPOCH:{}/{}  '.format(epoch, self.num_epoches) + \
                        'time cost this epoch:{:.2f}/s  '.format(epoch_cost_time) + \
                        'epoch average loss:{:.2f}  '.format(loss_per_epoch / batch_step) + \
                        'epoch average nt_accu:{:.2f}%  '.format(100*n_accu_per_epoch / batch_step) + \
                        'epoch average tt_accu:{:.2f}%  '.format(100*t_accu_per_epoch / batch_step) + '\n'
            self.print_and_log(epoch_log)

        if self.saved_model:
            saver.save(session, model_save_dir + 'lastest_model.ckpt')
        self.print_and_log('model training finished...')
        session.close()

    def valid(self, session, epoch, global_step):
        valid_dir = sub_int_valid_dir + 'int_part1.json'
        with open(valid_dir, 'rb') as f:
            valid_data = pickle.load(f)
        batch_generator = self.generator.get_batch(valid_data)
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
                    self.keep_prob: 1.0,
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
    model = RnnModel(n_ntoken, n_ttoken, saved_model=True)
    model.train()
