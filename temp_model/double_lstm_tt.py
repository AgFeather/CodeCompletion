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

model_save_dir = base_setting.lstm_model_save_dir + 'tt_model/'
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
                 num_ntoken, num_ttoken, category,
                 is_training=True,
                 saved_model=True,
                 batch_size=50,
                 n_embed_dim=1500,
                 t_embed_dim=1500,
                 num_hidden_units=1500,
                 learning_rate=0.001,
                 num_epoches=4,
                 time_steps=50,
                 grad_clip=5,):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.n_embed_dim = n_embed_dim
        self.num_ntoken = num_ntoken
        self.num_ttoken = num_ttoken
        self.t_embed_dim = t_embed_dim
        self.num_hidden_units = num_hidden_units
        self.learning_rate = learning_rate
        self.num_epoches = num_epoches
        self.grad_clip = grad_clip
        self.saved_model = saved_model
        self.is_training = is_training
        self.category = category

        if not self.is_training:
            self.batch_size = 1
            self.time_steps = 1

        self.build_model()

    def build_input(self):
        n_input = tf.placeholder(
            tf.int32, [None, None], name='n_input')
        t_input = tf.placeholder(
            tf.int32, [None, None], name='t_input')
        target = tf.placeholder(
            tf.int32, [None, None], name='target')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return n_input, t_input, target, keep_prob

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
        cell = lstm_cell()
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        return cell, init_state

    def build_dynamic_rnn(self, cells, lstm_input, lstm_state):
        lstm_output, final_state = tf.nn.dynamic_rnn(
            cells, lstm_input, initial_state=lstm_state)
        # 将lstm_output的形状由[batch_size, time_steps, n_units] 转换为 [batch_size*time_steps, n_units]
        lstm_output = tf.concat(lstm_output, axis=1)
        lstm_output = tf.reshape(lstm_output, [-1, self.num_hidden_units])
        return lstm_output, final_state

    def build_output(self, lstm_output):
        if self.category == 'ntoken':
            weight = tf.Variable(tf.truncated_normal(
                [self.num_hidden_units, self.num_ntoken], stddev=0.1))
            bias = tf.Variable(tf.zeros(self.num_ntoken))
        else:
            weight = tf.Variable(tf.truncated_normal(
                [self.num_hidden_units, self.num_ttoken], stddev=0.1))
            bias = tf.Variable(tf.zeros(self.num_ttoken))
        logits = tf.matmul(lstm_output, weight) + bias
        output = tf.nn.softmax(logits=logits, name='nonterminal_output')
        return logits, output

    def build_loss(self, logits, targets):
        n_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=targets)
        n_loss = tf.reduce_mean(n_loss)
        return n_loss

    def bulid_accuracy(self, output, target):
        equal = tf.equal(tf.argmax(output, axis=1), tf.argmax(target, axis=1))
        accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
        return accuracy

    def bulid_optimizer(self, loss):
        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradient_pair = optimizer.compute_gradients(loss)
        clip_gradient_pair = []
        for grad, var in gradient_pair:
            grad = tf.clip_by_value(grad, -self.grad_clip, self.grad_clip)
            clip_gradient_pair.append((grad, var))
        optimizer = optimizer.apply_gradients(clip_gradient_pair)
        return optimizer

    def bulid_onehot_target(self, target):
        if self.category =='ntoken':
            onehot_target = tf.one_hot(target, self.num_ntoken)
            shape = (self.batch_size * self.time_steps, self.num_ntoken)
        else:
            onehot_target = tf.one_hot(target, self.num_ttoken)
            shape = (self.batch_size * self.time_steps, self.num_ttoken)
        onehot_target = tf.reshape(onehot_target, shape)
        return onehot_target

    def build_summary(self, summary_dict):
        for key, value in summary_dict.items():
            tf.summary.scalar(key, value)
        merged_op = tf.summary.merge_all()
        return merged_op

    def build_model(self):
        tf.reset_default_graph()
        self.n_input, self.t_input, self.target, self.keep_prob = self.build_input()
        n_input_embedding, t_input_embedding = self.build_input_embed(
            self.n_input, self.t_input)
        onehot_target = self.bulid_onehot_target(self.target)
        lstm_input = tf.add(n_input_embedding, t_input_embedding)
        cells, self.init_state = self.build_rnn(self.keep_prob)
        self.lstm_state = self.init_state
        lstm_output, self.final_state = self.build_dynamic_rnn(cells, lstm_input, self.lstm_state)

        logits, self.output = self.build_output(lstm_output)

        self.loss = self.build_loss(logits, onehot_target)

        self.accu = self.bulid_accuracy(self.output, onehot_target)
        self.optimizer = self.bulid_optimizer(self.loss)

        summary_dict = {self.category+' loss': self.loss, self.category+'accuracy': self.accu,}
        self.merged_op = self.build_summary(summary_dict)
        print('lstm model has been created...')


class DoubleLstmModel():
    def __init__(self, num_ntoken, num_ttoken,
                 batch_size=50,
                 time_steps=50,
                 num_epochs=4):
        self.num_ntoken = num_ntoken
        self.num_ttoken = num_ttoken
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.num_epochs = num_epochs
        self.nt_graph = tf.Graph()
        self.tt_graph = tf.Graph()
        # with self.nt_graph.as_default():
        #     self.nt_model = RnnModel(
        #         self.num_ntoken, self.num_ttoken, num_hidden_units=500, category='ntoken')
        self.tt_model = RnnModel(
                self.num_ntoken, self.num_ttoken, num_hidden_units=1500, category='ttoken')



    def train(self):
        pass
        # nt_session = tf.Session(graph=self.nt_graph)
        # tt_session = tf.Session(graph=self.tt_graph)
        # nt_saver = tf.train.Saver()
        # tt_saver = tf.train.Saver()
        # self.generator = DataGenerator(self.batch_size, self.time_steps)
        # nt_tb_writer = tf.summary.FileWriter(tensorboard_log_dir, nt_session.graph)
        # tt_tb_writer = tf.summary.FileWriter(tensorboard_log_dir, tt_session.graph)
        # global_step = 0
        # nt_session.run(tf.global_variables_initializer())
        # tt_session.run(tf.global_variables_initializer())
        # for epoch in range(1, self.num_epochs+1):
        #     epoch_start_time = time.time()
        #     batch_step = 0
        #     loss_per_epoch = 0.0
        #     n_accu_per_epoch = 0.0
        #
        #     subset_generator = self.generator.get_subset_data()
        #     for data in subset_generator:
        #         batch_generator = self.generator.get_batch(data_seq=data)
        #         nt_lstm_state = nt_session.run(self.nt_model.init_state)
        #         tt_lstm_state = tt_session.run(self.tt_model.init_state)
        #         for b_nt_x, b_nt_y, b_tt_x, b_tt_y in batch_generator:
        #             batch_step += 1
        #             global_step += 1
        #             nt_feed = {self.nt_model.t_input: b_tt_x,
        #                     self.nt_model.n_input: b_nt_x,
        #                     self.nt_model.target: b_nt_y,
        #                     self.nt_model.keep_prob: 0.5,
        #                     self.nt_model.lstm_state:nt_lstm_state,
        #                     self.nt_model.global_step:global_step}
        #             n_loss, n_accu, n_summary_str = self.train_simple_model(nt_session, nt_feed)
        #             nt_tb_writer.add_summary(n_summary_str, global_step)
        #             nt_tb_writer.flush()
        #
        #             tt_feed = {self.tt_model.t_input: b_tt_x,
        #                     self.tt_model.n_input: b_nt_x,
        #                     self.tt_model.target: b_tt_y,
        #                     self.tt_model.keep_prob: 0.5,
        #                     self.tt_model.lstm_state:tt_lstm_state,
        #                     self.tt_model.global_step:global_step}
        #             t_loss, t_accu, t_summary_str = self.train_simple_model(tt_session, tt_feed)


    def train_simple_model(self,session, feed):
        pass
        # loss, accu, _, summary_str = \
        #     session.run([
        #         self.nt_model.loss,
        #         self.nt_model.accu,
        #         self.nt_model.optimizer,
        #         self.nt_model.merged_op], feed_dict=feed)
        # return loss, accu, summary_str

    def train_nt_model(self):
        pass
        # self.print_and_log('non-terminal model training...')
        # saver = tf.train.Saver()
        # session = tf.Session()
        # self.generator = DataGenerator(self.batch_size, self.time_steps)
        # tb_writer = tf.summary.FileWriter(tensorboard_log_dir, session.graph)
        # global_step = 0
        # session.run(tf.global_variables_initializer())
        #
        # for epoch in range(1, self.num_epochs+1):
        #     epoch_start_time = time.time()
        #     batch_step = 0
        #     loss_per_epoch = 0.0
        #     n_accu_per_epoch = 0.0
        #
        #     subset_generator = self.generator.get_subset_data()
        #     for data in subset_generator:
        #         batch_generator = self.generator.get_batch(data_seq=data)
        #         nt_lstm_state = session.run(self.nt_model.init_state)
        #         for b_nt_x, b_nt_y, b_tt_x, b_tt_y in batch_generator:
        #             batch_step += 1
        #             global_step += 1
        #             nt_feed = {self.nt_model.t_input: b_tt_x,
        #                     self.nt_model.n_input: b_nt_x,
        #                     self.nt_model.target: b_nt_y,
        #                     self.nt_model.keep_prob: 0.5,
        #                     self.nt_model.lstm_state:nt_lstm_state,
        #                     self.nt_model.global_step:global_step}
        #
        #             batch_start_time = time.time()
        #             n_loss, n_accu, _, summary_str = \
        #                 session.run([
        #                     self.nt_model.loss,
        #                     self.nt_model.accu,
        #                     self.nt_model.optimizer,
        #                     self.nt_model.merged_op], feed_dict=nt_feed)
        #
        #             tb_writer.add_summary(summary_str, global_step)
        #             tb_writer.flush()
        #
        #             loss_per_epoch += n_loss
        #             n_accu_per_epoch += n_accu
        #             batch_end_time = time.time()
        #
        #             if global_step % show_every_n == 0:
        #                 log_info = 'epoch:{}/{}  '.format(epoch, self.num_epochs) + \
        #                            'global_step:{}  '.format(global_step) + \
        #                            'n_loss:{:.2f}  '.format(n_loss) + \
        #                            'nt_accu:{:.2f}%  '.format(n_accu * 100) + \
        #                            'time cost per batch:{:.2f}/s'.format(batch_end_time - batch_start_time)
        #                 self.print_and_log(log_info)
        #
        #             if global_step % valid_every_n == 0:
        #                 self.valid(session, epoch, global_step)
        #
        #             if self.nt_model.saved_model and global_step % save_every_n == 0:
        #                 saver.save(session, model_save_dir + 'e{}_b{}.ckpt'.format(epoch, batch_step))
        #                 print('model saved: epoch:{} global_step:{}'.format(epoch, global_step))
        #     epoch_end_time = time.time()
        #     epoch_cost_time = epoch_end_time - epoch_start_time
        #     epoch_log = 'EPOCH:{}/{}  '.format(epoch, self.num_epochs) + \
        #                 'time cost this epoch:{:.2f}/s  '.format(epoch_cost_time) + \
        #                 'epoch average loss:{:.2f}  '.format(loss_per_epoch / batch_step) + \
        #                 'epoch average nt_accu:{:.2f}%  '.format(100*n_accu_per_epoch / batch_step)
        #     self.print_and_log(epoch_log)
        #
        # if self.nt_model.saved_model:
        #     saver.save(session, model_save_dir + 'lastest_model.ckpt')
        # self.print_and_log('model training finished...')
        # session.close()

    def train_tt_model(self):
        self.print_and_log('terminal model training...')
        saver = tf.train.Saver()
        session = tf.Session()
        self.generator = DataGenerator(self.batch_size, self.time_steps)
        tb_writer = tf.summary.FileWriter(tensorboard_log_dir, session.graph)
        global_step = 0
        session.run(tf.global_variables_initializer())

        for epoch in range(1, self.num_epochs+1):
            epoch_start_time = time.time()
            batch_step = 0
            loss_per_epoch = 0.0
            t_accu_per_epoch = 0.0

            subset_generator = self.generator.get_subset_data()
            for data in subset_generator:
                batch_generator = self.generator.get_batch(data_seq=data)
                tt_lstm_state = session.run(self.tt_model.init_state)
                for b_nt_x, b_nt_y, b_tt_x, b_tt_y in batch_generator:
                    batch_step += 1
                    global_step += 1
                    tt_feed = {self.tt_model.t_input: b_tt_x,
                            self.tt_model.n_input: b_nt_x,
                            self.tt_model.target: b_nt_y,
                            self.tt_model.keep_prob: 0.5,
                            self.tt_model.lstm_state:tt_lstm_state,
                            self.tt_model.global_step:global_step}

                    batch_start_time = time.time()
                    t_loss, t_accu, _, summary_str = \
                        session.run([
                            self.tt_model.loss,
                            self.tt_model.accu,
                            self.tt_model.optimizer,
                            self.tt_model.merged_op], feed_dict=tt_feed)

                    tb_writer.add_summary(summary_str, global_step)
                    tb_writer.flush()

                    loss_per_epoch += t_loss
                    t_accu_per_epoch += t_accu
                    batch_end_time = time.time()

                    if global_step % show_every_n == 0:
                        log_info = 'epoch:{}/{}  '.format(epoch, self.num_epochs) + \
                                   'global_step:{}  '.format(global_step) + \
                                   't_loss:{:.2f}  '.format(t_loss) + \
                                   'tt_accu:{:.2f}%  '.format(t_accu * 100) + \
                                   'time cost per batch:{:.2f}/s'.format(batch_end_time - batch_start_time)
                        self.print_and_log(log_info)

                    if global_step % valid_every_n == 0:
                        self.valid_tt_model(session, epoch, global_step)

                    if self.tt_model.saved_model and global_step % save_every_n == 0:
                        saver.save(session, model_save_dir + 'e{}_b{}.ckpt'.format(epoch, batch_step))
                        print('model saved: epoch:{} global_step:{}'.format(epoch, global_step))
            epoch_end_time = time.time()
            epoch_cost_time = epoch_end_time - epoch_start_time
            epoch_log = 'EPOCH:{}/{}  '.format(epoch, self.num_epochs) + \
                        'time cost this epoch:{:.2f}/s  '.format(epoch_cost_time) + \
                        'epoch average loss:{:.2f}  '.format(loss_per_epoch / batch_step) + \
                        'epoch average tt_accu:{:.2f}%  '.format(100*t_accu_per_epoch / batch_step)
            self.print_and_log(epoch_log)

        if self.tt_model.saved_model:
            saver.save(session, model_save_dir + 'lastest_model.ckpt')
        self.print_and_log('model training finished...')
        session.close()

    def valid_tt_model(self, session, epoch, global_step):
        valid_dir = sub_int_valid_dir + 'int_part1.json'
        with open(valid_dir, 'rb') as f:
            valid_data = pickle.load(f)
        batch_generator = self.generator.get_batch(valid_data)
        valid_step = 0
        valid_n_accuracy = 0.0
        valid_times = 200
        valid_start_time = time.time()
        lstm_state = session.run(self.tt_model.init_state)
        for b_nt_x, b_nt_y, b_tt_x, b_tt_y in batch_generator:
            valid_step += 1
            feed = {self.tt_model.t_input: b_tt_x,
                       self.tt_model.n_input: b_nt_x,
                       self.tt_model.target: b_nt_y,
                       self.tt_model.keep_prob: 0.5,
                       self.tt_model.lstm_state: lstm_state,
                       self.tt_model.global_step: global_step}
            n_accuracy, lstm_state = session.run([self.tt_model.accu, self.tt_model.final_state], feed)
            valid_n_accuracy += n_accuracy
            if valid_step >= valid_times:
                break

        valid_n_accuracy /= valid_step
        valid_end_time = time.time()
        valid_log = "VALID epoch:{}/{}  ".format(epoch, self.num_epochs) + \
                    "global step:{}  ".format(global_step) + \
                    "valid_tt_accu:{:.2f}%  ".format(valid_n_accuracy * 100) + \
                    "valid time cost:{:.2f}s".format(valid_end_time - valid_start_time)
        self.print_and_log(valid_log)

    def valid(self, session, epoch, global_step):
        pass
        # valid_dir = sub_int_valid_dir + 'int_part1.json'
        # with open(valid_dir, 'rb') as f:
        #     valid_data = pickle.load(f)
        # batch_generator = self.generator.get_batch(valid_data)
        # valid_step = 0
        # valid_n_accuracy = 0.0
        # valid_times = 1
        # valid_start_time = time.time()
        # lstm_state = session.run(self.nt_model.init_state)
        # for b_nt_x, b_nt_y, b_tt_x, b_tt_y in batch_generator:
        #     valid_step += 1
        #     feed = {self.nt_model.t_input: b_tt_x,
        #                self.nt_model.n_input: b_nt_x,
        #                self.nt_model.target: b_nt_y,
        #                self.nt_model.keep_prob: 0.5,
        #                self.nt_model.lstm_state: lstm_state,
        #                self.nt_model.global_step: global_step}
        #     n_accuracy, lstm_state = session.run([self.nt_model.accu, self.nt_model.final_state], feed)
        #     valid_n_accuracy += n_accuracy
        #     if valid_step >= valid_times:
        #         break
        #
        # valid_n_accuracy /= valid_step
        # valid_end_time = time.time()
        # valid_log = "VALID epoch:{}/{}  ".format(epoch, self.num_epochs) + \
        #             "global step:{}  ".format(global_step) + \
        #             "valid_nt_accu:{:.2f}%  ".format(valid_n_accuracy * 100) + \
        #             "valid time cost:{:.2f}s".format(valid_end_time - valid_start_time)
        # self.print_and_log(valid_log)

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
    model = DoubleLstmModel(n_ntoken, n_ttoken)
    model.train_tt_model()
