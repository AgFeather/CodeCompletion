import tensorflow as tf
import pickle
import time
import os

from setting import Setting
from data_generator import DataGenerator


base_setting = Setting()

model_save_dir = base_setting.lstm_model_save_dir
tensorboard_log_dir = base_setting.lstm_tb_log_dir
training_log_dir = base_setting.lstm_train_log_dir
valid_log_dir = base_setting.lstm_valid_log_dir

show_every_n = base_setting.show_every_n
save_every_n = base_setting.save_every_n
valid_every_n = base_setting.valid_every_n


class RnnModel(object):
    """A basic LSTM model for code completion"""
    def __init__(self,
                 num_ntoken, num_ttoken,
                 batch_size=50,
                 n_embed_dim=1500,
                 t_embed_dim=1500,
                 num_hidden_units=1500,
                 learning_rate=0.001,
                 num_epochs=5,
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
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip

        self.build_model()

    def build_input(self):
        """create input and target placeholder"""
        n_input = tf.placeholder(tf.int32, [None, None], name='n_input')
        t_input = tf.placeholder(tf.int32, [None, None], name='t_input')
        n_target = tf.placeholder(tf.int64, [None, None], name='n_target')
        t_target = tf.placeholder(tf.int64, [None, None], name='t_target')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return n_input, t_input, n_target, t_target, keep_prob

    def build_input_embed(self, n_input, t_input):
        """create input embedding matrix and return embedding vector"""
        n_embed_matrix = tf.Variable(tf.truncated_normal(
            [self.num_ntoken, self.n_embed_dim]), name='n_embed_matrix')
        t_embed_matrix = tf.Variable(tf.truncated_normal(
            [self.num_ttoken, self.t_embed_dim]), name='t_embed_matrix')
        n_input_embedding = tf.nn.embedding_lookup(n_embed_matrix, n_input)
        t_input_embedding = tf.nn.embedding_lookup(t_embed_matrix, t_input)
        return n_input_embedding, t_input_embedding

    def build_lstm(self, keep_prob):
        """create lstm cell and init state"""
        def get_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell
        lstm_cell = get_cell()
        init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        return lstm_cell, init_state

    def build_dynamic_rnn(self, cells, lstm_input, lstm_state):
        """using dynamic rnn to run LSTM automatically"""
        lstm_output, final_state = tf.nn.dynamic_rnn(cells, lstm_input, initial_state=lstm_state)
        # reshape lstm_output from [batch_size, time_steps, n_units] to [batch_size*time_steps, n_units]
        lstm_output = tf.concat(lstm_output, axis=1)
        lstm_output = tf.reshape(lstm_output, [-1, self.num_hidden_units])
        return lstm_output, final_state

    def build_n_output(self, lstm_output):
        """using a trainable matrix to transform the output of lstm to non-terminal token prediction"""
        with tf.variable_scope('non_terminal_softmax'):
            nt_weight = tf.Variable(tf.truncated_normal(
                [self.num_hidden_units, self.num_ntoken], stddev=0.1))
            nt_bias = tf.Variable(tf.zeros(self.num_ntoken))
        nonterminal_logits = tf.matmul(lstm_output, nt_weight) + nt_bias
        nonterminal_output = tf.nn.softmax(logits=nonterminal_logits, name='nonterminal_output')
        return nonterminal_logits, nonterminal_output

    def build_t_output(self, lstm_output):
        """using a trainable matrix to transform the otuput of lstm to terminal token prediction"""
        with tf.variable_scope('terminal_softmax'):
            t_weight = tf.Variable(tf.truncated_normal([self.num_hidden_units, self.num_ttoken], stddev=0.1))
            t_bias = tf.Variable(tf.zeros(self.num_ttoken))
        terminal_logits = tf.matmul(lstm_output, t_weight) + t_bias
        termnial_output = tf.nn.softmax(
            logits=terminal_logits, name='terminal_output')
        return terminal_logits, termnial_output

    def build_loss(self, n_loss, t_loss):
        """add n_loss, t_loss together"""
        loss = tf.add(n_loss, t_loss)
        return loss

    def build_nt_loss(self, n_logits, n_target):
        """calculate the loss function of non-terminal prediction"""
        n_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_logits, labels=n_target)
       # n_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=n_logits, labels=n_targets)
        n_loss = tf.reduce_mean(n_loss)
        return n_loss

    def build_tt_loss(self, t_logits, t_target):
        """calculate the loss function of terminal prediction"""
        t_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=t_logits, labels=t_target)
        # t_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=t_logits, labels=t_targets)
        t_loss = tf.reduce_mean(t_loss)
        return t_loss

    def build_accuracy(self, n_output, n_target, t_output, t_target):
        """calculate the predction accuracy of non-terminal terminal prediction"""
        n_equal = tf.equal(tf.argmax(n_output, axis=1), n_target)
        t_equal = tf.equal(tf.argmax(t_output, axis=1), t_target)
        n_accuracy = tf.reduce_mean(tf.cast(n_equal, tf.float32))
        t_accuracy = tf.reduce_mean(tf.cast(t_equal, tf.float32))
        return n_accuracy, t_accuracy

    def build_topk_accuracy(self, n_output, n_target, t_output, t_target, define_k=3):
        """calculate the accuracy of non-terminal terminal top k prediction"""
        n_topk_equal = tf.nn.in_top_k(n_output, n_target, k=define_k)
        t_topk_equal = tf.nn.in_top_k(t_output, t_target, k=define_k)
        n_topk_accu = tf.reduce_mean(tf.cast(n_topk_equal, tf.float32))
        t_topk_accu = tf.reduce_mean(tf.cast(t_topk_equal, tf.float32))
        return n_topk_accu, t_topk_accu

    def build_topk_prediction(self, n_output, t_output, define_k=3):
        """return the top k prediction by model"""
        n_topk_possibility, n_topk_prediction = tf.nn.top_k(n_output, k=define_k)
        t_topk_possibility, t_topk_prediction = tf.nn.top_k(t_output, k=define_k)
        return n_topk_prediction, n_topk_possibility, t_topk_prediction, t_topk_possibility

    def build_optimizer(self, loss):
        """build optimizer for model, using learning rate decay and gradient clip"""
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

    def build_onehot_target(self, n_target, t_target):
        """not used, transform int target to one-hot-encoding target"""
        onehot_n_target = tf.one_hot(n_target, self.num_ntoken)
        n_shape = (self.batch_size * self.time_steps, self.num_ntoken)
        t_shape = (self.batch_size * self.time_steps, self.num_ttoken)
        onehot_n_target = tf.reshape(onehot_n_target, n_shape)
        onehot_t_target = tf.one_hot(t_target, self.num_ttoken)
        onehot_t_target = tf.reshape(onehot_t_target, t_shape)
        return onehot_n_target, onehot_t_target

    def build_summary(self, summary_dict):
        """summary model info for tensorboard"""
        for key, value in summary_dict.items():
            tf.summary.scalar(key, value)
        merged_op = tf.summary.merge_all()
        return merged_op

    def build_model(self):
        """create model"""
        tf.reset_default_graph()
        self.n_input, self.t_input, self.n_target, self.t_target, self.keep_prob = self.build_input()
        n_input_embedding, t_input_embedding = self.build_input_embed(
            self.n_input, self.t_input)

        lstm_input = tf.add(n_input_embedding, t_input_embedding)
        cells, self.init_state = self.build_lstm(self.keep_prob)
        self.lstm_state = self.init_state
        lstm_output, self.final_state = self.build_dynamic_rnn(cells, lstm_input, self.lstm_state)

        n_logits, self.n_output = self.build_n_output(lstm_output)
        t_logits, self.t_output = self.build_t_output(lstm_output)

        n_target = tf.reshape(self.n_target, [self.batch_size*self.time_steps])
        t_target = tf.reshape(self.t_target, [self.batch_size*self.time_steps])
        self.n_loss = self.build_nt_loss(n_logits, n_target)
        self.t_loss = self.build_tt_loss(t_logits, t_target)

        # onehot_n_target, onehot_t_target = self.build_onehot_target(
        #        self.n_target, self.t_target)
        # self.n_loss = self.build_nt_loss(n_logits, onehot_n_target)
        # self.t_loss = self.build_tt_loss(t_logits, onehot_t_target)
        
        self.loss = self.build_loss(self.n_loss, self.t_loss)
        self.n_accu, self.t_accu = self.build_accuracy(
            self.n_output, n_target, self.t_output, t_target)
        # top k prediction accuracy
        self.n_top_k_accu, self.t_top_k_accu = self.build_topk_accuracy(
            self.n_output, n_target, self.t_output, t_target)
        self.optimizer = self.build_optimizer(self.loss)

        # top k prediction with possibility
        self.n_topk_pred, self.n_topk_poss, self.t_topk_pred, self.t_topk_poss = \
            self.build_topk_prediction(self.n_output, self.t_output)

        summary_dict = {'train loss': self.loss, 'non-terminal loss': self.t_loss,
                        'terminal loss': self.t_loss, 'n_accuracy': self.n_accu,
                        't_accuracy': self.t_loss}
        self.merged_op = self.build_summary(summary_dict)

        print('lstm model has been created...')

    def train(self):
        self.print_and_log('model training...')
        model_info = 'basic lstm model  ' + \
            'time_step:{},  batch_size:{},  hidden_units:{}'.format(
                self.time_steps, self.batch_size, self.num_hidden_units)
        self.print_and_log(model_info)
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
            n_accu_per_epoch = 0.0
            t_accu_per_epoch = 0.0
            subset_generator = self.generator.get_train_subset_data()

            for data in subset_generator:
                batch_generator = self.generator.get_batch(data_seq=data)
                lstm_state = session.run(self.init_state)

                for b_nt_x, b_nt_y, b_t_x, b_t_y in batch_generator:
                    batch_step += 1
                    global_step += 1
                    batch_start_time = time.time()

                    feed = {self.t_input: b_t_x,
                            self.n_input: b_nt_x,
                            self.n_target: b_nt_y,
                            self.t_target: b_t_y,
                            self.keep_prob: 0.5,
                            self.lstm_state: lstm_state,
                            self.global_step: global_step}
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
                        log_info = 'epoch:{}/{}  '.format(epoch, self.num_epochs) + \
                                   'global_step:{}  '.format(global_step) + \
                                   'loss:{:.2f}(n_loss:{:.2f} + t_loss:{:.2f})  '.format(loss, n_loss, t_loss) + \
                                   'nt_accu:{:.2f}%  '.format(n_accu * 100) + \
                                   'tt_accu:{:.2f}%  '.format(t_accu * 100) + \
                                   'time cost per batch:{:.2f}/s'.format(batch_end_time - batch_start_time)
                        self.print_and_log(log_info)

                    if global_step % valid_every_n == 0:
                        self.valid(session, epoch, global_step)

                    if global_step % save_every_n == 0:
                        saver.save(session, model_save_dir + 'e{}_b{}.ckpt'.format(epoch, batch_step))
                        print('model saved: epoch:{} global_step:{}'.format(epoch, global_step))
            epoch_end_time = time.time()
            epoch_cost_time = epoch_end_time - epoch_start_time

            epoch_log = 'EPOCH:{}/{}  '.format(epoch, self.num_epochs) + \
                        'time cost this epoch:{:.2f}/s  '.format(epoch_cost_time) + \
                        'epoch average loss:{:.2f}  '.format(loss_per_epoch / batch_step) + \
                        'epoch average nt_accu:{:.2f}%  '.format(100*n_accu_per_epoch / batch_step) + \
                        'epoch average tt_accu:{:.2f}%  '.format(100*t_accu_per_epoch / batch_step) + '\n'
            self.print_and_log(epoch_log)

        saver.save(session, model_save_dir + 'lastest_model.ckpt')
        self.print_and_log('model training finished...')
        session.close()

    def valid(self, session, epoch, global_step):
        """valid model when it is trained"""
        valid_data = self.generator.get_valid_subset_data()
        batch_generator = self.generator.get_batch(valid_data)
        valid_step = 0
        valid_n_accuracy = 0.0
        valid_t_accuracy = 0.0
        valid_times = 400
        valid_start_time = time.time()
        lstm_state = session.run(self.init_state)
        for b_nt_x, b_nt_y, b_t_x, b_t_y in batch_generator:
            valid_step += 1
            feed = {self.t_input: b_t_x,
                    self.n_input: b_nt_x,
                    self.n_target: b_nt_y,
                    self.t_target: b_t_y,
                    self.keep_prob: 1.0,
                    self.lstm_state: lstm_state,
                    self.global_step: global_step}
            n_accuracy, t_accuracy, lstm_state = session.run(
                [self.n_accu, self.t_accu, self.final_state], feed)
            valid_n_accuracy += n_accuracy
            valid_t_accuracy += t_accuracy
            if valid_step >= valid_times:
                break

        valid_n_accuracy /= valid_step
        valid_t_accuracy /= valid_step
        valid_end_time = time.time()
        valid_log = "VALID epoch:{}/{}  ".format(epoch, self.num_epochs) + \
                    "global step:{}  ".format(global_step) + \
                    "valid_nt_accu:{:.2f}%  ".format(valid_n_accuracy * 100) + \
                    "valid_tt_accu:{:.2f}%  ".format(valid_t_accuracy * 100) + \
                    "valid time cost:{:.2f}s".format(valid_end_time - valid_start_time)
        if not os.path.exists(valid_log_dir):
            self.valid_file = open(valid_log_dir, 'w')
        valid_info = '{} {} {}\n'.format(global_step, valid_n_accuracy, valid_t_accuracy)
        self.valid_file.write(valid_info)
        self.print_and_log(valid_log)

    def print_and_log(self, info):
        if not os.path.exists(training_log_dir):
            self.log_file = open(training_log_dir, 'w')
        self.log_file.write(info)
        self.log_file.write('\n')
        print(info)


if __name__ == '__main__':
    num_terminal = base_setting.num_terminal
    num_non_terminal = base_setting.num_non_terminal
    model = RnnModel(num_non_terminal, num_terminal)
    model.train()
