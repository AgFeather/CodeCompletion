import tensorflow as tf
import pickle
import numpy as np

import utils


num_subset_train_data = 20
subset_data_dir = 'split_js_data/train_data/'
model_save_dir = 'lstm_model/'
show_every_n = 200
save_every_n = 1000
num_terminal = 30000

class LSTM_Model(object):
    def __init__(self,
                 num_ntoken, num_ttoken,
                 batch_size=64,
                 n_embed_dim=64,
                 t_embed_dim=200,
                 num_hidden_units=256,
                 num_hidden_layers=2,
                 learning_rate=0.001,
                 num_epoches=1,
                 time_steps=50, ):
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

        self.build_model()

    def build_input(self):
        n_input = tf.placeholder(tf.float32, [self.batch_size, self.time_steps], name='n_input')
        t_input = tf.placeholder(tf.float32, [self.batch_size, self.time_steps], name='t_input')
        n_target = tf.placeholder(tf.float32, [self.batch_size, self.time_steps], name='n_target')
        t_target = tf.placeholder(tf.float32, [self.batch_size, self.time_steps], name='t_target')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return n_input, t_input, n_target, t_target, keep_prob

    def build_input_embed(self, n_input, t_input):
        n_embed_matrix = tf.Variable(tf.truncated_normal(
            [self.num_ntoken, self.n_embed_dim]), name='n_embed_matrix')
        t_embed_matrix = tf.Variable(tf.truncated_normal(
            [self.num_ttoken, self.n_embed_dim]), name='t_embed_matrix')
        n_input_embedding = tf.nn.embedding_lookup(n_embed_matrix, n_input)
        t_input_embedding = tf.nn.embedding_lookup(t_embed_matrix, t_input)
        return n_input_embedding, t_input_embedding

    def build_lstm(self, keep_prob):
        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTM(self.num_hidden_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell
        cell_list = [lstm_cell() for _ in range(self.num_hidden_layers)]
        cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        init_state = cells.zero_state(self.batch_size, dtype=tf.float32)
        return cells, init_state

    def build_n_output(self, lstm_output):
        # 将lstm_output的形状由[batch_size, time_steps, n_units] 转换为 [batch_size*time_steps, n_units]
        seq_output = tf.concat(lstm_output, axis=1)
        seq_output = tf.reshape(seq_output, [-1, self.num_hidden_units])

        with tf.variable_scope('non_terminal_softmax'):
            nt_weight = tf.Variable(tf.truncated_normal([self.num_hidden_units, self.num_ntoken], stddev=0.1))
            nt_bias = tf.Variable(tf.zeros(self.num_ntoken))

        nonterminal_logits = tf.matmul(seq_output, nt_weight) + nt_bias
        nonterminal_output = tf.nn.softmax(logits=nonterminal_logits, name='nonterminal_output')
        return nonterminal_logits, nonterminal_output

    def build_t_output(self, lstm_output):
        # 将lstm_output的形状由[batch_size, time_steps, n_units] 转换为 [batch_size*time_steps, n_units]
        seq_output = tf.concat(lstm_output, axis=1)
        seq_output = tf.reshape(seq_output, [-1, self.num_hidden_units])
        with tf.variable_scope('terminal_softmax'):
            t_weight = tf.Variable(tf.truncated_normal([self.num_ntoken, self.num_ttoken], stddev=0.1))
            t_bias = tf.Variable(tf.zeros(self.num_ttoken))

        terminal_logits = tf.matmul(seq_output, t_weight) + t_bias
        termnial_output = tf.nn.softmax(logits=terminal_logits, name='terminal_output')
        return terminal_logits, termnial_output



    def build_loss(self, n_logits, n_target, t_logits, t_target):
        n_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=n_logits, labels=n_target)
        t_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=t_logits, labels=t_target)
        loss = tf.add(n_loss, t_loss)
        loss = tf.reduce_mean(loss)
        return loss, n_loss, t_loss

    def bulid_accuracy(self, n_output, n_target, t_output, t_target):
        n_equal = tf.equal(tf.argmax(n_output, axis=1), tf.argmax(n_target, axis=1))
        t_equal = tf.equal(tf.argmax(t_output, axis=1), tf.argmax(t_target, axis=1))
        n_accuracy = tf.reduce_mean(tf.cast(n_equal, tf.float32))
        t_accuracy = tf.reduce_mean(tf.cast(t_equal, tf.float32))
        return n_accuracy, t_accuracy

    def bulid_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradient_pair = optimizer.compute_gradients(loss)
        clip_gradient_pair = []
        for grad, var in gradient_pair:
            grad = tf.clip_by_value(grad, -2, 2)
            clip_gradient_pair.append((grad, var))
        optimizer = optimizer.apply_gradients(clip_gradient_pair)
        return optimizer

    def build_model(self):
        tf.reset_default_graph()
        self.n_input, self.t_input, self.n_target, self.t_target, self.keep_prob = self.build_input()
        n_input_embedding, t_input_embedding = self.build_input_embed(self.n_input, self.t_input)
        lstm_input = tf.concat(n_input_embedding, t_input_embedding)
        cells, self.init_state = self.build_lstm(self.keep_prob)
        lstm_output, self.final_state = tf.nn.dynamic_rnn(cells, lstm_input, initial_state=self.init_state)
        t_logits, t_output = self.build_t_output(lstm_output)
        n_logits, n_output = self. build_n_output(lstm_output)
        self.loss, self.n_loss, self.t_loss = self.build_loss(n_logits, self.n_target, t_logits, self.t_target)
        self.n_accu, self.t_accu = self.bulid_accuracy(n_output, self.n_target, t_output, self.t_target)
        self.optimizer = self.bulid_optimizer(self.loss)

    def get_batch(self, data_seq):
        data_seq = np.array(data_seq)
        total_length = self.time_steps * self.batch_size
        n_batches = len(data_seq) // total_length
        data_seq = data_seq[:total_length * n_batches]
        data_seq = data_seq.reshape((self.batch_size, -1))
        for n in range(0, data_seq.shape[1], self.time_steps):
            x = data_seq[:, n:n + self.time_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

    def get_subset_data(self):
        for i in range(1, num_subset_train_data + 1):
            data_path = subset_data_dir + f'part{i}.json'
            file = open(data_path, 'rb')
            data = pickle.load(file)
            yield data

    def train(self):
        print('model training...')
        saver = tf.train.Saver()
        session = tf.Session()
        global_step = 0
        for epoch in range(self.num_epoches):
            batch_step = 0
            subset_generator = self.get_subset_data()
            for data in subset_generator:
                batch_generator = self.get_batch(data)
                for b_ntoken, b_ttoken, b_target in batch_generator:
                    batch_step += 1
                    global_step += 1
                    feed = {self.t_input: b_ttoken,
                            self.n_input:b_ntoken,
                            self.target:b_target,
                            self.keep_prob:0.5}
                    show_loss, show_accu, _ = session.run(
                        [self.loss, self.accu, self.optimizer],feed_dict=feed)

                if global_step % show_every_n == 0:
                    print(f'epoch: {epoch}/{self.num_epoches}...',
                          f'global_step: {global_step}',
                          f'loss: {show_loss}...',
                          f'accuracy: {show_accu}...')
                if global_step % save_every_n == 0:
                    saver.save(session, model_save_dir + f'e{epoch}' + f'b{batch_step}.ckpt')

        session.close()








if __name__ == '__main__':
    terminalToken2int, terminalInt2token, nonTerminalToken2int, nonTerminalInt2token = utils.load_dict_parameter()
    num_ntoken = len(nonTerminalInt2token)
    num_ttoken = len(terminalInt2token)
    model = LSTM_Model(num_ntoken, num_ttoken)
    model.train()
