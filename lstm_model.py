import tensorflow as tf
import pickle

import utils


num_subset_train_data = 20
subset_data_dir = 'split_js_data/train_data/'


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
        target = tf.placeholder(tf.float32, [self.batch_size, 1])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return n_input, t_input, target, keep_prob

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

    def build_output(self, lstm_output):
        # 将lstm_output的形状由[batch_size, time_steps, n_units] 转换为 [batch_size*time_steps, n_units]
        seq_output = tf.concat(lstm_output, axis=1)
        seq_output = tf.reshape(seq_output, [-1, self.num_hidden_units])

        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([self.num_hidden_units, self.num_ntoken], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_ntoken))

        logits = tf.matmul(seq_output, softmax_w) + softmax_b
        softmax_output = tf.nn.softmax(logits=logits, name='softmax_output')
        return softmax_output, logits

    def build_loss(self, logits, targets):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=targets)
        loss = tf.reduce_mean(loss)
        return loss

    def bulid_accuracy(self, softmax_output, targets):
        equal = tf.equal(tf.argmax(softmax_output, axis=1), tf.argmax(targets, axis=1))
        accuracy = tf.cast(equal, tf.float32)
        accuracy = tf.reduce_mean(accuracy)
        return accuracy

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
        self.n_input, self.t_input, self.target, self.keep_prob = self.build_input()
        n_input_embedding, t_input_embedding = self.build_input_embed(self.n_input, self.t_input)
        lstm_input = tf.concat(n_input_embedding, t_input_embedding)
        cells, self.init_state = self.build_lstm(self.keep_prob)
        lstm_output, self.final_state = tf.nn.dynamic_rnn(cells, lstm_input, initial_state=self.init_state)
        softmax_output, logits = self.build_output(lstm_output)
        self.loss = self.build_loss(logits, self.target)
        self.accu = self.bulid_accuracy(softmax_output, self.target)
        self.optimizer = self.bulid_optimizer(self.loss)

    def get_batch(self, data):
        pass

    def train(self):
        print('model training...')
        saver = tf.train.Saver()
        session = tf.Session()
        global_step = 0
        for epoch in range(self.num_epoches):
            subset_generator = get_subset_data()
            for data in get_subset_data():
                batch_generator = self.get_batch(data)
                for batch_x, batch_y in batch_generator:
                    global_step += 1

        session.close()





def get_subset_data():
    for i in range(1, num_subset_train_data+1):
        data_path = subset_data_dir + f'part{i}.json'
        file = open(data_path, 'rb')
        data = pickle.load(file)
        yield data



if __name__ == '__main__':
    terminalToken2int, terminalInt2token, nonTerminalToken2int, nonTerminalInt2token = utils.load_dict_parameter()
    num_ntoken = len(nonTerminalInt2token)
    num_ttoken = len(terminalInt2token)
    model = LSTM_Model(num_ntoken, num_ttoken)
    model.train()
