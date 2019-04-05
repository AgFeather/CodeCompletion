import tensorflow as tf
import pickle
import sys

sys.path.append('..')

from setting import Setting

base_setting = Setting()

model_save_dir = base_setting.lstm_model_save_dir
tensorboard_log_dir = base_setting.lstm_tb_log_dir
training_log_dir = base_setting.lstm_train_log_dir
valid_log_dir = base_setting.lstm_valid_log_dir

show_every_n = base_setting.show_every_n
save_every_n = base_setting.save_every_n
valid_every_n = base_setting.valid_every_n



class LSTM_Node_Embedding(object):
    """Node2Vec + LSTM 模型，使用前者对vector进行初始化，后者实现预测"""
    def __init__(self,
                 num_ntoken, num_ttoken, is_training=True,
                 batch_size=50,
                 word2vec_dim=300,
                 num_hidden_units=1500,
                 learning_rate=0.001,
                 time_steps=50,
                 grad_clip=5,):

        self.num_ntoken = num_ntoken
        self.num_ttoken = num_ttoken
        self.word2vec_dim = word2vec_dim
        self.num_hidden_units = num_hidden_units
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.time_steps = time_steps
        self.batch_size = batch_size

        if not is_training:
            self.batch_size = 1
            self.time_steps = 50

        self.nt_init_matrix, self.tt_init_matrix = self.get_represent_matrix()
        self.build_model()

    def get_represent_matrix(self):
        file = open('temp_data/tt_embedding_matrix.pkl', 'rb')
        tt_matrix = pickle.load(file)
        file = open('temp_data/nt_embedding_matrix.pkl', 'rb')
        nt_matrix = pickle.load(file)
        return nt_matrix, tt_matrix

    def build_input(self):
        """create input and target placeholder"""
        n_input = tf.placeholder(tf.int32, [None, None], name='n_input')
        t_input = tf.placeholder(tf.int32, [None, None], name='t_input')
        n_target = tf.placeholder(tf.int32, [None, None], name='n_target')
        t_target = tf.placeholder(tf.int32, [None, None], name='t_target')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return n_input, t_input, n_target, t_target, keep_prob

    def build_represent_embed(self, n_token, t_token):
        self.n_embed_matrix = tf.Variable(
            initial_value=self.nt_init_matrix, trainable=True, name='n_embed_matrix') # trainable 问题
        self.t_embed_matrix = tf.Variable(
            initial_value=self.tt_init_matrix, trainable=True, name='t_embed_matrix')
        n_input_embedding = tf.nn.embedding_lookup(self.n_embed_matrix, n_token)
        t_input_embedding = tf.nn.embedding_lookup(self.t_embed_matrix, t_token)
        return n_input_embedding, t_input_embedding

    def build_lstm(self, keep_prob):
        """create lstm cell and init state"""
        def get_cell():
            cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell
        lstm_cell = get_cell()
        init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        return lstm_cell, init_state

    def build_dynamic_rnn(self, cells, lstm_input, lstm_state):
        """using dynamic rnn to run LSTM automatically"""
        lstm_output, final_state = tf.nn.dynamic_rnn(cells, lstm_input, initial_state=lstm_state)
        # reshape lstm_output from [batch_size, time_steps, n_units] to [batch_size*time_steps, n_units]
        lstm_output = tf.reshape(lstm_output, [-1, self.num_hidden_units])

        return lstm_output, final_state

    def build_n_output(self, lstm_output):
        """using a trainable matrix to transform the output of lstm to non-terminal token prediction"""
        with tf.variable_scope('non_terminal_softmax'):
            nt_weight = tf.Variable(tf.random_uniform(
                [self.num_hidden_units, self.word2vec_dim], minval=-0.05, maxval=0.05))
            nt_bias = tf.Variable(tf.zeros(self.word2vec_dim))
        nt_logits = tf.matmul(lstm_output, nt_weight) + nt_bias
        return nt_logits

    def build_t_output(self, lstm_output):
        """using a trainable matrix to transform the otuput of lstm to terminal token prediction"""
        with tf.variable_scope('terminal_softmax'):
            t_weight = tf.Variable(tf.random_uniform(
                [self.num_hidden_units, self.word2vec_dim], minval=-0.05, maxval=0.05))
            t_bias = tf.Variable(tf.zeros(self.word2vec_dim))
        tt_logits = tf.matmul(lstm_output, t_weight) + t_bias
        return tt_logits

    def build_softmax(self, logits):
        softmax_output = tf.nn.softmax(logits=logits)
        return softmax_output

    def build_loss(self, n_loss, t_loss):
        """add n_loss, t_loss together"""
        loss = tf.add(n_loss, t_loss)
        return loss

    def build_nt_loss(self, n_logits, n_targets):
        """calculate the loss function of non-terminal prediction"""
        n_loss = tf.sqrt(tf.square(n_logits - n_targets))
        n_loss = tf.reduce_mean(n_loss)
        return n_loss

    def build_tt_loss(self, t_logits, t_targets):
        """calculate the loss function of terminal prediction"""
        t_loss = tf.sqrt(tf.square(t_logits - t_targets))
        t_loss = tf.reduce_mean(t_loss)
        return t_loss

    def build_accuracy(self, n_output, n_target, t_output, t_target):
        n_most_similar = self.get_most_similar(self.n_embed_matrix, n_output)
        t_most_similar = self.get_most_similar(self.t_embed_matrix, t_output)
        n_most_similar = tf.cast(n_most_similar, tf.int32)
        t_most_similar = tf.cast(t_most_similar, tf.int32)
        # print(t_most_similar.get_shape())
        # print(t_target.get_shape())
        # print(n_most_similar.get_shape())
        # print(n_target.get_shape())
        n_equal = tf.equal(n_most_similar, n_target)
        t_equal = tf.equal(t_most_similar, t_target)
        n_accuracy = tf.reduce_mean(tf.cast(n_equal, tf.float32))
        t_accuracy = tf.reduce_mean(tf.cast(t_equal, tf.float32))
        return n_accuracy, t_accuracy

    def get_most_similar(self, represent_matrix, output):
        """计算给定embedding matrix中距离最近的vector对应的index，使用TensorFlow中距离计算函数"""
        num_batch = tf.shape(output)[0]
        num_matrix = tf.shape(represent_matrix)[0]
        output_new = tf.expand_dims(output, -1)
        output_new = tf.tile(output_new, tf.stack([1, 1, num_matrix]))

        matrix_new = tf.expand_dims(represent_matrix, -1)
        matrix_new = tf.tile(matrix_new, tf.stack([1, 1, num_batch]))
        matrix_new = tf.transpose(matrix_new, perm=[2, 1, 0])

        diff = tf.subtract(output_new, matrix_new)
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, 1) # 距离矩阵

        most_similar_index = tf.argmin(square_dist, axis=1) # todo 现在返回的是其本身，修改成计算最近的别的

        return most_similar_index

    def build_optimizer(self, loss):
        """build optimizer for model, using learning rate decay and gradient clip"""
        self.decay_epoch = tf.Variable(0, trainable=False)
        decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.decay_epoch, 0.2, 0.9)
        optimizer = tf.train.AdamOptimizer(decay_learning_rate)
        gradient_pair = optimizer.compute_gradients(loss)
        clip_gradient_pair = []
        for grad, var in gradient_pair:
            grad = tf.clip_by_value(grad, -self.grad_clip, self.grad_clip)
            clip_gradient_pair.append((grad, var))
        optimizer = optimizer.apply_gradients(clip_gradient_pair)
        return optimizer, decay_learning_rate

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
        n_input_embedding, t_input_embedding = self.build_represent_embed(
            self.n_input, self.t_input)
        n_target_embedding, t_target_embedding = self.build_represent_embed(
            self.n_target, self.t_target)
        lstm_input = tf.add(n_input_embedding, t_input_embedding)

        cells, self.init_state = self.build_lstm(self.keep_prob)
        self.lstm_state = self.init_state
        lstm_output, self.final_state = self.build_dynamic_rnn(cells, lstm_input, self.lstm_state)

        n_logits = self.build_n_output(lstm_output)
        t_logits = self.build_t_output(lstm_output)

        # loss calculate
        # n_target = tf.reshape(self.n_target, [self.batch_size*self.time_steps])
        # t_target = tf.reshape(self.t_target, [self.batch_size*self.time_steps])
        n_target_embedding = tf.reshape(n_target_embedding, [-1, self.word2vec_dim])
        t_target_embedding = tf.reshape(t_target_embedding, [-1, self.word2vec_dim])

        # print(n_logits.get_shape())
        # print(t_logits.get_shape())
        # print(t_output_embedding.get_shape())
        # print(n_output_embedding.get_shape())

        self.n_loss = self.build_nt_loss(n_logits, n_target_embedding)
        self.t_loss = self.build_tt_loss(t_logits, t_target_embedding)
        self.loss = self.build_loss(self.n_loss, self.t_loss)
        # optimizer
        self.optimizer, self.decay_learning_rate = self.build_optimizer(self.loss)

        # # top one accuracy
        temp_n_target = tf.reshape(self.n_target, [-1]) # should be equal to self.batch_size*self.time_steps
        temp_t_target = tf.reshape(self.t_target, [-1])
        self.n_accu, self.t_accu = self.build_accuracy(n_logits, temp_n_target, t_logits, temp_t_target)
        #
        # # top k accuracy
        # self.n_topk_accu, self.t_topk_accu = self.build_accuracy(
        #     n_logits, n_target, t_logits, t_target, topk=3)

        # top k prediction with possibility
        # self.n_output = self.build_softmax(n_logits)
        # self.t_output = self.build_softmax(t_logits)
        # self.n_topk_pred, self.n_topk_poss, self.t_topk_pred, self.t_topk_poss = \
        #     self.build_topk_prediction(self.n_output, self.t_output)

        summary_dict = {'train loss': self.loss,
                        'non-terminal loss': self.t_loss, 'terminal loss': self.t_loss,
                        'learning_rate': self.decay_learning_rate}
        self.merged_op = self.build_summary(summary_dict)

        print('lstm model with Word2Vec has been created...')












if __name__ == '__main__':
    num_terminal = base_setting.num_terminal
    num_non_terminal = base_setting.num_non_terminal
    model = LSTM_Node_Embedding(num_non_terminal, num_terminal)
