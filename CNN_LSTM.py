import tensorflow as tf
import numpy as np
import time

import data_utils


'''
Build a seq2seq structure, 
encoder is a CNN with sliding windows which is used to extract the feature of context,
decoder is a LSTM which can predict the next token in time sequence.
'''


x_train_data_path = 'processed_data/num_train_data.p'
y_train_data_path = 'processed_data/y_train_data.p'
train_data_parameter = 'processed_data/x_y_parameter.p'

tensorboard_log_path = 'logs/CNN_LSTM'
model_save_dir = 'checkpoints/cnn_lstm/'
query_dir = 'dataset/programs_200/'



sliding_window = [2,3,4,5]
show_every_n = 100
save_every_n = 500


class CNN_LSTM(object):

    def __init__(self,
                 token_set,
                 batch_size=64,
                 learning_rate=0.01,
                 num_epoches=1,
                 embed_dim=32,
                 filter_num=4,
                 time_steps=50,
                 context_size=15,
                 rnn_units=128,
                 num_rnn_layers=2,
                 cnn_units=128):

        self.num_tokens = len(token_set)
        self.representation_shape = len(sliding_window) * filter_num * embed_dim
        self.batch_size = batch_size
        self.num_epoches = num_epoches
        self.time_steps = time_steps
        self.context_size = context_size
        self.learning_rate = learning_rate
        self.embed_dim = embed_dim
        self.filter_num = filter_num
        self.rnn_units = rnn_units
        self.num_rnn_layers=num_rnn_layers
        self.cnn_units = cnn_units

        self.build_model()

    def bulid_input(self):
        input_x = tf.placeholder(
            dtype=tf.int32, shape=[self.batch_size, self.context_size], name='input_x')
        output_y = tf.placeholder(
            dtype=tf.float32, shape=[self.batch_size, self.num_tokens], name='output_y')
        keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        return input_x, output_y, keep_prob


    # neural network functions
    def bulid_CNN(self, input_x, keep_prob):
        self.embedding_matrix = tf.Variable(tf.truncated_normal(
            [self.num_tokens, self.embed_dim]), name='embedding_matrix')
        self.embedding_represent = tf.nn.embedding_lookup(self.embedding_matrix, input_x,
                                                          name='embedding_represent')
        self.embedding_represent = tf.expand_dims(self.embedding_represent, -1)
        # print(self.embedding_represent.get_shape()) # (None, 5, 32, 1)

        conv_layers_list = []
        weights = {}
        biases = {}
        for window in sliding_window:
            with tf.name_scope('convolution_layer_{}'.format(window)):
                conv_weight = tf.Variable(tf.truncated_normal(
                    shape=[window, self.embed_dim, 1, self.filter_num]), name='conv_weight')
                conv_bias = tf.Variable(tf.truncated_normal(
                    shape=[window, self.embed_dim, 1, self.filter_num]), name='conv_bias')
                weights['conv%d' % window] = conv_weight
                biases['conv%d' % window] = conv_bias
                conv_layer = tf.nn.conv2d(
                    self.embedding_represent, conv_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv_layer_1')
                relu_layer = tf.nn.relu(conv_layer, name='relu_layer')
                avgpool_layer = tf.nn.avg_pool(
                    relu_layer, [1, self.context_size, 1, 1], [1, 1, 1, 1], padding='VALID', name='avgpool_layer')
                # maxpooling or avgpooling? parameter adjust？
                conv_layers_list.append(avgpool_layer)  # (?, 1, 32, 4)
                # print(avgpool_layer.get_shape())

        with tf.name_scope('dropout_layer'):
            dropout_layer = tf.concat(conv_layers_list, 3, name='concat_conv_layers')
            dropout_layer = tf.reshape(dropout_layer, [-1, self.representation_shape])
            dropout_layer = tf.nn.dropout(dropout_layer, keep_prob, name='dropout_layer')
            # print(dropout_layer.get_shape()) #(?, 512)
        weights['h1'] = tf.Variable(tf.truncated_normal(
            shape=[self.representation_shape, self.cnn_units]), name='h1_weight')
        biases['h1'] = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[self.cnn_units]))
        representation_layer = tf.matmul(dropout_layer, weights['h1']) + biases['h1']
        return representation_layer

    def bulid_loss(self, logits, targets):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=targets)
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

    def bulid_accuracy(self, logits, targets):
        logits = tf.argmax(logits, axis=1)
        targets = tf.argmax(targets, axis=1)
        equality = tf.equal(logits, targets)
        accuracy = tf.cast(equality, tf.float32)
        accuracy = tf.reduce_mean(accuracy)
        return accuracy

    def bulid_output(self, outputs):
        output_weight = tf.Variable(tf.random_uniform(
            shape=[self.rnn_units, self.num_tokens], dtype=tf.float32))
        output_bias = tf.Variable(tf.constant([0.1], dtype=tf.float32, shape=[self.num_tokens]))
        logits = tf.matmul(outputs, output_weight) + output_bias
        return logits

    def bulid_LSTM(self, keep_prob, representation_layer):
        representation_layer = tf.reshape(
            representation_layer, [-1, self.time_steps, self.representation_shape])
        def get_cell(cell='LSTM'):
            if cell == 'LSTM':
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_units, forget_bias=1.0, state_is_tuple=True)
                drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            else:
                gru_cell = tf.contrib.rnn.GRUCell(self.rnn_units, forget_bias=1.0, state_is_tuple=True)
                drop = tf.contrib.rnn.DropoutWrapper(gru_cell, output_keep_prob=keep_prob)
            return drop
        cell_list = []
        for num in range(self.num_rnn_layers):
            cell_list.append(get_cell())
        cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        init_state = cells.zero_state(self.batch_size, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(
            cells, representation_layer, initial_state=init_state, time_major=False)
        outputs = tf.reshape(outputs, [-1, self.rnn_units])
        return outputs, init_state, final_state


    def build_model(self):
        self.input_x, self.output_y, self.keep_prob = self.bulid_input()
        self.representation_layer = self.bulid_CNN(self.input_x, self.keep_prob)
        self.outputs, self.init_state, self.final_state = self.bulid_LSTM(self.keep_prob, self.representation_layer)
        self.logits = self.bulid_output(self.outputs)
        self.accuracy = self.bulid_accuracy(self.logits, self.output_y)
        self.loss = self.bulid_loss(self.logits, self.output_y)
        self.optimizer = self.bulid_optimizer(self.loss)

        tf.summary.scalar('train loss', self.loss)
        tf.summary.scalar('train accuracy', self.accuracy)
        self.merged_op = tf.summary.merge_all()


    def train(self, train_x, train_y):
        self.sess = tf.Session()
        writer = tf.summary.FileWriter(tensorboard_log_path, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        global_step = 0
        for epoch in range(self.num_epoches):
            batch_generator = self.get_batch(train_x, train_y)
            for batch_x, batch_y in batch_generator:
                start_time = time.time()
                global_step += 1
                feed = {
                    self.input_x: batch_x,
                    self.output_y: batch_y,
                    self.keep_prob: 0.5}
                show_loss, show_accu, summary_str, _ = self.sess.run(
                    [self.loss, self.accuracy, self.merged_op, self.optimizer], feed_dict=feed)
                writer.add_summary(summary_str, i)
                writer.flush()
                end_time = time.time()
                if global_step % show_every_n == 0:
                    print('epoch:{}/{}..'.format(epoch, self.num_epoches),
                          'steps:{}..'.format(global_step),
                          'loss:{:.3f}..'.format(show_loss),
                          'accuracy:{:.4f}..'.format(show_accu),
                          'batch time cost:{.2f}'.format(end_time-start_time))
                if global_step % save_every_n == 0:
                    saver.save(self.sess, model_save_dir+'e{}_s{}'.format(epoch, global_step))

        saver.save(self.sess, model_save_dir+'latest_model')
        print('model train finished')

    def get_batch(self, train_x, train_y):
        for i in range(0, len(train_x), self.batch_size):
            if i >= self.context_size - 1:
                batch_x = np.zeros(shape=[self.batch_size, self.context_size], dtype=np.int32)
                batch_y = np.array(train_y[i:i + self.batch_size])
                for b in range(self.batch_size):
                    batch_x[i] = train_x[i + b:i + b + self.context_size]
                yield batch_x, batch_y



class CodeCompletion(object):
    def __int__(self, token_set, token2int, int2token):
        self.token2int = token2int
        self.int2token = int2token
        self.model = CNN_LSTM(token_set)
        self.sess = tf.Session()
        checkpoint = tf.train.latest_checkpoint(model_save_dir)
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint)

    def query_test(self, prefix, suffix):
        '''
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        In this function, use only one token before hole token to predict
        return: the most probable token
        '''
        new_state = self.model.init_state()
        prediction = None
        for i, token in enumerate(prefix):
            x = np.zeros((1,1), dtype=np.int32)
            x[0,0] = token
            feed = {self.model.init_state:new_state,
                    self.model.keep_prob:1.0,
                    self.model.input_x:x}
            prediction ,new_state = self.sess.run(
                [self.model.logits, self.model.final_state],feed_dict=feed)
        prediction = self.int2token[prediction]
        return prediction

    def test_model(self, query_test_data):
        correct = 0.0
        start_time=time.time()
        correct_token_list = []
        incorrect_token_list = []
        for token_sequence in query_test_data:
            prefix, expection, suffix = data_utils.create_hole(token_sequence)
            prefix = self.token_to_int[prefix]
            prediction = self.query_test(prefix, suffix)
            if data_utils.token_equals([prediction], expection):
                correct += 1
                correct_token_list.append({'expection': expection, 'prediction': prediction})
            else:
                incorrect_token_list.append({'expection': expection, 'prediction': prediction})
        accuracy = correct / len(query_test_data)
        return accuracy

    def token_to_int(self, token_seq):
        int_token_seq = []
        for token in token_seq:
            int_token = self.token2int[data_utils.token_to_string(token)]
            int_token_seq.append(int_token)
        return int_token_seq


if __name__ == '__main__':


    x_data = data_utils.load_data_with_pickle(x_train_data_path)
    y_data = data_utils.load_data_with_pickle(y_train_data_path)
    token_set, string2int, int2string = data_utils.load_data_with_pickle(train_data_parameter)

    #model train
    model = CNN_LSTM(token_set)
    model.train(x_data, y_data)

    # test model
    query_test_data = data_utils.load_data_with_file(query_dir)
    test_accuracy = 0.0
    test_epoches = 5
    for i in range(test_epoches):
        accuracy = model.test_model(query_test_data)
        print('test epoch: %d, query test accuracy: %.3f'%(i, accuracy))
        test_accuracy += accuracy
    print('total test accuracy: %.3f'%(test_accuracy/test_epoches))