import tensorflow as tf
import numpy as np
import time
import random

import utils
from lstm_model import RnnModel


subset_int_data_dir = 'split_js_data/train_data/int_format/'
model_save_dir = 'trained_model/lstm_model/'
tensorboard_log_dir = 'tensorboard_log/lstm/'
curr_time = time.strftime('_%Y_%M_%d_%H', time.localtime())
training_log_dir = 'training_log/lstm_log' + str(curr_time) + '.txt'

num_subset_train_data = 20
num_subset_test_data = 10
show_every_n = 100
save_every_n = 1500
num_terminal = 30000


class CodeCompletion(object):
    def __init__(self,
                 num_ntoken,
                 num_ttoken):
        self.model = RnnModel(num_ntoken, num_ttoken, is_training=False)
        self.last_chackpoints = tf.train.latest_checkpoint(
            checkpoint_dir=model_save_dir)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self.last_chackpoints)

    # query test
    def query_test(self, prefix, suffix):
        '''
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        '''
        new_state = self.sess.run(self.model.init_state)
        n_prediction = None
        t_prediction = None
        for i, (nt_token, tt_token) in enumerate(prefix):
            nt_x = np.zeros((1, 1), dtype=np.int32)
            tt_x = np.zeros((1, 1), dtype=np.int32)
            nt_x[0, 0] = nt_token
            tt_x[0, 0] = tt_token
            feed = {self.model.n_input: nt_x,
                    self.model.t_input: tt_x,
                    self.model.keep_prob: 1.,
                    self.model.init_state: new_state}
            n_prediction, t_prediction, new_state = self.sess.run(
                [self.model.n_output, self.model.t_output, self.model.final_state], feed_dict=feed)

        assert n_prediction is not None and t_prediction is not None
        n_prediction = np.argmax(n_prediction)
        t_prediction = np.argmax(t_prediction)
        return n_prediction, t_prediction

    def test(self, query_test_data):
        print('test step is beginning..')
        start_time = time.time()
        t_correct = 0.0
        n_correct = 0.0
        for token_sequence in query_test_data:
            prefix, expection, suffix = self.create_hole(token_sequence)
            n_prediction, t_prediction = self.query_test(prefix, suffix)
            n_expection, t_expection = expection
            if self.token_equal(n_prediction, n_expection):
                n_correct += 1
            if self.token_equal(t_prediction, t_expection):
                t_correct += 1
        n_accuracy = n_correct / len(query_test_data)
        t_accuracy = t_correct / len(query_test_data)
        end_time = time.time()
        print(
            'test finished, time cost:{:.2f}..'.format(
                end_time -
                start_time))

        return n_accuracy, t_accuracy

    def token_equal(self, prediction, expection):
        # todo: implement
        return False

    def create_hole(self, nt_token_seq, hole_size=1):
        hole_start_index = random.randint(
            len(nt_token_seq) // 2,
            len(nt_token_seq) - hole_size)
        hole_end_index = hole_start_index + hole_size
        prefix = nt_token_seq[0:hole_start_index]
        expection = nt_token_seq[hole_start_index:hole_end_index]
        suffix = nt_token_seq[hole_end_index:-1]
        return prefix, expection, suffix


if __name__ == '__main__':
    # test step
    terminalToken2int, terminalInt2token, nonTerminalToken2int, nonTerminalInt2token = utils.load_dict_parameter()
    num_ntoken = len(nonTerminalInt2token)
    num_ttoken = len(terminalInt2token)
    test_model = CodeCompletion(num_ntoken, num_ttoken)
