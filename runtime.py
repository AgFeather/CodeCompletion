import tensorflow as tf
import numpy as np
import time
import random
import pickle

import utils
from lstm_model import RnnModel
from setting import Setting


test_setting = Setting()
test_subset_data_dir = test_setting.sub_int_test_dir
model_save_dir = test_setting.lstm_model_save_dir
test_log_dir = test_setting.lstm_test_log_dir


num_subset_test_data = test_setting.num_sub_test_data
seq_per_subset = 5000
num_non_terminal = test_setting.num_non_terminal
num_terminal = test_setting.num_terminal

test_time_step = 50


class OnlineCompletion(object):

    def __init__(self,
                 num_ntoken,
                 num_ttoken,):
        self.model = RnnModel(num_ntoken, num_ttoken, is_training=False)
        self.sess = tf.Session()
        self.last_chackpoints = tf.train.latest_checkpoint(
            checkpoint_dir=model_save_dir)

        saver = tf.train.Saver()
        saver.restore(self.sess, self.last_chackpoints)
        self.log_file = open(test_log_dir, 'w')

    # query test
    def eval(self, prefix):
        """
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        """
        terminalToken2int, terminalInt2token, nonTerminalToken2int, nonTerminalInt2token = utils.load_dict_parameter()
        new_state = self.sess.run(self.model.init_state)
        test_batch = self.get_batch(prefix)
        n_prediction, t_prediction = None, None
        for nt_token, tt_token in test_batch:
            feed = {self.model.n_input: nt_token,
                    self.model.t_input: tt_token,
                    self.model.keep_prob: 1.,
                    self.model.lstm_state: new_state}
            n_prediction, t_prediction, new_state = self.sess.run(
                [self.model.n_output, self.model.t_output, self.model.final_state], feed_dict=feed)

        assert n_prediction is not None and t_prediction is not None
        n_prediction = n_prediction[-1, :]
        t_prediction = t_prediction[-1, :]
        n_prediction = np.argmax(n_prediction)
        t_prediction = np.argmax(t_prediction)
        return n_prediction, t_prediction

    def prediction(self, prefix):
        self.eval(prefix=prefix)

    def get_batch(self, prefix):
        prefix = np.array(prefix)
        for index in range(0, len(prefix), test_time_step):
            nt_token = prefix[index: index+test_time_step, 0].reshape([1, -1])
            tt_token = prefix[index: index+test_time_step, 1].reshape([1, -1])
            yield nt_token, tt_token

    def subset_generator(self):
        for index in range(1, num_subset_test_data+1):
            with open(test_subset_data_dir + 'int_part{}.json'.format(index), 'rb') as file:
                subset_data = pickle.load(file)
                yield index, subset_data


    def test_log(self, log_info):
        self.log_file.write(log_info)
        self.log_file.write('\n')
        print(log_info)


def load_code_file():
    pass


def code_to_ast():
    pass


def ast_to_nt_seq():
    pass

def to_int_nt_seq():
    pass


if __name__ == '__main__':

    test_model = OnlineCompletion(num_non_terminal, num_terminal)
    code_file = load_code_file()
    ast_data = code_to_ast()
    nt_seq = ast_to_nt_seq()
    int_nt_seq = to_int_nt_seq()
    nt_token_prediction, tt_token_prediction = test_model.prediction(int_nt_seq)
