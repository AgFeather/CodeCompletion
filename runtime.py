import tensorflow as tf
import numpy as np
import time
import random
import pickle
import json

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



def load_code_file(code_file_path='js_parser/test.json'):
    file = open(code_file_path, 'r')
    source_code = file.readlines()
    ast = json.load(source_code)
    return ast

def ast_to_nt_seq():
    pass

def to_int_nt_seq():
    pass


if __name__ == '__main__':

    test_model = OnlineCompletion(num_non_terminal, num_terminal)
    code_file = load_code_file()
    nt_seq = ast_to_nt_seq()
    int_nt_seq = to_int_nt_seq()
