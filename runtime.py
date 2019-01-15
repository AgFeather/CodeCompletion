import tensorflow as tf
import numpy as np
import time
import random
import pickle
import json
import subprocess

import utils
from code_completion import CodeCompletion
from setting import Setting


test_setting = Setting()
test_subset_data_dir = test_setting.sub_int_test_dir
model_save_dir = test_setting.lstm_model_save_dir
test_log_dir = test_setting.lstm_test_log_dir
unknown_token = test_setting.unknown_token


num_subset_test_data = test_setting.num_sub_test_data
seq_per_subset = 5000
num_non_terminal = test_setting.num_non_terminal
num_terminal = test_setting.num_terminal




def code_to_ast(code_path='js_parser/js_helloworld.js'):
    out_bytes = subprocess.check_output('node js_parser/js_parser.js ' + code_path, shell=True)
    out_text = out_bytes.decode('utf-8')
    ast = json.loads(out_text)
    return ast


def load_code_file(code_file_path='js_parser/test.json'):
    file = open(code_file_path, 'r')
    source_code = file.readlines()
    ast = json.loads(source_code)
    return ast


def ast_to_nt_seq(ast):
    binary_tree = utils.bulid_binary_tree(ast)
    nt_seq = utils.ast_to_seq(binary_tree, run_or_process='run')
    return nt_seq


def to_int_nt_seq(nt_seq):
    tt_token_to_int, _, nt_token_to_int, __ = utils.load_dict_parameter()
    unknown_tt_int = tt_token_to_int[unknown_token]
    int_seq = [(nt_token_to_int.get(n_token), tt_token_to_int.get(t_token, unknown_tt_int))
                for n_token, t_token in nt_seq]
    return int_seq


class OnlineCompletion():
    def __init__(self):
        self.model = CodeCompletion(num_non_terminal, num_terminal)
        # checkpoint_path = tf.train.latest_checkpoint(model_save_dir)
        # self.session = tf.Session()
        # saver = tf.train.Saver()
        # saver.restore(self.session, checkpoint_path)

    def complete(self, code_path, topk=3):
        ast = code_to_ast(code_path=code_path)
        nt_seq = ast_to_nt_seq(ast)
        int_seq = to_int_nt_seq(nt_seq)
        topk_token_pairs, topk_pairs_poss = self.model.predict(int_seq, topk=topk)
        print('\nthe token you may want to write is:')
        for index, (token, poss) in enumerate(zip(topk_token_pairs, topk_pairs_poss)):
            n_token, t_token = token
            n_poss, t_poss = poss
            print('top{} n_token:{} with possibility:{:.2f}'.format(index+1, n_token, n_poss))
            print('top{} t_token:{} with possibility:{:.2f}'.format(index+1, t_token, t_poss))
        return topk_token_pairs, topk_pairs_poss


if __name__ == '__main__':
    model = OnlineCompletion()
    model.complete(code_path='js_parser/test.json')

