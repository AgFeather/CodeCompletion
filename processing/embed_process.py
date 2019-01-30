import numpy as np
import sys


import pickle
import json
from collections import Counter
from json.decoder import JSONDecodeError

from setting import Setting

base_setting = Setting()


js_train_data_dir = base_setting.origin_train_data_dir

data_parameter_dir = base_setting.data_parameter_dir

train_pair_dir = 'js_dataset/train_pair_data/'

num_sub_valid_data = base_setting.num_sub_valid_data
num_sub_train_data = base_setting.num_sub_train_data
num_sub_test_data = base_setting.num_sub_test_data

most_common_termial_num = 30000
unknown_token = base_setting.unknown_token
time_steps = base_setting.time_steps



def dataset_split(subset_size=5000):
    """读取原始AST数据集，并将其分割成多个subset data
    对每个AST，生成多个training pair"""

    data_path = js_train_data_dir
    total_size = 100000
    saved_to_path = train_pair_dir

    file = open(data_path, 'r')
    train_pairs_list = []
    for i in range(1, total_size + 1):
        try:
            line = file.readline()  # read a lind from file(one ast)
            ast = json.loads(line)  # transform it to json format
            one_ast_pairs = generate_train_pair(ast)
        except UnicodeDecodeError as error:  # arise by readline
            print(error)
        except JSONDecodeError as error:  # arise by json_load
            print(error)
        except RecursionError as error:
            print(error)
        except BaseException:
            print('other unknown error, plesae check the code')
        else:
            train_pairs_list.append(one_ast_pairs)

        if i % subset_size == 0:  # 当读入的ast已经等于给定的subset的大小时
            sub_path = saved_to_path + \
                'part{}'.format(i // subset_size) + '.json'
            pickle_save(sub_path, subset_list)  # 将subset dataset保存
            subset_list = []



def generate_train_pair(ast):
    train_pairs = []
