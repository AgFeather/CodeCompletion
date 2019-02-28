import numpy as np
import sys


import pickle
import json
from collections import Counter
from json.decoder import JSONDecodeError

from setting import Setting
from utils import pickle_save

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
            pickle_save(sub_path, train_pairs_list)  # 将subset dataset保存
            train_pairs_list = []


def add_two_bits_info(ast, node, brother_map):
    # 向每个节点添加两bit的额外信息：hasNonTerminalChild和hasSibling
    node['hasNonTerminalChild'] = False
    for child_index in node['children']:
        if 'children' in ast[child_index]:
            node['hasNonTerminalChild'] = True
            break
    if brother_map.get(node['id'], -1) == -1:
        node['hasSibling'] = False
    else:
        node['hasSibling'] = True


def add_info(ast):
    """给每个节点添加一个father list"""
    brother_map = {0: -1}
    ast[0]['father'] = []  # 给根节点添加一个空的father list
    info_ast = []
    for index, node in enumerate(ast):

        if not isinstance(node, dict) and node == 0:  # AST中最后添加一个'EOF’标识
            ast[index] = 'EOF'
            break

        if 'children' in node.keys():
            for child_index in node['children']:
                child = ast[child_index]
                child['father'] = []
                child['father'].extend(node['father'])
                child['father'].append(node['id'])

        if 'children' in node.keys():
            node['isTerminal'] = False
            add_two_bits_info(ast, node, brother_map)  # 向节点添加额外两bit信息
            child_list = node['children']
            for i, bro in enumerate(child_list):
                if i == len(child_list) - 1:
                    break
                brother_map[bro] = child_list[i + 1]
        else:
            node['isTerminal'] = True
    return ast


def node_to_string(node):
    # 将一个node转换为string
    if node == 'EMPTY':
        string_node = 'EMPTY'
    elif node['isTerminal']:  # 如果node为terminal
        string_node = str(node['type'])
        if 'value' in node.keys():
            # Note:There are some tokens(like:break .etc）do not contains 'value'
            string_node += '=$$=' + str(node['value'])

    else:  # 如果是non-terminal

        string_node = str(node['type']) + '=$$=' + \
            str(node['hasSibling']) + '=$$=' + \
            str(node['hasNonTerminalChild'])  # 有些non-terminal包含value，探索该value的意义？（value种类非常多）

    return string_node

def string_to_int(node):
    pass

def generate_train_pair(ast, nt_v_dim=2, tt_v_dim=2, tt_h_dim=2):
    info_ast = add_info(ast)
    nt_train_pair_list = []
    tt_train_pair_list = []
    for node in info_ast:
        if not node['isTerminal']:

            # 该node为nt-node，所以构建一个nt_train_pair
            nt_train_x = node_to_string(node)
            nt_train_ny_index = node['father'][-nt_v_dim:]  # 对于一个nt-node所有father context的index
            if node['hasNonTerminalChild']:
                nt_train_ty_index = [child for child in node['children'] if ast[child]['isTerminal']]
            else:
                nt_train_ty_index = []  #todo:修改如果该nt-node不包含terminal node时的表示
            nt_train_ny = []
            nt_train_ty = []
            for ny_index in nt_train_ny_index:
                # 将所有的father context 转换为对应string
                string_node = node_to_string(ast[ny_index])
                # 再转换为对应的int
                nt_train_ny.append(string_to_int(string_node))
            for ty_index in nt_train_ty_index:
                # 将所有terminal context转换为对应的string
                string_node = node_to_string(ast[ty_index])
                nt_train_ty.append(string_to_int(string_node))

            nt_train_pair = (nt_train_x, nt_train_ny, nt_train_ty)
            nt_train_pair_list.append(nt_train_pair)

            if node['hasNonTerminalChild']:
                # 说明该nt-node包含terminal child node，将所有terminal node也用来构建train pair
                for child in node['children']:
                    if ast[child]['isTerminal']:
                        tt_train_x = node_to_string(ast[child])


