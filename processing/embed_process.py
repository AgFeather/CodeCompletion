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

data_parameter_dir = '../' + base_setting.data_parameter_dir

nt_train_pair_dir = 'js_dataset/train_pair_data/nt_train_pair/'
tt_train_pair_dir = 'js_dataset/train_pair_data/tt_train_pair/'

num_sub_valid_data = base_setting.num_sub_valid_data
num_sub_train_data = base_setting.num_sub_train_data
num_sub_test_data = base_setting.num_sub_test_data

most_common_termial_num = 30000
unknown_token = base_setting.unknown_token
time_steps = base_setting.time_steps


parameter_file = open(data_parameter_dir, 'rb')
tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = pickle.load(parameter_file)


def dataset_split(subset_size=5000):
    """读取原始AST数据集，并将其分割成多个subset data
    对每个AST，生成多个training pair"""

    data_path = js_train_data_dir
    total_size = 100000

    file = open(data_path, 'r')
    nt_train_pairs_list = []
    tt_train_pairs_list = []
    for i in range(1, total_size + 1):
        try:
            line = file.readline()  # read a lind from file(one ast)
            ast = json.loads(line)  # transform it to json format
            nt_train_pairs, tt_train_pairs = generate_train_pair(ast)
        except UnicodeDecodeError as error:  # arise by readline
            print(error)
        except JSONDecodeError as error:  # arise by json_load
            print(error)
        except RecursionError as error:
            print(error)
        except BaseException:
            print('other unknown error, plesae check the code')
        else:
            nt_train_pairs_list.extend(nt_train_pairs)
            tt_train_pairs_list.extend(tt_train_pairs)

        if i % subset_size == 0:  # 当读入的ast已经等于给定的subset的大小时
            nt_pair_path = nt_train_pair_dir + \
                'part{}'.format(i // subset_size) + '.json'
            tt_pair_path = tt_train_pair_dir + \
                'part{}'.format(i // subset_size) + '.json'
            pickle_save(nt_pair_path, nt_train_pairs_list)
            pickle_save(tt_pair_path, tt_train_pairs_list)
            nt_train_pairs_list = []
            tt_train_pairs_list = []


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

        if 'children' in node.keys():  # 说明是non-terminal node
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


def string_to_int(string_node, tt_or_nt):
    if tt_or_nt == 'tt':
        if string_node not in tt_token_to_int.keys():
            int_node = tt_token_to_int[unknown_token]
        else:
            int_node = tt_token_to_int[string_node]
        return int_node
    elif tt_or_nt == 'nt':
        return nt_token_to_int[string_node]
    else:
        raise AttributeError


def has_terminal_child(ast, node):  # 判断给定节点是否包含Teminal node child
    for child in node['children']:
        if ast[child]['isTerminal']:
            return True
    return False


def get_nt_train_pair(ast, node, nt_v_dim):
    nt_train_ny = []
    nt_train_ty = []
    nt_train_x = None
    try:
        string_train_x = node_to_string(node)
        nt_train_x = string_to_int(string_train_x, 'nt')
    except:
        print('ERROR2')
    nt_train_ny_index = node['father'][-nt_v_dim:]  # 对于一个nt-node所有father context的index

    if has_terminal_child(ast, node):  # 找到所有Teminal context
        nt_train_ty_index = [child for child in node['children'] if ast[child]['isTerminal']]
    else:
        nt_train_ty_index = 'EMPTY'


    try:
        for ny_index in nt_train_ny_index:
            # 将所有的father context 转换为对应string, 再转换为对应的int
            string_node = node_to_string(ast[ny_index])
            nt_train_ny.append(string_to_int(string_node, 'nt'))

        if isinstance(nt_train_ty_index, str):  # 说明是empty
            print(node)
            nt_train_ty.append(string_to_int(nt_train_ty_index, 'tt'))
        else:
            for ty_index in nt_train_ty_index:
                # 将所有terminal context转换为对应的string, 再转换为对应的int
                string_node = node_to_string(ast[ty_index])
                nt_train_ty.append(string_to_int(string_node, 'tt'))
    except KeyError:
        raise KeyError

    nt_train_pair = (nt_train_x, nt_train_ny, nt_train_ty)
    return nt_train_pair


def get_tt_train_pair(ast, node, tt_v_dim, tt_h_dim):
    # 根据输入的nt node的所有children terminal node构建 tt training pair
    train_x = None
    train_ny = []
    train_ty = []
    train_pair_list = []
    for index, child_index in enumerate(node['children']):
        child = ast[child_index]
        if child['isTerminal']:
            # 该child node为terminal node，构建一个training pair
            try:
                string_train_x = node_to_string(child)
                train_x = string_to_int(string_train_x, 'tt')
            except:
                print('ERROR1')
            train_ny_index = child['father'][-tt_v_dim:] # 构建指定的non-terminal father context
            train_ty_index = []
            for i in range(index - tt_h_dim, index + tt_h_dim + 1):
                if i == index or i >= len(node['children']) or i < 0:
                    continue
                terminal_context_index = node['children'][i]
                if ast[terminal_context_index]['isTerminal']:
                    train_ty_index.append(terminal_context_index)

            try:
                for ny_index in train_ny_index:
                    string_node = node_to_string(ast[ny_index])
                    train_ny.append(string_to_int(string_node, 'nt'))
                for ty_index in train_ty_index:
                    string_node = node_to_string(ast[ty_index])
                    train_ty.append(string_to_int(string_node, 'tt'))
            except KeyError:
                raise KeyError

            if len(train_ty) == 0: # 当前的train x没有 terminal context，所以添加一个EMTPY node
                train_ty.append(string_to_int('EMPTY', 'tt'))

            train_pair = (train_x, train_ny, train_ty)
            train_pair_list.append(train_pair)
    return train_pair_list


# todo:检查生成的train pair是否正确
def generate_train_pair(ast, nt_v_dim=2, tt_v_dim=2, tt_h_dim=2):
    """生成nt-train-pair 和 tt-train-pair"""
    info_ast = add_info(ast)
    nt_train_pair_list = []
    tt_train_pair_list = []
    for node in info_ast:
        if isinstance(node, str):  # 过滤掉最后的EOF
            continue

        if not node['isTerminal']:
            # 该node为nt-node，所以构建一个nt_train_pair
            try:
                nt_train_pair = get_nt_train_pair(ast, node, nt_v_dim)
            except KeyError:
                print("nt key error")
            else:
                nt_train_pair_list.append(nt_train_pair)

            if has_terminal_child(ast, node):
                # 说明该nt-node包含terminal child node，将所有terminal node也用来构建train pair
                try:
                    tt_train_pair = get_tt_train_pair(ast, node, tt_v_dim, tt_h_dim)
                except KeyError:
                    print('tt key error')
                else:
                    tt_train_pair_list.extend(tt_train_pair)

    return nt_train_pair_list, tt_train_pair_list





if __name__ == '__main__':
    import examples
    ast = examples.ast_example
    # info_ast = add_info(ast)
    # print(info_ast)
    nt_train_pair_list, tt_train_pair_list = generate_train_pair(ast)
    print(nt_train_pair_list)
    print(tt_train_pair_list)
