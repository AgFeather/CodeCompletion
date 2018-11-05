import numpy as np
import pandas as pd
import pickle
import json

js_test_data_dir = 'js_dataset/js_programs_eval.json'
js_train_data_dir = 'js_dataset/js_programs_training.json'


def dataset_split(is_training=True, subset_size=5000):
    if is_training:
        data_path = js_train_data_dir
        total_size = 100000
        saved_to_path = 'split_js_data/train_data/'
    else:
        data_path = js_test_data_dir
        total_size = 50000
        saved_to_path = 'split_js_data/eval_data/'

    file = open(data_path, 'r')
    subset_list = []
    error_count = 0
    for i in range(1, total_size + 1):
        try:
            line = file.readline()
            line = json.loads(line)
        except:
            error_count += 1
        # print('UTF-8 error: {}/{}'.format(error_count, i))
        subset_list.append(line)
        if i % subset_size == 0:
            sub_path = saved_to_path + 'part{}'.format(i // subset_size) + '.json'
            save_file = open(sub_path, 'wb')
            pickle.dump(subset_list, save_file)
            subset_list = []
    print('data seperating finished..., utf-8 error:{}'.format(error_count))


file = open(js_test_data_dir)
line = file.readline()
# line = json.loads(line)
print(line)

# dataset_split()

def load_data_with_pickle(path='split_js_data/train_data/part1.json'):
    file = open(path, 'rb')
    data = pickle.load(file)
    return data

def add_two_bits_info(node, brother_map):
    # 向每个节点添加两bit的额外信息：isTerminal和hasSibling
    if 'children' in node.keys():
        node['isTerminal'] = False
    else:
        node['isTerminal'] = True
    if brother_map.get(node['id'], -1) == -1:
        node['hasSibling'] = False
    else:
        node['hasSilbing'] = True


def bulid_binary_tree(data):
    '''
    transform the AST(one node may have several childNodes) to
    Left-Child-Right-Sibling(LCRS) binary tree.
    '''
    brother_map = {0: -1}
    for index, node in enumerate(data): # 顺序遍历每个AST中的node
        if type(node) != dict and node == 0: # AST中最后添加一个'EOF’标识
            data[index] = 'EOF'
            return data
        # 向每个节点添加两bit的额外信息
        add_two_bits_info(node, brother_map)
        node['right'] = brother_map.get(node['id'], -1)
        # 表示该node为non-terminal
        if 'children' in node.keys():
            child_list = node['children']
            node['left'] = child_list[0] # 构建该node的left node
            for i, bro in enumerate(child_list): # 为该node的所有children构建right sibling
                if i == len(child_list)-1:
                    break
                brother_map[bro] = child_list[i+1]
            node.pop('children')
    return data


ast_example = [{'id': 0, 'type': 'Program', 'children': [1]},
               {'id': 1, 'type': 'ExpressionStatement', 'children': [2]},
               {'id': 2, 'type': 'CallExpression', 'children': [3, 8, 9, 10]},
               {'id': 3, 'type': 'MemberExpression', 'children': [4, 7]},
               {'id': 4, 'type': 'MemberExpression', 'children': [5, 6]},
               {'id': 5, 'type': 'Identifier', 'value': 'CKEDITOR'},
               {'id': 6, 'type': 'Property', 'value': 'plugins'},
               {'id': 7, 'type': 'Property', 'value': 'setLang'},
               {'id': 8, 'type': 'LiteralString', 'value': 'iframe'},
               {'id': 9, 'type': 'LiteralString', 'value': 'ka'},
               {'id': 10, 'type': 'ObjectExpression', 'children': [11, 13, 15, 17, 19]},
               {'id': 11, 'type': 'Property', 'value': 'border', 'children': [12]},
               {'id': 12, 'type': 'LiteralString', 'value': 'ჩარჩოს გამოჩენა'},
               {'id': 13, 'type': 'Property', 'value': 'noUrl', 'children': [14]},
               {'id': 14, 'type': 'LiteralString', 'value': 'აკრიფეთ iframe-ის URL'},
               {'id': 15, 'type': 'Property', 'value': 'scrolling', 'children': [16]},
               {'id': 16, 'type': 'LiteralString', 'value': 'გადახვევის ზოლების დაშვება'},
               {'id': 17, 'type': 'Property', 'value': 'title', 'children': [18]},
               {'id': 18, 'type': 'LiteralString', 'value': 'IFrame-ის პარამეტრები'},
               {'id': 19, 'type': 'Property', 'value': 'toolbar', 'children': [20]},
               {'id': 20, 'type': 'LiteralString', 'value': 'IFrame'}, 0]

from collections import Counter

output = []
token2int = {}
int2token = {}
terminalCountMap = Counter()
nonTerminalSet = set()


def node_to_string(node):
    '''transform a node to a string representation'''
    if node == 'EMPTY':
        string_node = 'EMPTY'

    # 如果node为terminal，将其加入Counter中统计出现频率
    if node['isTerminal']:
        string_node = str(node['type']) + '$$' + str(node['value'])
        terminalCountMap[string_node] += 1

    # 如果是non-terminal，将其加入non-terminal set中，统计种类个数
    else:
        try:
            string_node = str(node['type']) + '$$' + \
                          str(node['hasSibling']) + '$$' + \
                          str(node['isTerminal'])  # 重新考察两bit信息
        except:
            print('ERROR')
            print(len(nonTerminalSet))
            print(node)
        if 'value' in node.keys():  # 注意，有些non-terminal也包含value，需要加入
            string_node += '$$' + str(node['value'])
        nonTerminalSet.add(string_node)
    return string_node


def in_order_traversal(data, index):
    node = data[index]
    if 'left' in node.keys():
        in_order_traversal(data, node['left'])
    print(node)
    # 如果该节点为non-terminal，则构建NT-pair并加入到sequence中。
    if 'isTerminal' in node.keys() and node['isTerminal'] == False:
        '''如果该node是non-terminal
        如果该non-terminal包含一个terminal 子节点，则和该子节点组成NT_pair保存在output中
        否则将NT_pair的T设为字符串EMPTY'''
        N_pair = node_to_string(node)
        T_pair = None
        if data[node['left']]['isTerminal'] == True:
            assert data[node['left']]['id'] == node['left']
            T_pair = node_to_string(data[node['left']])
        else:
            T_pair = node_to_string('EMPTY')
        NT_pair = (N_pair, T_pair)
        output.append(NT_pair)
    else:
        node_to_string(node)
    # 遍历right side
    if node['right'] != -1:
        in_order_traversal(data, node['right'])


def AST_to_seq(data):
    bi_tree = bulid_binary_tree(data)
    #    print(bi_tree)
    in_order_traversal(bi_tree, 0)




if __name__ == '__main__':

    AST_to_seq(ast_example)
    print(output)