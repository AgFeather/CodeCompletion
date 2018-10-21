import numpy as np
import pandas as pd
import pickle

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
    for i in range(1, total_size + 1):
        line = file.readline()
        subset_list.append(line)
        if i % total_size == 0:
            sub_path = saved_to_path + 'part{}'.format(i // total_size) + 'json'
            save_file = open(sub_path, 'wb')
            pickle.dump(subset_list, save_file)
            subset_list = []
    print('data seperating finished...')





example = [{'id': 0, 'child': [1, 2, 3]},
           {'id': 1, 'child': [4, 5, 6]}, {'id': 2, 'child': [7]}, {'id': 3, 'child': [8, 9]},
           {'id': 4}, {'id': 5}, {'id': 6}, {'id': 7}, {'id': 8}, {'id': 9}]

def binary_tree(data):
    brother_map = {1: []}
    for node in data:
        node['right'] = brother_map.get(node['id'], [])
        if 'child' in node.keys():
            brother_map[node['child'][0]] = node['child'][1:]
            node['left'] = node['child'][0]
    return data


def bulid_tree_seq(data):
    bi_tree = binary_tree(data)
    help(bi_tree, 0)


output = []
def help(data, index):
    if 'left' in data[index].keys():
        help(data, data[index]['left'])
    brother_list = data[index]['right']
    output.append(data[index])
    for node_index in brother_list:
        help(data, node_index)


bulid_tree_seq(example)
print(output)


