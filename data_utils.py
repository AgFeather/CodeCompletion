import os
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import random


train_dir = 'dataset/programs_800/'
query_dir = 'dataset/programs_200/'
model_dir = 'saved_model/model_parameter'


def load_tokens(train_flag=False, is_simplify=True):
    '''
    load token sequence data from input path: token_dir.
    return a list whose elements are lists of a token sequence
    读入所有code文件，并返回list，list每个元素是一个token
    '''
    if train_flag:
        token_dir = train_dir
    else:
        token_dir = query_dir
    token_files = []  # stored the file's path which ends with 'tokens.json'
    for f in os.listdir(token_dir):
        file_path = os.path.join(token_dir, f)
        if os.path.isfile(file_path) and f.endswith('_tokens.json'):
            token_files.append(file_path)

    token_list = []
    for f in token_files:
        token_list.extend(json.load(open(f, encoding='utf-8')))

    if is_simplify:
        for token in token_list:
            simplify_token(token)
    else:
        pass

    return token_list

def simplify_token(token):
    '''
    Because there are too many values for type: "Identifier", "String", "Numeric", "RegularExpression"
    So this function transforms these types of variables to a common value
    '''
    if token['type'] == 'Identifier':
        token['value'] = 'id'
    elif token['type'] == 'Numeric':
        token['value'] = '1'
    elif token['type'] == 'String':
        token['value'] = 'string'
    elif token['type'] == 'RegularExpression':
        token['value'] = 'RExpression'
    else:
        pass



# data processing functions
def token_to_string(token):
    return token['type'] + '~$$~' + token['value']

def string_to_token(string):
    tokens = string.split('~$$~')
    return {'type': tokens[0], 'value': tokens[1]}

def get_token_set(token_list):
    token_set = set()
    for token in token_list:
        token_set.add(token_to_string(token))
    return token_set



def data_processing(token_list):
    string_token_list = []
    for token in token_list:
        string_token_list.append(token_to_string(token))
    print(len(string_token_list))#1713662
    pickle.dump(string_token_list, open('string_token_list.p', 'wb'))
    return string_token_list




def load_data_with_file(data_dir=query_dir, is_simplify=True):
    '''
    读取指定数据集的文件，返回一个list，并且每个list是一个token sequence
    :param data_dir:
    :return:
    '''
    token_files = []  # stored the file's path which ends with 'tokens.json'
    for f in os.listdir(data_dir):
        file_path = os.path.join(data_dir, f)
        if os.path.isfile(file_path) and f.endswith('_tokens.json'):
            token_files.append(file_path)

    # load to a list, element is a token sequence of source code
    token_lists = [json.load(open(f, encoding='utf-8')) for f in token_files]

    if is_simplify:
        for token_sequence in token_lists:
            for token in token_sequence:
                simplify_token(token)
    else:
        pass

    return token_lists



# function for test
def create_hole(tokens, max_hole_size = 2):
    '''
    input: a tokens sequence of source code and max_hole_size
    return: hole token to be predicted (expection)
            token sequence before the hole(prefix)
            token sequence after the hole(suffix)
    '''
    hole_size = min(random.randint(1, max_hole_size), len(tokens) - 1)
    hole_start_index = random.randint(1, len(tokens) - hole_size)
    hole_end_index = hole_start_index + hole_size
    prefix = tokens[0 : hole_start_index]
    expection = tokens[hole_start_index : hole_end_index]
    suffix = tokens[hole_end_index : 0]
    return prefix, expection, suffix

def token_equals(token1, token2):
    '''
    Determining whether input two tokens are equal or not
    '''
    if len(token1) != len(token2):
        return False
    for index, t1 in enumerate(token1):
        t2 = token2[index]
        if t1['type'] != t2['type'] or t1['value'] != t2['value']:
            return False
    return True


if __name__ == '__main__':
    test_data = load_tokens(True)
 #   print(test_data[:5])

    test_data = data_processing(test_data)
  #  print(test_data[:3])   #['Identifier~$$~id', 'Punctuator~$$~.', 'Identifier~$$~id']
