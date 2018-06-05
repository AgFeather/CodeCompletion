import os
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder


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
    Because there are too many values for type: "Identifier", "String", "Numeric",
    So this function transforms these types of variables to a common value
    '''
    if token['type'] == 'Identifier':
        token['value'] = 'id'
    elif token['type'] == 'Numeric':
        token['value'] = '1'
    elif token['type'] == 'String':
        token['value'] = 'string'
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


import pickle
def data_processing(token_list):
    string_token_list = []
    for token in token_list:
        string_token_list.append(token_to_string(token))
    print(len(string_token_list))#1713662
    pickle.dump(string_token_list, open('string_token_list.p', 'wb'))
    return string_token_list


if __name__ == '__main__':
    test_data = load_tokens(True)
 #   print(test_data[:5])

    test_data = data_processing(test_data)
  #  print(test_data[:3])   #['Identifier~$$~id', 'Punctuator~$$~.', 'Identifier~$$~id']
