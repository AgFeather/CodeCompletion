import os
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import random


train_dir = 'dataset/programs_800/'
query_dir = 'dataset/programs_200/'
model_dir = 'saved_model/model_parameter'

str_processed_train_data_path = 'processed_data/str_train_data.p'
vec_processed_train_data_path = 'processed_data/vec_train_data.p'
train_data_parameter_path = 'processed_data/train_parameter.p'

str_processed_test_data_path = 'processed_data/str_test_data.p'
vec_processed_test_data_path = 'processed_data/vec_test_data.p'
test_data_parameter_path = 'processed_data/test_parameter.p'

num_train_token = 86 #在训练集中token的种类
num_test_token = 74 #测试集中token的种类


def load_tokens(train_flag=False, is_simplify=True)->list:
    '''
    load token sequence data from input path: token_dir.
    return a list whose elements are lists of a token sequence
    读入文件夹中所有code文件，并返回list，list每个元素是一个token（去除掉文件信息）
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
    Because there are too many values for type:
    "Identifier", "String", "Numeric", "RegularExpression"
    So this function convert the value of these types to a same value
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
    '''
    获取输入token list的token集合
    '''
    token_set = set()
    for token in token_list:
        token_set.add(token_to_string(token))
    return token_set


def load_data_with_file(data_dir=query_dir, is_simplify=True):
    '''
    读取指定数据集的文件，
    返回一个list，并且每个list是一个token sequence（对应一个source code文件）
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





def one_hot_encoding(string, string2int, num_tokens=num_train_token):
    vector = [0] * num_tokens
    vector[string2int[string]] = 1
    return vector


def data_process_save(is_training_data=True):
    '''
    读取原始token sequence数据，经过处理后保存在本地，一共保存三个文件
    str_processed_train_data：所有未经过one_hot encoding的token list（去除掉单个文件信息）
    vec_processed_train_data：对所有token进行one hot encoding，组成token list然后保存到本地
    train_data_parameter_path：将int2string和string2int以及token set保存

    '''
    token_list = load_tokens(is_training_data)
    string_token_list = []
    vector_token_list = []
    token_set = list(get_token_set(token_list))
    string2int = {token:i for i, token in enumerate(token_set)}
    int2string = {i:token for i, token in enumerate(token_set)}
    for token in token_list:
        string_token = token_to_string(token)
        token_vect = one_hot_encoding(string_token, string2int)
        string_token_list.append(string_token)
        vector_token_list.append(token_vect)

    print(len(string_token_list))#1713662
    if is_training_data:
        pickle.dump(string_token_list, open(str_processed_train_data_path, 'wb'))
        pickle.dump(vector_token_list, open(vec_processed_train_data_path, 'wb'))
        pickle.dump((string2int, int2string, token_set), open(train_data_parameter_path, 'wb'))
    else:
        pickle.dump(string_token_list, open(str_processed_test_data_path, 'wb'))
        pickle.dump(vector_token_list, open(vec_processed_test_data_path, 'wb'))
        pickle.dump((string2int, int2string, token_set), open(test_data_parameter_path, 'wb'))





def load_data_with_pickle(path):
    return pickle.load(open(path, 'rb'))


# function for test
def create_hole(tokens, hole_size = 1):
    '''
    input: a tokens sequence of source code and max_hole_size
    return: hole token to be predicted (expection)
            token sequence before the hole(prefix)
            token sequence after the hole(suffix)
    '''
    hole_start_index = random.randint(1, len(tokens) - hole_size)
    hole_end_index = hole_start_index + hole_size
    prefix = tokens[0 : hole_start_index]
    expection = tokens[hole_start_index : hole_end_index]
    suffix = tokens[hole_end_index : -1]
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


def save_x_y_train_data():
    token_list = load_tokens(train_flag=True)
    x_data = []
    y_data = []
    token_set = list(get_token_set(token_list))
    string2int = {token: i for i, token in enumerate(token_set)}
    int2string = {i: token for i, token in enumerate(token_set)}
    for index, token in enumerate(token_list):
        if index > 0:
            prev_string_token = token_to_string(token_list[index - 1])
            prev_vect_token = one_hot_encoding(prev_string_token, string2int)
            x_data.append(prev_vect_token)

            curr_string_token = token_to_string(token)
            curr_vect_token = one_hot_encoding(curr_string_token, string2int)
            y_data.append(curr_vect_token)
    # print(len(x_data))1713661

    # print(len(y_data))1713661
    # print(len(x_data[0]))86
    # print(len(y_data[0]))86
    # print(len(string2int))86
    # print(len(token_set))86

    pickle.dump(x_data, open('processed_data/x_train_data.p', 'wb'))
    pickle.dump(y_data, open('processed_data/y_train_data.p', 'wb'))
    pickle.dump((token_set, string2int, int2string), open('processed_data/x_y_parameter.p', 'wb'))


if __name__ == '__main__':

    # dataset = load_tokens(False, True)
    # print(len(get_token_set(dataset)))
   # data_process_save(is_training_data=True)
    save_x_y_train_data()



    pass

   # test_data = data_process_save()
  #  print(test_data[:3])   #['Identifier~$$~id', 'Punctuator~$$~.', 'Identifier~$$~id']
