import numpy as np
import pandas as pd

import data_utils

train_dir = 'dataset/programs_800'
test_dir = 'dataset/programs_200'
saved_token_table_path = 'saved_model/token_table.csv'


def create_token_table(dataset):
    token_list = []
    for token in dataset:
        token_list.append(data_utils.token_to_string(token))
    token_set = data_utils.get_token_set(dataset)
    num_token = len(token_set)
    token_cate_list = list(token_set)
    print(num_token)
    token_table = pd.DataFrame(
        np.zeros(num_token * num_token).reshape(num_token, num_token), index=token_cate_list, columns=token_cate_list)

    for index, token in enumerate(token_list):
        if index > 0:
            cur_token = token
            pre_token = token_list[index - 1]
            token_table[cur_token][pre_token] += 1

    return token_table



def test_benchmark(token_table):
    test_data = data_utils.load_data_with_file(test_dir)
    num_accurate = 0
    mistake = []

    for token_sequence in test_data:
        prefix, expection, suffix = data_utils.create_hole(token_sequence, 1)
        pre_token = prefix[-1]
        pre_token = data_utils.token_to_string(pre_token)
#        print(len(expection))
        expection = data_utils.token_to_string(expection[0])
        prediction_list = token_table[pre_token]
        prediction = prediction_list.argmax()
        if prediction == expection:
            num_accurate += 1
        else:
            miss_token = {'pred':prediction, 'expc':expection}
            mistake.append(miss_token)
 #   accuracy /= len(test_data)
 #    print(mistake[:10])
 #    print(len(mistake))
 #    print(len(test_data), num_accurate)
    return num_accurate/len(test_data)


def save_table(token_table):
    token_table.to_csv(saved_token_table_path)

def load_table():
    return pd.read_csv(saved_token_table_path, index_col=0)#index_col表示已第几列为index，默认为None，会自动加上range的为index


if __name__ == '__main__':

    use_saved_table = True
    if use_saved_table:
        token_table = load_table()
    else:
        dataset = data_utils.load_tokens(True)
        token_table = create_token_table(dataset)


    accuracy = test_benchmark(token_table)
    print(accuracy)

