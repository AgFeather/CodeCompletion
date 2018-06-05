import numpy as np
import pandas as pd

import data_utils

train_dir = 'dataset/programs_800'
test_dir = 'dataset/programs_200'


def create_token_table(dataset):
    token_list = []
    for token in dataset:
        token_list.append(data_utils.token_to_string(token))
    token_set = data_utils.get_token_set(dataset)
    num_token = len(token_set)
    token_cate_list = list(token_set)

    token_table = pd.DataFrame(
        np.zeros(num_token * num_token).reshape(num_token, num_token), index=token_cate_list, columns=token_cate_list)

    for index, token in enumerate(token_list):
        if index > 0:
            cur_token = token
            pre_token = token_list[index - 1]
            token_table[cur_token][pre_token] += 1

    return token_table



def test_benchmark(token_table):
    test_token_sequence = data_utils.load_data_with_file(test_dir)


if __name__ == '__main__':
    dataset = data_utils.load_tokens(True)
    token_table = create_token_table(dataset)

