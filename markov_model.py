import random

import data_utils
string_processed_data_path = 'processed_data/str_train_data.p'


def markov_model(token_list, max_depth=1):
    '''
    create a markov model with the depth from 1 to max_depth
    {
        depth:{
            key1:[value1, value2 ..]
        }
    }
    '''
    markov_table = {}
    for depth in range(1, max_depth+1):
        temp_table = {}
        for index in range(depth, len(token_list)):
            words = tuple(token_list[index-depth:index])
            if words in temp_table.keys():
                temp_table[words].append(token_list[index])
            else:
                temp_table.setdefault(words, []).append(token_list[index])
        markov_table[depth] = temp_table
    return markov_table


def random_predict(markov_table, token):
    token_length = len(token)
    if token_length > len(markov_table):
        pass
    candidate_list = markov_table[token_length][token]
    random_index = random.randint(0, len(candidate_list)-1)
    return candidate_list[random_index]



if __name__ == '__main__':
    string_token_list = data_utils.load_data_with_pickle(string_processed_data_path)
    markov_table = markov_model(string_token_list, max_depth=1)
    print(markov_table[1].keys())