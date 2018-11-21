import random
from collections import Counter
import data_utils
string_processed_data_path = 'processed_data/str_train_data.p'

class Markov_Model(object):

    def __init__(self, max_length=1, is_most=False):
        self.markov_table = {}
        self.max_length = 1
        self.is_most = False

    def create_model(self, token_list, max_depth=1, is_most=False):
        '''
        create a markov model with the depth from 1 to max_depth
        {
            depth1:{
                key1:[value1, value2 ..]
            }
        }
        '''
        self.is_most = is_most
        self.max_length = max_depth
        for depth in range(1, max_depth+1):
            temp_table = {}
            for index in range(depth, len(token_list)):
                words = tuple(token_list[index-depth:index])
                if words in temp_table.keys():
                    temp_table[words].append(token_list[index])
                else:
                    temp_table.setdefault(words, []).append(token_list[index])
            if is_most:
                for key,value in temp_table.items():
                    temp = Counter(value).most_common(1)[0][0]
                    temp_table[key] = temp
                self.markov_table[depth] = temp_table
            else:
                self.markov_table[depth] = temp_table
        return self.markov_table

    def test_model(self, test_token_lists, depth=1):
        correct = 0
        correct_token_list = []
        incorrect_token_list = []

        for tokens in test_token_lists:
            prefix, expection, suffix = data_utils.create_hole(tokens)
            prediction = self.query_test(prefix, depth=depth)
            if prediction['type']==expection[0]['type'] and prediction['value'] == expection[0]['value']:
                correct += 1
                correct_token_list.append({'expection': expection, 'prediction': prediction})
            else:
                incorrect_token_list.append({'expection': expection, 'prediction': prediction})
        accuracy = correct / len(test_token_lists)
        return accuracy


    def query_test(self, pre_tokens, depth=1):
        while(depth>self.max_length):
            depth -= 1
        used_tokens = pre_tokens[-depth:]
        proceed_tokens = []
        for token in used_tokens:
            proceed_tokens.append(data_utils.token_to_string(token))
        proceed_tokens = tuple(proceed_tokens)
        while proceed_tokens not in self.markov_table[depth].keys() and depth > 1:
            depth -= 1
            proceed_tokens = tuple(proceed_tokens[-depth:])

        if self.is_most:
            candidate = self.markov_table[depth][proceed_tokens]
        else:
            candidate_list = self.markov_table[depth][proceed_tokens]
            random_index = random.randint(0, len(candidate_list)-1)
            candidate = candidate_list[random_index]
        prediction = data_utils.string_to_token(candidate)
        return prediction

if __name__ == '__main__':
    string_token_list = data_utils.load_data_with_pickle(string_processed_data_path)
    markov_model = Markov_Model()
    markov_table = markov_model.create_model(string_token_list, max_depth=6, is_most=True)
    test_token_sequences = data_utils.load_data_with_file()
    accuracy = 0.0
    test_epoch = 10
    for i in range(test_epoch):
        accuracy += markov_model.test_model(test_token_sequences, depth=6)
    accuracy /= test_epoch
    print(accuracy)