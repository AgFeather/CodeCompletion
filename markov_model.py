import random

import data_utils
string_processed_data_path = 'processed_data/str_train_data.p'



class Markov_Model(object):

    def __init__(self):
        self.markov_table = {}
        self.max_length = 1

    def create_model(self, token_list, max_depth=1):
        '''
        create a markov model with the depth from 1 to max_depth
        {
            depth1:{
                key1:[value1, value2 ..]
            }
        }
        '''
        for depth in range(1, max_depth+1):
            temp_table = {}
            for index in range(depth, len(token_list)):
                words = tuple(token_list[index-depth:index])
                if words in temp_table.keys():
                    temp_table[words].append(token_list[index])
                else:
                    temp_table.setdefault(words, []).append(token_list[index])
            self.markov_table[depth] = temp_table
        self.max_length = max(self.markov_table.keys())
        return self.markov_table

    def test_model(self, test_token_lists, depth=1):
        correct = 0
        correct_token_list = []
        incorrect_token_list = []

        for tokens in test_token_lists:
            prefix, expection, suffix = data_utils.create_hole(tokens)
            prediction = self.query_test(prefix, suffix)
            if data_utils.token_equals(prediction, expection):
                correct += 1
                correct_token_list.append({'expection': expection, 'prediction': prediction})
            else:
                incorrect_token_list.append({'expection': expection, 'prediction': prediction})
        accuracy = correct / len(test_token_lists)
        return accuracy


    def query_test(self, pre_tokens, suc_tokens, depth=1):
        while(depth>self.max_length):
            depth -= 1
        print('predict markov model with depth: ', depth)
        used_tokens = pre_tokens[len(pre_tokens)-depth:]
        candidate_list = self.markov_table[depth][used_tokens]
        random_index = random.randint(0, len(candidate_list)-1)
        prediction = data_utils.string_to_token(candidate_list[random_index])

        return prediction



if __name__ == '__main__':
    markov_model = Markov_Model()
    string_token_list = data_utils.load_data_with_pickle(string_processed_data_path)
    markov_model.create_model(string_token_list, max_depth=1)
 #   print(markov_table[1].keys())
    test_token_sequences = data_utils.load_data_with_file()
    accuracy = markov_model.test_model(test_token_sequences)