from gensim.models import word2vec
import os
import pickle

from setting import Setting
import utils


"""
Using Word2Vec to pre-train the repesentation of each token
"""

base_setting = Setting()

model_save_path = base_setting.word2vec_save_path
sub_int_train_dir = base_setting.sub_int_train_dir
sub_train_data_dir = base_setting.sub_train_data_dir

num_sub_train_data = base_setting.num_sub_train_data
num_non_terminal = base_setting.num_non_terminal
num_terminal = base_setting.num_terminal
embed_dim = base_setting.word2vec_embed_dim
unknown_token = base_setting.unknown_token

save_to_path = sub_train_data_dir + 'int_for_word2vec/'




class TokenToVec():
    """对输入数据corpus中的每个token训练一个representation vector"""
    def __init__(self):
        # self.model = word2vec.Word2Vec(window=6, size=embed_dim)
        print('WordToVec model has been created...')

    def train(self, dataset):
        # self.model.build_vocab(dataset)
        # self.model.train(dataset, total_examples=self.model.corpus_count, epochs=self.model.iter)
        self.model = word2vec.Word2Vec(dataset, size=300, window=10)
        print('model training finished...')
        self.model.save(model_save_path)
        print('model has saved...')


def load_model():
    model = word2vec.Word2Vec.load(model_save_path)
    return model

def load_all_data():
    all_seq = []
    for file_number in range(1, num_sub_train_data+1):
        file_path = save_to_path + 'int_with_seq{}.json'.format(file_number)
        file = open(file_path, 'rb')
        sub_data = pickle.load(file)
        for nt_seq in sub_data:
            sub_seq = []
            for n_pair, t_pair in nt_seq:
                sub_seq.append(str(n_pair))
                sub_seq.append(str(t_pair + num_non_terminal))  # 为防止nt和tt的index重复，将所有的terminal的index都向后移num_nt位
            all_seq.append(sub_seq)
        print(file_path, 'has been loaded...')
    print('all data has been loaded...')
    return all_seq

def string_to_int_sequence():
    # 将训练集的string nt-sequence转换成int-nt-sequence，并且完全保存各个sub-sequence的结构
    # 也就是说生成的sequence的元素仍然是一个sequence，该sequence表示一个ast。这个sequence的元素才是nt-pair
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = utils.load_dict_parameter()

    def get_subset_data():  # 对每个part的nt_sequence读取并返回，等待进行处理
        for i in range(1, num_sub_train_data + 1):
            data_path = sub_train_data_dir + 'part{}.json'.format(i)
            data = utils.pickle_load(data_path)
            yield (i, data)

    subset_generator = get_subset_data()
    for index, sub_data in subset_generator:
        data_seq = []
        for one_ast in sub_data:
            nt_int_seq = [(nt_token_to_int[n], tt_token_to_int.get(
                t, tt_token_to_int[unknown_token])) for n, t in one_ast]
            data_seq.append(nt_int_seq)
        print('there are {} ast in {} sub dataset'.format(len(data_seq), index))
        one_saved_path = save_to_path + 'int_with_seq{}.json'.format(index)
        utils.pickle_save(one_saved_path, data_seq)
    print('all training data has been processed...')




if __name__ == '__main__':
    step_choice = [0, 1]
    step = step_choice[1]
    if step == 0:
        string_to_int_sequence()
    elif step == 1:
        all_data = load_all_data()
        print(len(all_data))  # number of token:34546856 totally. number of seq:100000 totally.
        model = TokenToVec()
        model.train(all_data)

