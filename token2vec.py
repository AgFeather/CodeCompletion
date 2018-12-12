from gensim.models import word2vec
import os
import pickle

from setting import Setting


"""
Using Word2Vec to pre-train the repesentation of each token
"""

base_setting = Setting()
embed_dim = base_setting.word2vec_embed_dim
model_save_path = base_setting.word2vec_save_path
sub_int_train_dir = base_setting.sub_int_train_dir
num_subset_train_data = base_setting.num_sub_train_data
num_non_terminal = base_setting.num_non_terminal
num_terminal = base_setting.num_terminal

class TokenToVec():
    """对输入数据corpus中的每个token训练一个representation vector"""
    def __init__(self):
        pass

    def train(self, dataset):
        self.model = word2vec.Word2Vec(dataset, size=embed_dim)
        self.model.save(model_save_path)
        print('model has saved...')



def load_model():
    model = word2vec.Word2Vec.load(model_save_path)
    return model

def load_all_data():
    all_data = []
    for file_number in range(1, 1+1):
        file_path = sub_int_train_dir + 'int_part{}.json'.format(file_number)
        file = open(file_path, 'rb')
        sub_data = pickle.load(file)
        for n_pair, t_pair in sub_data:
            all_data.append(n_pair)
            all_data.append(t_pair + num_non_terminal)  # 为防止nt和tt的index重复，将所有的terminal的index都向后移num_nt位
        print(file_path, 'has been loaded...')
    print('all data has been loaded...')
    return all_data



if __name__ == '__main__':
    all_data = load_all_data()
    print(all_data[:10])
    model = TokenToVec()
    model.train(all_data)

