from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import PathLineSentences
import pickle

from setting import Setting
import utils


"""
Using Word2Vec to pre-train the repesentation of each token
"""

base_setting = Setting()

model_save_path = base_setting.word2vec_save_path
sub_train_data_dir = base_setting.sub_train_data_dir

num_sub_train_data = base_setting.num_sub_train_data
num_non_terminal = base_setting.num_non_terminal
num_terminal = base_setting.num_terminal
embed_dim = base_setting.word2vec_embed_dim
unknown_token = base_setting.unknown_token

wor2vec_data_save_dir = sub_train_data_dir + 'int_for_word2vec/'
repre_matrix_dir = base_setting.temp_info + 'token2vec_repre_matrix.p'



class TokenToVec():
    """对输入数据corpus中的每个token训练一个representation vector"""
    def __init__(self):
        pass

    def train(self, data_path):
        self.dataset = PathLineSentences(data_path)
        print('data has been loaded...')
        print('Token2Vec model is training...')
        self.model = Word2Vec(self.dataset, size=300, window=20, min_count=1, iter=6)
        self.model.save(model_save_path)
        print('WordToVec model has been trained and saved...')

    def load_model(self):
        model = Word2Vec.load(model_save_path)
        return model

    def get_token_representation_matrix(self):
        """加载已经训练好的word2vec模型，并将所有non-terminal的矩阵表示和所有terminal的矩阵表示提取并返回"""
        model = self.load_model()
        nt_represent_matrix = []
        tt_represent_matrix = []
        print(model.wv)
        for i in range(num_terminal + num_non_terminal):
            vector = model[str(i)]
            if i < num_non_terminal:  # < 123 (0到122为non-terminal)
                nt_represent_matrix.append(vector)
            else:
                tt_represent_matrix.append(vector)
        pickle.dump([nt_represent_matrix, tt_represent_matrix], open('repre_matrix_dir', 'wb'))
        print('token2vec representation matrix has been saved...')
        return nt_represent_matrix, tt_represent_matrix


def get_sub_dataet():
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
        data_seq = ''
        for one_ast in sub_data:
            for n_pair, t_pair in one_ast:
                data_seq += str(nt_token_to_int[n_pair]) + ' '
                data_seq += str(tt_token_to_int.get(t_pair, tt_token_to_int[unknown_token]) + num_non_terminal) + ' '
            data_seq += '\n'
        one_save_path = wor2vec_data_save_dir + 'int_with_seq{}.txt'.format(index)
        file = open(one_save_path, 'w', encoding='utf-8')
        file.write(data_seq)
        print(one_save_path, ' has been saved...')

if __name__ == '__main__':
    step_choice = ['data_processing', 'model training', 'represent matrix']
    step = step_choice[2]
    if step == 'data_processing':
        get_sub_dataet()
    elif step == 'model training':
        model = TokenToVec()
        model.train(wor2vec_data_save_dir)
    elif step == 'represent matrix':
        model = TokenToVec()
        model.get_token_representation_matrix()


