import tensorflow as tf
import pickle

import node2vec_tt
import setting

embed_setting = setting.Setting()
num_ntoken = embed_setting.num_non_terminal
num_ttoken = embed_setting.num_terminal

class NodeEmbedding(object):
    def __init__(self):
        self.model = node2vec_tt.NodeToVec_TT(num_ntoken, num_ttoken)
        self.session = tf.Session()
        trained_model_path = '../trained_model/node2vec_tt/'
        checkpoints_path = tf.train.latest_checkpoint(trained_model_path)
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoints_path)

    def get_representation(self, string):
        """输入token的type，返回所有属于该type token的representation vector"""
        dict_parameter_saved_path = '../js_dataset/split_js_data/parameter.p'
        tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
            pickle.load(open(dict_parameter_saved_path, 'rb'))

        token_index_list = []
        for string_token, index in tt_token_to_int.items():
            node_type = string_token.split('=$$=')[0]
            if node_type == string:
                token_index_list.append(index)

        embedding_vector_list = []
        for index in token_index_list:
            embedding_vector_list.append(self.get_embedding_vector(index))

        return embedding_vector_list


    def get_embedding_vector(self, node_index):
        """给定一个int型的token index，返回训练好的represent vector"""
        return self.model.build_embedding(node_index)



if __name__ == '__main__':
    model = NodeEmbedding()
    represent_list = model.get_representation('LiteralString')
    print(len(represent_list))