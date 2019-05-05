import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

import setting

"""对经由Node2Vec训练产生的representation embedding matrix进行提取，并通过各种方法检测re-vector的效果
方法包括对re-vector进行降维并可视化，以及计算给定两个表示向量之间的相似度"""

embed_setting = setting.Setting()
num_ntoken = embed_setting.num_non_terminal
num_ttoken = embed_setting.num_terminal



class NodeEmbedding(object):
    """加载已经训练好的Node2vec数据"""
    def __init__(self):
        if os.path.exists('temp_data/nt_embedding_matrix_300.pkl') and \
                os.path.exists('temp_data/tt_embedding_matrix_300.pkl'):
            self.load_matrix_from_file()
        else:
            self.load_matrix_from_model()

    def load_matrix_from_file(self):
        """从本地文件中加载每个node的表示向量"""
        with open('temp_data/tt_embedding_matrix_300.pkl', 'rb') as file:
            self.tt_embedding_matrix = pickle.load(file)
        with open('temp_data/nt_embedding_matrix_300.pkl', 'rb') as file:
            self.nt_embedding_matrix = pickle.load(file)
        print("load nt/tt embedding matrix from file....")

    def load_matrix_from_model(self):
        """从训练好的模型中加载每个node的表示向量"""
        with tf.Session() as sess:
            trained_model_path = 'trained_model/node2vec_tt/'
            saver = tf.train.import_meta_graph(trained_model_path + 'EPOCH4.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint(trained_model_path))
            self.tt_embedding_matrix = sess.run('embedding_matrix/Variable:0')
            print('loading terminal matrix with shape:',self.tt_embedding_matrix.shape)
        tf.reset_default_graph()
        with tf.Session() as sess:
            trained_model_path = 'trained_model/node2vec_nt/'
            saver = tf.train.import_meta_graph(trained_model_path + 'EPOCH8.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint(trained_model_path))
            self.nt_embedding_matrix = sess.run('embedding_matrix/Variable:0')
            print('loading non-terminal matrix with shape:', self.nt_embedding_matrix.shape)

    def save_embedding_matrix(self):
        """将已经训练好的embedding matrix保存到指定路径中"""
        with open('temp_data/tt_embedding_matrix.pkl', 'wb') as file:
            pickle.dump(self.tt_embedding_matrix, file)
            print('terminal embedding matrix has saved...')
        with open('temp_data/nt_embedding_matrix.pkl', 'wb') as file:
            pickle.dump(self.nt_embedding_matrix, file)
            print('non-terminal embedding matrix has saved...')

    def get_embedding_representation(self, type_string, nt_or_tt):
        """输入terminal token的type，返回所有属于该type token的representation vector"""
        dict_parameter_saved_path = 'js_dataset/split_js_data/parameter.p'
        tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
            pickle.load(open(dict_parameter_saved_path, 'rb'))

        token_index_list = []
        for string_token, index in tt_token_to_int.items():
            node_type = string_token.split('=$$=')[0]
            if node_type == type_string:
                token_index_list.append(index)

        if nt_or_tt == 'nt':
            matrix = self.nt_embedding_matrix
        elif nt_or_tt == 'tt':
            matrix = self.tt_embedding_matrix
        else:
            raise KeyError

        embedding_vector_list = []
        for index in token_index_list:
            represent_vector = matrix[index]  # 给定一个int型的token index，返回训练好的represent vector
            embedding_vector_list.append(represent_vector)
        embedding_vector_list = np.array(embedding_vector_list)
        return embedding_vector_list

    def get_most_similar(self, string_node, nt_or_tt, topk=5):
        """计算与给定node余弦相似度最大的top k个node"""
        dict_parameter_saved_path = 'js_dataset/split_js_data/parameter.p'
        tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
            pickle.load(open(dict_parameter_saved_path, 'rb'))
        if nt_or_tt == 'nt':
            embedding_matrix = self.nt_embedding_matrix
            token_to_int = nt_token_to_int
            int_to_token = nt_int_to_token
        elif nt_or_tt == 'tt':
            embedding_matrix = self.tt_embedding_matrix
            token_to_int = tt_token_to_int
            int_to_token = tt_int_to_token
        else:
            raise KeyError
        index_node = token_to_int[string_node]
        node_vector = embedding_matrix[index_node]
        norm_vector = node_vector / np.sqrt(np.sum(np.square(node_vector)))
        norm = np.sqrt(np.sum(np.square(embedding_matrix), 1, keepdims=True))
        normalizad_embedding = embedding_matrix / norm

        similarity = np.matmul(norm_vector, np.transpose(normalizad_embedding))
        similar_index = np.argsort(-similarity)
        similar_value = -np.sort(-similarity)
        near_list = []
        for i in range(topk):
            near_index = similar_index[i]
            near_value = similar_value[i]
            near_node = int_to_token[near_index]
            near_list.append((near_node, near_value))

        return near_list

    def calculate_distance(self, token1, token2, nt_or_tt):
        """计算给定的两个向量的相L2距离"""
        dict_parameter_saved_path = 'js_dataset/split_js_data/parameter.p'
        tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
            pickle.load(open(dict_parameter_saved_path, 'rb'))
        if nt_or_tt == 'nt':
            embedding_matrix = self.nt_embedding_matrix
            token_to_int = nt_token_to_int
        elif nt_or_tt == 'tt':
            embedding_matrix = self.tt_embedding_matrix
            token_to_int = tt_token_to_int
        else:
            raise KeyError

        repre_vector1 = embedding_matrix[token_to_int[token1]]
        repre_vector2 = embedding_matrix[token_to_int[token2]]
        distance = np.sqrt(np.sum(np.square(repre_vector1 - repre_vector2)))
        return distance

    def calculate_similarity(self, string_node1, string_node2, nt_or_tt):
        """计算两个node的余弦相似度"""
        dict_parameter_saved_path = 'js_dataset/split_js_data/parameter.p'
        tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
            pickle.load(open(dict_parameter_saved_path, 'rb'))
        if nt_or_tt == 'nt':
            embedding_matrix = self.nt_embedding_matrix
            token_to_int = nt_token_to_int
            int_to_token = nt_int_to_token
        elif nt_or_tt == 'tt':
            embedding_matrix = self.tt_embedding_matrix
            token_to_int = tt_token_to_int
            int_to_token = tt_int_to_token
        else:
            raise KeyError

        node_vector1 = embedding_matrix[token_to_int[string_node1]]
        norm_vector1 = node_vector1 / np.sqrt(np.sum(np.square(node_vector1)))

        node_vector2 = embedding_matrix[token_to_int[string_node2]]
        norm_vector2 = node_vector2 / np.sqrt(np.sum(np.square(node_vector2)))

        similarity = np.matmul(norm_vector1, np.transpose(norm_vector2))

        return similarity

    def fix_model(self, string_node1, string_node2):
        """修正给定的两个向量，让其更加接近对方，具体做法是将两个向量相加并除以2，并用这个向量代替其中一个向量"""
        dict_parameter_saved_path = 'js_dataset/split_js_data/parameter.p'
        tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
            pickle.load(open(dict_parameter_saved_path, 'rb'))
        embedding_matrix = self.tt_embedding_matrix
        token_to_int = tt_token_to_int
        int_to_token = tt_int_to_token

        node_vector1 = embedding_matrix[token_to_int[string_node1]]
        node_vector2 = embedding_matrix[token_to_int[string_node2]]
        new_vector = (node_vector1 + node_vector2) / 2
        norm_vector1 = node_vector1 / np.sqrt(np.sum(np.square(node_vector1)))
        norm_vector_new = new_vector / np.sqrt(np.sum(np.square(new_vector)))
        similarity = np.matmul(norm_vector1, np.transpose(norm_vector_new))
        print(similarity)
        return similarity







if __name__ == '__main__':
    model = NodeEmbedding()
    #similarity = model.calculate_similarity('LiteralString=$$=size', 'LiteralString=$$=length', 'tt')
    similarity = model.calculate_similarity('Identifier=$$=size', 'Identifier=$$=length', 'tt')
    print(similarity)
    model.fix_model('LiteralString=$$=size', 'LiteralString=$$=length')
    # similarity = model.calculate_similarity('LiteralString=$$=size', 'LiteralNumber=$$=1', 'tt')
    # print(similarity)
    # distance = model.calculate_distance('LiteralString=$$=size', 'LiteralString=$$=length', 'tt')
    # near_list = model.get_most_similar('LiteralString=$$=size', 'tt')
    # print(near_list)

    #model.save_embedding_matrix()
    #terminal_embedding_test(model)
    #non_terminal_embedding_test(model)
