import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import setting

embed_setting = setting.Setting()
num_ntoken = embed_setting.num_non_terminal
num_ttoken = embed_setting.num_terminal



class NodeEmbedding(object):
    def __init__(self):
        self.session = tf.Session()
        trained_model_path = 'trained_model/node2vec_tt/'
        checkpoints_path = tf.train.latest_checkpoint(trained_model_path)
        print('model is loaded from', checkpoints_path)
        saver = tf.train.import_meta_graph(trained_model_path + 'EPOCH4.ckpt.meta')
        saver.restore(self.session, checkpoints_path)
        self.embedding_matrix = self.session.run('embedding_matrix/Variable:0')
        print(self.embedding_matrix.shape)

    def get_representation(self, string):
        """输入token的type，返回所有属于该type token的representation vector"""
        dict_parameter_saved_path = 'js_dataset/split_js_data/parameter.p'
        tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
            pickle.load(open(dict_parameter_saved_path, 'rb'))

        token_index_list = []
        for string_token, index in tt_token_to_int.items():
            node_type = string_token.split('=$$=')[0]
            if node_type == string:
                token_index_list.append(index)

        embedding_vector_list = []
        for index in token_index_list:
            represent_vector = self.embedding_matrix[index]  # 给定一个int型的token index，返回训练好的represent vector
            embedding_vector_list.append(represent_vector)
        embedding_vector_list = np.array(embedding_vector_list)
        return embedding_vector_list



def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 #color=plt.cm.Set1(label[i] / 10.),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def fit_main(data):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    return result

def plot_main(result, label):
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits')
    plt.show(fig)


if __name__ == '__main__':
    model = NodeEmbedding()
    string_represent = model.get_representation('LiteralString')
    property_represent = model.get_representation('Property')
    print(string_represent.shape)
    print(property_represent.shape)

    string_label = np.zeros([string_represent.shape[0]], dtype=np.int16)
    property_label = np.ones([property_represent.shape[0]], dtype=np.int16)
    print(string_label.shape)
    print(property_label.shape)

    data = np.vstack([string_represent, property_represent])
    label = np.concatenate([string_label, property_label])
    print(data.shape)
    print(label.shape)

    result = fit_main(data)
    plot_main(result, label)