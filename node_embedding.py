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
    """加载已经训练好的Node2vec数据"""
    def __init__(self, nt_or_tt):
        self.nt_or_tt = nt_or_tt
        self.session = tf.Session()
        if self.nt_or_tt == 'tt':
            trained_model_path = 'trained_model/node2vec_tt/'
            saver = tf.train.import_meta_graph(trained_model_path + 'EPOCH4.ckpt.meta')
        elif self.nt_or_tt == 'nt':
            trained_model_path = 'trained_model/node2vec_nt/'
            saver = tf.train.import_meta_graph(trained_model_path + 'EPOCH8.ckpt.meta')
        else:
            raise KeyError
        checkpoints_path = tf.train.latest_checkpoint(trained_model_path)
        print('model is loaded from', checkpoints_path)
        saver.restore(self.session, checkpoints_path)
        self.embedding_matrix = self.session.run('embedding_matrix/Variable:0')  # (30001, 300)
        print('shape of embedding matrix:', self.embedding_matrix.shape)

    def save_embedding_matrix(self):
        """将已经训练好的embedding matrix保存到指定路径中"""
        import pickle
        if self.nt_or_tt == 'tt':
            file = open('temp_data/tt_embedding_matrix.pkl', 'wb')
        elif self.nt_or_tt == 'nt':
            file = open('temp_data/nt_embedding_matrix.pkl', 'wb')
        else:
            raise KeyError
        pickle.dump(self.embedding_matrix, file)
        print(self.nt_or_tt, 'embedding matrix has saved...')


    def get_terminal_representation(self, type_string):
        """输入terminal token的type，返回所有属于该type token的representation vector"""
        dict_parameter_saved_path = 'js_dataset/split_js_data/parameter.p'
        tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
            pickle.load(open(dict_parameter_saved_path, 'rb'))

        token_index_list = []
        for string_token, index in tt_token_to_int.items():
            node_type = string_token.split('=$$=')[0]
            if node_type == type_string:
                token_index_list.append(index)

        embedding_vector_list = []
        for index in token_index_list:
            represent_vector = self.embedding_matrix[index]  # 给定一个int型的token index，返回训练好的represent vector
            embedding_vector_list.append(represent_vector)
        embedding_vector_list = np.array(embedding_vector_list)
        return embedding_vector_list

    def get_nonterminal_representation(self, type_string):
        """输入non-terminal token的type，返回所有属于该type token的representation vector"""
        dict_parameter_saved_path = 'js_dataset/split_js_data/parameter.p'
        tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
            pickle.load(open(dict_parameter_saved_path, 'rb'))

        token_index_list = []
        for string_token, index in nt_token_to_int.items():
            node_type = string_token.split('=$$=')[0]
            if node_type == type_string:
                token_index_list.append(index)

        embedding_vector_list = []
        for index in token_index_list:
            represent_vector = self.embedding_matrix[index]  # 给定一个int型的token index，返回训练好的represent vector
            embedding_vector_list.append(represent_vector)
        embedding_vector_list = np.array(embedding_vector_list)
        return embedding_vector_list




def TSNE_fit(data):
    """使用t-SNE对原始的embedding representation vector进行可视化"""
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    return result


def plot_embedding(data, label):
    """对经过降维的数据进行可视化"""
    x_y_min, x_y_max = np.min(data, 0), np.max(data, 0)

    data = (data - x_y_min) / (x_y_max - x_y_min)

    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]) # 设置坐标刻度
    plt.yticks([])
    plt.xlim((-0.05, 1.05)) # 设置坐标轴范围
    plt.ylim((-0.05, 1.05))
    plt.title('t-SNE embedding for Node2vec')
    plt.show(fig)


def normalization(data, bias=5, num=100):
    average_x, average_y = np.average(data, axis=0)
    print(average_x, average_y)
    normal_data = []
    for x, y in data:
        if abs(x - average_x) <= bias and abs(y - average_y) <= bias:
            normal_data.append((x, y))
        if len(normal_data) == num:
            break
    return np.array(normal_data)

def normalization_nt(data):
    average_x, average_y = np.average(data, axis=0)
    print(average_x, average_y)
    return np.array((average_x, average_y)).reshape((1 ,2))


def terminal_embedding_test(model):
    string_represent = model.get_terminal_representation('LiteralString')  # 4662
    property_represent = model.get_terminal_representation('Property')  # 9617
    identifier_represent = model.get_terminal_representation('Identifier')  # 13453
    literal_number_represent = model.get_terminal_representation('LiteralNumber')  # 1369
    this_expression_represent = model.get_terminal_representation('ThisExpression')  # 1
    literal_boolean_represent = model.get_terminal_representation('LiteralBoolean')  # 2

    total_data = np.vstack([string_represent, property_represent,
                            identifier_represent, literal_number_represent])
    total_result = TSNE_fit(total_data)

    string_result = normalization(total_result[:len(string_represent)])
    property_result = normalization(
        total_result[len(string_represent):len(string_represent) + len(property_represent)])
    identifier_result = normalization(
        total_result[len(string_represent) + len(property_represent):
                     len(string_represent) + len(property_represent) + len(identifier_represent)])
    literal_number_result = normalization(
        total_result[len(string_represent) + len(property_represent) + len(identifier_represent):
                     len(string_represent) + len(property_represent) + len(identifier_represent) + len(
                         literal_number_represent)])

    string_label = np.ones([string_result.shape[0]], dtype=np.int16) * 0
    property_label = np.ones([property_result.shape[0]], dtype=np.int16) * 1
    identifier_label = np.ones(identifier_result.shape[0], dtype=np.int16) * 2
    literal_number_label = np.ones(literal_number_result.shape[0], dtype=np.int16) * 3

    plot_data = np.vstack([string_result, property_result, identifier_result,
                           literal_number_result])
    plot_label = np.concatenate([string_label, property_label, identifier_label,
                                 literal_number_label])

    plot_embedding(plot_data, plot_label)


def non_terminal_embedding_test(model):
    member_exp_represent = model.get_nonterminal_representation('MemberExpression')
    call_exp_represent = model.get_nonterminal_representation('CallExpression')
    expression_stat_represent = model.get_nonterminal_representation('ExpressionStatement')
    block_stat_represent = model.get_nonterminal_representation('BlockStatement')
    property_represent = model.get_nonterminal_representation('Property')
    binary_exp_represent = model.get_nonterminal_representation('BinaryExpression')
    assignment_exp_represent = model.get_nonterminal_representation('AssignmentExpression')
    variable_exp_represent = model.get_nonterminal_representation('VariableDeclarator')
    total_data = np.vstack([member_exp_represent, call_exp_represent, expression_stat_represent,
                            block_stat_represent, property_represent, binary_exp_represent,
                            assignment_exp_represent, variable_exp_represent])
    total_result = TSNE_fit(total_data)

    member_len = len(member_exp_represent)
    call_len = len(call_exp_represent)
    express_len = len(expression_stat_represent)
    block_len = len(block_stat_represent)
    property_len = len(property_represent)
    binary_len = len(binary_exp_represent)
    assign_len = len(assignment_exp_represent)
    variable_len = len(variable_exp_represent)

    member_result = normalization_nt(total_result[:member_len])
    call_result = normalization_nt(total_result[member_len: member_len + call_len])
    express_result = normalization_nt(total_result[member_len + call_len: member_len + call_len + express_len])
    block_result = normalization_nt(total_result[member_len + call_len + express_len:
                                                 member_len + call_len + express_len + block_len])
    property_result = normalization_nt(total_result[member_len + call_len + express_len + block_len:
                                                    member_len + call_len + express_len + block_len + property_len])
    binary_result = normalization_nt(total_result[member_len + call_len + express_len + block_len + property_len:
                                                  member_len + call_len + express_len + block_len + property_len + binary_len])
    assign_result = normalization_nt(
        total_result[member_len + call_len + express_len + block_len + property_len + binary_len:
                     member_len + call_len + express_len + block_len + property_len + binary_len + assign_len])
    variable_result = normalization_nt(
        total_result[member_len + call_len + express_len + block_len + property_len + binary_len + assign_len:])

    member_label = np.ones(member_result.shape[0], dtype=np.int16) * 0
    call_label = np.ones(call_result.shape[0], dtype=np.int16) * 1
    express_label = np.ones(express_result.shape[0], dtype=np.int16) * 2
    block_label = np.ones(block_result.shape[0], dtype=np.int16) * 3
    property_label = np.ones(property_result.shape[0], dtype=np.int16) * 4
    binary_label = np.ones(binary_result.shape[0], dtype=np.int16) * 5
    assign_label = np.ones(assign_result.shape[0], dtype=np.int16) * 6
    variable_label = np.ones(variable_result.shape[0], dtype=np.int16) * 7

    plot_data = np.vstack([member_result, call_result, express_result, block_result, property_result,
                           binary_result, assign_result, variable_result])
    plot_label = np.concatenate((member_label, call_label, express_label, block_label, property_label,
                                 binary_label, assign_label, variable_label))
    plot_embedding(plot_data, plot_label)





if __name__ == '__main__':
    model = NodeEmbedding('nt')
    model.save_embedding_matrix()
    #terminal_embedding_test(model)
    #non_terminal_embedding_test(model)
