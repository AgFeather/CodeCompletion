import matplotlib.pyplot as plt
import numpy as np
import pickle


def TSNE_fit(data):
    """使用t-SNE对原始的embedding representation vector进行降维到2-D"""
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, init='pca', perplexity=4, early_exaggeration=20, random_state=0, method='exact')
    # n_components表示嵌入维度，perplexity表示考虑临近点多少，early_exaggeration表示簇间距大小，越大可视化后的簇间距越大。
    result = tsne.fit_transform(data)
    return result


def plot_embedding(data, label):
    """对经过降维到2-D的数据进行可视化"""
    x_y_min, x_y_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_y_min) / (x_y_max - x_y_min)
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 12})  # 对于non-terminal，调大字体大小
    plt.xticks([]) # 设置坐标刻度
    plt.yticks([])
    plt.xlim((-0.05, 1.05)) # 设置坐标轴范围
    plt.ylim((-0.05, 1.05))
    plt.title('t-SNE embedding for Node2vec')
    plt.show(fig)


def normalization_tt(data, bias=5, num=100):
    """对被降维到2-D的terminal represent vector进行归一化"""
    average_x, average_y = np.average(data, axis=0)
    # print(average_x, average_y)
    normal_data = []
    for x, y in data:
        if abs(x - average_x) <= bias and abs(y - average_y) <= bias:
            normal_data.append((x, y))
        if len(normal_data) == num:
            break
    return np.array(normal_data)


def normalization_nt(data):
    """对被降维到2-D的non-terminal represent vector进行归一化"""
    average_x, average_y = np.average(data, axis=0)
    print(average_x, average_y)
    return np.array((average_x, average_y)).reshape((1 ,2))


def terminal_embedding_test(model):
    """对terminal embedding representation vector进行可视化的整个流程
    首先获取多个type的terminal embedding vector，
    然后使用TSNE算法对所有vector进行降维，
    然后进行可视化"""
    string_represent = model.get_terminal_representation('LiteralString')  # 4662
    property_represent = model.get_terminal_representation('Property')  # 9617
    identifier_represent = model.get_terminal_representation('Identifier')  # 13453
    literal_number_represent = model.get_terminal_representation('LiteralNumber')  # 1369
    this_expression_represent = model.get_terminal_representation('ThisExpression')  # 1
    literal_boolean_represent = model.get_terminal_representation('LiteralBoolean')  # 2

    total_data = np.vstack([string_represent, property_represent,
                            identifier_represent, literal_number_represent])
    total_result = TSNE_fit(total_data)

    string_result = normalization_tt(total_result[:len(string_represent)])
    property_result = normalization_tt(
        total_result[len(string_represent):len(string_represent) + len(property_represent)])
    identifier_result = normalization_tt(
        total_result[len(string_represent) + len(property_represent):
                     len(string_represent) + len(property_represent) + len(identifier_represent)])
    literal_number_result = normalization_tt(
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
    """对non-terminal embedding representation vector进行可视化的整个流程
    首先获取多个type的terminal embedding vector，
    然后使用TSNE算法对所有vector进行降维，
    然后进行可视化"""
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


with open('temp_data/tt_embedding_matrix_temp.pkl', 'rb') as file:
    tt_embedding_matrix = pickle.load(file)
dict_parameter_saved_path = 'js_dataset/split_js_data/parameter.p'
tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
    pickle.load(open(dict_parameter_saved_path, 'rb'))
def node_string_to_vector(node_string_list):
    """给定一个string node list，将该list中的所有node都转换成对应的表示向量"""
    vector_list = []
    string_list = []
    for str_node in node_string_list:
        temp_vec = tt_embedding_matrix[tt_token_to_int[str_node]]
        vector_list.append(temp_vec)
        temp_str = str_node #.split("=$$=")[-1]
        string_list.append(temp_str)

    return vector_list, string_list


def save_matrix():
    pickle.dump(tt_embedding_matrix, open('temp_data/tt_embedding_matrix_temp.pkl', 'wb'))
    print("embeddig matrix has been saved...")


def calculate_similarity(string_node1, string_node2):
    """计算两个node的余弦相似度"""
    node_vector1 = tt_embedding_matrix[tt_token_to_int[string_node1]]
    node_vector2 = tt_embedding_matrix[tt_token_to_int[string_node2]]
    sim = similarity(node_vector1, node_vector2)
    print(sim)
    return sim


def fix_model(string_node1, string_node2):
    """修正给定的两个向量，让其更加接近对方，具体做法是将两个向量相加并除以2，并用这个向量代替其中一个向量"""
    node_vector1 = tt_embedding_matrix[tt_token_to_int[string_node1]]
    node_vector2 = tt_embedding_matrix[tt_token_to_int[string_node2]]
    center_vector = (node_vector1 + node_vector2) / 2
    new_vector1 = (node_vector1 + center_vector) / 2
    new_vector2 = (node_vector2 + center_vector) / 2

    tt_embedding_matrix[tt_token_to_int[string_node1]] = new_vector1
    tt_embedding_matrix[tt_token_to_int[string_node2]] = new_vector2
    sim = similarity(new_vector1, new_vector1)
    print(sim)
    return sim


def similarity(new_vector1, new_vector2):
    norm_vector1 = new_vector1 / np.sqrt(np.sum(np.square(new_vector1)))
    norm_vector2 = new_vector2 / np.sqrt(np.sum(np.square(new_vector2)))
    similarity = np.matmul(norm_vector1, np.transpose(norm_vector2))
    return similarity


# def visual_model_all(string_list):
#     # 分别在fix前后计算所有的相似度并打印，用以验证fix是否成功
#     vector_list = []
#     for string_node in string_list:
#         vector_list.append(tt_embedding_matrix[tt_token_to_int[string_node]])
#
#     simi_list = []
#     for i in range(len(vector_list)-1):
#         simi_list.append(similarity(vector_list[i], vector_list[i+1]))
#     print(simi_list)
#
#     vector_list = np.array(vector_list)
#     center_vector = np.mean(vector_list, axis=0)
#     vector_list = (vector_list + center_vector) / 2
#     for i, string_node in enumerate(string_list):
#         tt_embedding_matrix[tt_token_to_int[string_node]] = vector_list[i]
#
#     simi_list = []
#     for i in range(len(vector_list)-1):
#         simi_list.append(similarity(vector_list[i], vector_list[i+1]))
#     print(simi_list)
#
#     return vector_list


def tt_single_plot(string_node_list):
    def normalize(data):
        max_value = np.max(data, axis=0)
        min_value = np.min(data, axis=0)
        normal_data = 4 * (data - min_value) / (max_value - min_value + 0.00001) - 2
        return normal_data
    def plot_embedding(data, label):
        """对经过降维到2-D的数据进行可视化"""
        def get_color(label):
            if label.startswith('Property'):
                return 'green'
            elif label.startswith('Identifier'):
                return 'blue'
            else:
                return 'red'
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot((111))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))
        for i in range(data.shape[0]):
            # color = plt.cm.Set1(label[i]),
            color = get_color(label[i])
            plt.text(data[i, 0], data[i, 1], str(label[i].split('=$$=')[-1]),
                     color = color,
                     fontdict={'weight': 'bold', 'size': 10})  # 调大字体大小
        # plt.xticks([])  # 设置坐标刻度
        # plt.yticks([])
        plt.xlim((-2.05, 2.05))  # 设置坐标轴范围
        plt.ylim((-2.05, 2.05))
        plt.show(fig)

    vector_list, label_list = node_string_to_vector(string_node_list)
    assert len(vector_list) == len(label_list) == len(string_node_list)

    d2_data = TSNE_fit(vector_list)
    d2_data = normalize(d2_data)
    plot_embedding(d2_data, label_list)


def visualize_accuracy():
    """将准确率曲线进行可视化"""
    index = []
    n_accu = []
    t_accu = []
    file = open('log_info/accu_log/origin_lstm_2019_05_04_13_06.txt', 'r')
    one_line = file.readline()
    fig = plt.figure(figsize=(8, 8))
    while one_line:
        split_data = one_line.split(';')
        index.append(float(split_data[0]))
        n_accu.append(float(split_data[1]))
        t_accu.append(float(split_data[2]))
        if len(index) >= 10000:
            break
        one_line = file.readline()
    assert len(n_accu) == len(t_accu) == len(index)
    plt.plot(index, n_accu)
    plt.show()



if __name__ == '__main__':
    node_list1 = ['Identifier=$$=size', 'Identifier=$$=length', 'Identifier=$$=len', 'Identifier=$$=_len']
    node_list2 = ['Identifier=$$=userName', 'Identifier=$$=name', 'Identifier=$$=id', 'Identifier=$$=user_id', 'Identifier=$$=player']
    node_list3 = ['Property=$$=push', 'Property=$$=get', 'Property=$$=set']
    node_list4 = ['Property=$$=extend', 'Property=$$=append', 'Property=$$=add']
    #node_list5 = ['Property=$$=type', 'Property=$$=value', 'Property=$$=']
    data_list = node_list1 + node_list2  + node_list3
    sim = calculate_similarity(node_list1[0], node_list1[1])
    print('similarity between {} and {} is {}'.format(node_list1[0], node_list1[1], sim))

    sim = calculate_similarity(node_list1[0], node_list2[0])
    print('similarity between {} and {} is {}'.format(node_list1[0], node_list2[0], sim))

    sim = calculate_similarity(node_list2[0], node_list2[1])
    print('similarity between {} and {} is {}'.format(node_list2[0], node_list2[1], sim))



    #tt_single_plot(data_list)
    #visualize_accuracy()

#    calculate_similarity(node_list1[0], node_list1[1])

    # 对应的node的表示向量
    # new_data1 = visual_model_all(node_list1)
    # new_data2 = visual_model_all(node_list2)
    # new_data3 = visual_model_all(node_list3)
    # save_matrix()

