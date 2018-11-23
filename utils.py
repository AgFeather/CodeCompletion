import pickle
import json
from collections import Counter

from setting import Setting

base_setting = Setting()

js_test_data_dir = base_setting.origin_test_data_dir
js_train_data_dir = base_setting.origin_train_data_dir

data_parameter_dir = base_setting.data_parameter_dir
train_subset_dir = base_setting.sub_train_data_dir
sub_test_data_dir = base_setting.sub_test_data_dir
sub_train_data_dir = base_setting.sub_train_data_dir
sub_int_train_dir = base_setting.sub_int_train_dir
sub_int_test_dir = base_setting.sub_test_data_dir

most_common_termial_num = base_setting.num_terminal
unknown_token = base_setting.unknown_token
num_sub_train_data = base_setting.num_sub_train_data
num_subset_train_data = base_setting.num_sub_test_data


def dataset_split(is_training=True, subset_size=5000):
    """读取原始AST数据集，并将其分割成多个subset data
    对每个AST，将其转换成二叉树的形式，然后进行中序遍历生成一个nt-sequence"""
    if is_training:  # 对training数据集进行分割
        data_path = js_train_data_dir
        total_size = 100000
        saved_to_path = train_subset_dir
    else:  # 对test数据集进行分割
        data_path = js_test_data_dir
        total_size = 50000
        saved_to_path = sub_test_data_dir

    file = open(data_path, 'r')
    subset_list = []
    error_count = 0
    for i in range(1, total_size + 1):
        try:
            line = file.readline()  # 从文件中读取一个AST
            line = json.loads(line)  # 将string类型转换成为json
            nt_seq = ast_to_seq(line)  # 将一个AST按照规则转换成nt_sequence
        except BaseException:
            error_count += 1
        subset_list.append(nt_seq)  # 将生成的nt sequence加入到list中

        if i % subset_size == 0:  # 当读入的ast已经等于给定的subset的大小时
            sub_path = saved_to_path + \
                'part{}'.format(i // subset_size) + '.json'
            pickle_save(sub_path, subset_list)  # 将subset dataset保存
            subset_list = []

    if is_training:  # 当处理训练数据集时，需要保存映射map，测试数据集则不需要
        save_string_int_dict()
        print('training data seperating finished...')
        print('encoding information has been save in {}'.format(data_parameter_dir))
    else:
        print('testing data seperating finished...')


terminal_count = Counter()  # 统计每个terminal token的出现次数
non_termial_set = set()  # 统计non_termial token 种类


def ast_to_seq(ast):
    # 将一个ast首先转换成二叉树，然后对该二叉树进行中序遍历，得到nt_sequence
    def node_to_string(node):
        # 将一个node转换为string
        if node == 'EMPTY':
            string_node = 'EMPTY'
        elif node['isTerminal']:  # 如果node为terminal
            string_node = str(node['type']) + '=$$=' + \
                str(node['value'])  # + '==' + str(node['id'])
            terminal_count[string_node] += 1
        else:  # 如果是non-terminal
            string_node = str(node['type']) + '=$$=' + \
                str(node['hasSibling']) + '=$$=' + \
                str(node['isTerminal'])  # 有些non-terminal包含value，探索该value的意义？（value种类非常多）
            non_termial_set.add(string_node)
        return string_node

    def in_order_traversal(data, index):
        # 对给定的二叉树进行中序遍历，并在中序遍历的时候，生成nt_pair
        node = data[index]
        if 'left' in node.keys():
            in_order_traversal(data, node['left'])

        if 'isTerminal' in node.keys() and node['isTerminal'] is False:
            # 如果该node是non-terminal，并且包含一个terminal 子节点，则和该子节点组成nt_pair保存在output中
            # 否则将nt_pair的T设为字符串EMPTY
            n_pair = node_to_string(node)
            if data[node['left']]['isTerminal']:
                assert data[node['left']]['id'] == node['left']
                t_pair = node_to_string(data[node['left']])
            else:
                t_pair = node_to_string('EMPTY')
            nt_pair = (n_pair, t_pair)
            output.append(nt_pair)
        else:  # 该token是terminal，只将其记录到counter中
            node_to_string(node)

        if node['right'] != -1:  # 遍历right side
            in_order_traversal(data, node['right'])

    output = []
    binary_tree = bulid_binary_tree(ast)  # AST转换为二叉树
    in_order_traversal(binary_tree, 0)
    return output


def pickle_save(path, data):
    """使用pickle将给定数据保存到给定路径中"""
    file = open(path, 'wb')
    pickle.dump(data, file)
    print(path + 'has been saved...')


def pickle_load(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    return data


def bulid_binary_tree(data):
    """transform the AST(one node may have several childNodes) to
    Left-Child-Right-Sibling(LCRS) binary tree."""
    brother_map = {0: -1}
    for index, node in enumerate(data):  # 顺序遍历每个AST中的node

        if not isinstance(node, dict) and node == 0:  # AST中最后添加一个'EOF’标识
            data[index] = 'EOF'
            break  # return data
        add_two_bits_info(node, brother_map)  # 向每个节点添加两bit的额外信息
        node['right'] = brother_map.get(node['id'], -1)

        if 'children' in node.keys():  # 表示该node为non-terminal
            child_list = node['children']
            node['left'] = child_list[0]  # 构建该node的left node
            for i, bro in enumerate(child_list):  # 为该node的所有children构建right sibling
                if i == len(child_list) - 1:
                    break
                brother_map[bro] = child_list[i + 1]
            node.pop('children')

    return data


def add_two_bits_info(node, brother_map):
    # 向每个节点添加两bit的额外信息：isTerminal和hasSibling
    if 'children' in node.keys():
        node['isTerminal'] = False
    else:
        node['isTerminal'] = True
    if brother_map.get(node['id'], -1) == -1:
        node['hasSibling'] = False
    else:
        node['hasSibling'] = True


def save_string_int_dict():
    # 将nonterminal和terminal对应的映射字典保存并返回
    # 其中，对于terminal只选用most frequent的30000个token
    tt_token_to_int = {}
    tt_int_to_token = {}
    nt_token_to_int = {}
    nt_int_to_token = {}

    most_common_tuple = terminal_count.most_common(most_common_termial_num)
    for index, (token, times) in enumerate(most_common_tuple):
        tt_token_to_int[token] = index
        tt_int_to_token[index] = token
    for index, token in enumerate(list(non_termial_set)):
        nt_token_to_int[token] = index
        nt_int_to_token[index] = token

    tt_int_to_token[len(tt_int_to_token)] = unknown_token  # terminal中添加UNK
    tt_token_to_int[unknown_token] = len(tt_token_to_int)

    pickle_save(data_parameter_dir,
                [tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token])  # 将映射字典保存到本地

    return tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token


def load_dict_parameter():
    # 加载terminal和nonterminal对应的映射字典
    file = open(data_parameter_dir, 'rb')
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = pickle.load(file)
    return tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token


def train_nt_seq_to_int(time_steps=50):
    # 对NT seq进行进一步的处理，首先将每个token转换为number，
    # 然后截取各个seq成time step的倍数，然后将所有AST都拼接到一起.
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = load_dict_parameter()
    total_num_nt_pair = 0

    def get_subset_data():  # 对每个part的nt_sequence读取并返回，等待进行处理
        for i in range(1, num_sub_train_data + 1):
            data_path = sub_train_data_dir + 'part{}.json'.format(i)
            data = pickle_load(data_path)
            yield (i, data)

    subset_generator = get_subset_data()
    for index, data in subset_generator:
        data_seq = []
        for one_ast in data:  # 将每个nt_seq进行截取，并encode成integer，然后保存
            num_steps = len(one_ast) // time_steps  # 将每个nt seq都切割成time steps的整数倍
            if num_steps == 0:  # 该ast大小不足time step 舍去
                continue
            one_ast = one_ast[:num_steps * time_steps]
            nt_int_seq = [(nt_token_to_int[n], tt_token_to_int.get(
                    t, tt_token_to_int[unknown_token])) for n, t in one_ast]
            data_seq.extend(nt_int_seq)

        total_num_nt_pair += len(data_seq)
        one_sub_train_int_data_dir = sub_int_train_dir + 'int_part{}.json'.format(index)
        pickle_save(one_sub_train_int_data_dir, data_seq)

    print('There are {} nt_pair in train dataset...'.format(total_num_nt_pair))  # total == 6970900


def test_nt_seq_to_int():
    # 读入已经被分割并处理成nt sequence的test data，
    # 然后根据处理training数据时生成的token2int映射字典将其转换成对应的int sequence
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = load_dict_parameter()
    total_num_nt_pair = 0

    def get_subset_data():  # 对每个part的nt_sequence读取并返回，等待进行处理
        for i in range(1, num_subset_train_data + 1):
            data_path = sub_test_data_dir + 'part{}.json'.format(i)
            file = open(data_path, 'rb')
            data = pickle.load(file)
            yield (i, data)

    subset_generator = get_subset_data()
    for index, data in subset_generator:
        data_seq = []
        num_nt_pair = 0
        for one_ast in data:  # 将每个由token组成的nt_seq，并encode成integer，然后保存
            nt_int_seq = [(nt_token_to_int[n],
                           tt_token_to_int.get(t, tt_token_to_int[unknown_token])) for n, t in one_ast]
            data_seq.append(nt_int_seq)
            num_nt_pair += len(nt_int_seq)
        total_num_nt_pair += num_nt_pair

        one_sub_train_int_data_dir = sub_int_test_dir + 'int_part{}.json'.format(index)
        pickle_save(one_sub_train_int_data_dir, data_seq)

    print('There are {} nt_pair in test data set...'.format(total_num_nt_pair))  # 4706813


if __name__ == '__main__':
    training_data_process = False
    if training_data_process:
        dataset_split(is_training=True)
        train_nt_seq_to_int()
    else:
        dataset_split(is_training=False)
        test_nt_seq_to_int()
        
    ast_example = [{'id': 0, 'type': 'Program', 'children': [1]},
                   {'id': 1, 'type': 'ExpressionStatement', 'children': [2]},
                   {'id': 2, 'type': 'CallExpression', 'children': [3, 8, 9, 10]},
                   {'id': 3, 'type': 'MemberExpression', 'children': [4, 7]},
                   {'id': 4, 'type': 'MemberExpression', 'children': [5, 6]},
                   {'id': 5, 'type': 'Identifier', 'value': 'CKEDITOR'},
                   {'id': 6, 'type': 'Property', 'value': 'plugins'},
                   {'id': 7, 'type': 'Property', 'value': 'setLang'},
                   {'id': 8, 'type': 'LiteralString', 'value': 'iframe'},
                   {'id': 9, 'type': 'LiteralString', 'value': 'ka'},
                   {'id': 10, 'type': 'ObjectExpression', 'children': [11, 13, 15, 17, 19]},
                   {'id': 11, 'type': 'Property', 'value': 'border', 'children': [12]},
                   {'id': 12, 'type': 'LiteralString', 'value': 'ჩარჩოს გამოჩენა'},
                   {'id': 13, 'type': 'Property', 'value': 'noUrl', 'children': [14]},
                   {'id': 14, 'type': 'LiteralString', 'value': 'აკრიფეთ iframe-ის URL'},
                   {'id': 15, 'type': 'Property', 'value': 'scrolling', 'children': [16]},
                   {'id': 16, 'type': 'LiteralString', 'value': 'გადახვევის ზოლების დაშვება'},
                   {'id': 17, 'type': 'Property', 'value': 'title', 'children': [18]},
                   {'id': 18, 'type': 'LiteralString', 'value': 'IFrame-ის პარამეტრები'},
                   {'id': 19, 'type': 'Property', 'value': 'toolbar', 'children': [20]},
                   {'id': 20, 'type': 'LiteralString', 'value': 'IFrame'}, 0]
