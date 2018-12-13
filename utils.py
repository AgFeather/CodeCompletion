import pickle
import json
import sys
from collections import Counter

from setting import Setting

base_setting = Setting()

js_test_data_dir = base_setting.origin_test_data_dir
js_train_data_dir = base_setting.origin_train_data_dir

data_parameter_dir = base_setting.data_parameter_dir

sub_train_data_dir = base_setting.sub_train_data_dir
sub_valid_data_dir = base_setting.sub_valid_data_dir
sub_test_data_dir = base_setting.sub_test_data_dir

sub_int_train_dir = base_setting.sub_int_train_dir
sub_int_valid_dir = base_setting.sub_int_valid_dir
sub_int_test_dir = base_setting.sub_int_test_dir

num_sub_valid_data = base_setting.num_sub_valid_data
num_sub_train_data = base_setting.num_sub_train_data
num_sub_test_data = base_setting.num_sub_test_data

most_common_termial_num = base_setting.num_terminal
unknown_token = base_setting.unknown_token
time_steps = base_setting.time_steps


def dataset_split(is_training=True, subset_size=5000):
    """读取原始AST数据集，并将其分割成多个subset data
    对每个AST，将其转换成二叉树的形式，然后进行中序遍历生成一个nt-sequence"""
    if is_training:  # 对training数据集进行分割
        data_path = js_train_data_dir
        total_size = 100000
        saved_to_path = sub_train_data_dir
    else:  # 对test数据集进行分割
        data_path = js_test_data_dir
        total_size = 50000
        saved_to_path = sub_test_data_dir

    file = open(data_path, 'r')
    subset_list = []
    nt_seq = None
    error_count = 0
    for i in range(1, total_size + 1):
        try:
            line = file.readline()  # 从文件中读取一个AST
            ast = json.loads(line)  # 将string类型转换成为json的ast
            binary_tree = bulid_binary_tree(ast)  # AST转换为二叉树
            nt_seq = ast_to_seq(binary_tree)  # 将一个AST按照规则转换成nt_sequence
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


def bulid_binary_tree(ast):
    """transform the AST(one node may have several childNodes) to
    Left-Child-Right-Sibling(LCRS) binary tree."""
    brother_map = {0: -1}
    for index, node in enumerate(ast):  # 顺序遍历每个AST中的node

        if not isinstance(node, dict) and node == 0:  # AST中最后添加一个'EOF’标识
            ast[index] = 'EOF'
            break  # return data

        node['right'] = brother_map.get(node['id'], -1)

        if 'children' in node.keys():  # 表示该node为non-terminal
            node['isTerminal'] = False
            add_two_bits_info(ast, node, brother_map)  # 向每个节点添加两bit的额外信息
            child_list = node['children']
            node['left'] = child_list[0]  # 构建该node的left node
            for i, bro in enumerate(child_list):  # 为该node的所有children构建right sibling
                if i == len(child_list) - 1:
                    break
                brother_map[bro] = child_list[i + 1]
        else:
            node['isTerminal'] = True
    return ast


def add_two_bits_info(ast, node, brother_map):
    # 向每个节点添加两bit的额外信息：hasNonTerminalChild和hasSibling
    node['hasNonTerminalChild'] = False
    for child_index in node['children']:
        if 'children' in ast[child_index]:
            node['hasNonTerminalChild'] = True
            break
    if brother_map.get(node['id'], -1) == -1:
        node['hasSibling'] = False
    else:
        node['hasSibling'] = True


terminal_count = Counter()  # 统计每个terminal token的出现次数
non_termial_set = set()  # 统计non_termial token 种类


def ast_to_seq(binary_tree):
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
                str(node['hasNonTerminalChild'])  # 有些non-terminal包含value，探索该value的意义？（value种类非常多）
            non_termial_set.add(string_node)
        return string_node

    def in_order_traversal(bin_tree, index):
        # 对给定的二叉树进行中序遍历，并在中序遍历的时候，生成nt_pair
        node = bin_tree[index]
        if 'left' in node.keys():
            in_order_traversal(bin_tree, node['left'])

        if 'isTerminal' in node.keys() and node['isTerminal'] is False:
            # 如果该node是non-terminal，并且包含一个terminal 子节点，则和该子节点组成nt_pair保存在output中
            # 否则将nt_pair的T设为字符串EMPTY
            n_pair = node_to_string(node)
            for child_index in node['children']:  # 遍历该Nterminal的所有child，分别用所有child构建NT-pair
                if bin_tree[child_index]['isTerminal']:
                    t_pair = node_to_string(bin_tree[child_index])
                else:
                    t_pair = node_to_string('EMPTY')
                nt_pair = (n_pair, t_pair)
                output.append(nt_pair)

        else:  # 该token是terminal，只将其记录到counter中
            node_to_string(node)

        if node['right'] != -1:  # 遍历right side
            in_order_traversal(bin_tree, node['right'])

    output = []
    in_order_traversal(binary_tree, 0)
    return output


def pickle_save(path, data):
    """使用pickle将给定数据保存到给定路径中"""
    file = open(path, 'wb')
    pickle.dump(data, file)
    print(path + ' has been saved...')


def pickle_load(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    return data


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


def train_nt_seq_to_int(time_steps=50, train_or_valid='TRAIN'):
    # 对NT seq进行进一步的处理，首先将每个token转换为number，
    # 然后截取各个seq成time step的倍数，然后将所有AST都拼接到一起.
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = load_dict_parameter()
    total_num_nt_pair = 0
    if train_or_valid == 'TRAIN':
        sub_data_dir = sub_train_data_dir
        num_sub_data = num_sub_train_data
        sub_int_data_dir = sub_int_train_dir
    elif train_or_valid == 'VALID':
        sub_data_dir = sub_valid_data_dir
        num_sub_data = num_sub_valid_data
        sub_int_data_dir = sub_int_valid_dir
    else:
        print('ERROR! Unknown commend!!')
        sys.exit(1)

    def get_subset_data():  # 对每个part的nt_sequence读取并返回，等待进行处理
        for i in range(1, 20 + 1):
            data_path = sub_data_dir + 'part{}.json'.format(i)
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
            data_seq.extend(nt_int_seq)  # todo：修改为append，让其分句

        total_num_nt_pair += len(data_seq)
        one_sub_int_data_dir = sub_int_data_dir + 'int_part{}.json'.format(index)
        pickle_save(one_sub_int_data_dir, data_seq)

    print('There are {} nt_pair in {} dataset...'.format(total_num_nt_pair, train_or_valid))  # old: 6,970,900  new: 14,976,250


def test_nt_seq_to_int():
    # 读入已经被分割并处理成nt sequence的test data，
    # 然后根据处理training数据时生成的token2int映射字典将其转换成对应的int sequence
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = load_dict_parameter()
    total_num_nt_pair = 0

    def get_subset_data():  # 对每个part的nt_sequence读取并返回，等待进行处理
        for i in range(1, num_sub_test_data + 1):
            data_path = sub_test_data_dir + 'part{}.json'.format(i)
            file = open(data_path, 'rb')
            data = pickle.load(file)
            yield (i, data)

    subset_generator = get_subset_data()
    short_count = 0
    for index, nt_data in subset_generator:
        data_seq = []
        num_nt_pair = 0
        for one_ast in nt_data:  # 将每个由token组成的nt_seq，并encode成integer，然后保存
            if len(one_ast) < time_steps:  # 对于长度小于50的ast，直接舍去
                short_count += 1
                continue
            nt_int_seq = [(nt_token_to_int[n],
                           tt_token_to_int.get(t, tt_token_to_int[unknown_token])) for n, t in one_ast]
            data_seq.append(nt_int_seq)
            num_nt_pair += len(nt_int_seq)
        total_num_nt_pair += num_nt_pair

        one_sub_train_int_data_dir = sub_int_test_dir + 'int_part{}.json'.format(index)
        pickle_save(one_sub_train_int_data_dir, data_seq)

    print('There are {} nt_sequence which length is shorter than {}'.format(short_count, time_steps))
    print('There are {} nt_pair in test data set...'.format(total_num_nt_pair))  # new: 1,557,285


if __name__ == '__main__':

    operation_list = ['TRAIN', 'TEST', 'VALID']
    data_process = operation_list[0]

    if data_process == 'TRAIN':
        # dataset_split(is_training=True)
        train_nt_seq_to_int(train_or_valid='TRAIN')
    elif data_process == 'TEST':
        # dataset_split(is_training=False)
        test_nt_seq_to_int()
    elif data_process == 'VALID':
        train_nt_seq_to_int(train_or_valid='VALID')