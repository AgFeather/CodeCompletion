import json
import sys
from collections import Counter
from json.decoder import JSONDecodeError
import pickle

from setting import Setting
import utils

"""rename all variables as arg1, arg2..."""


base_setting = Setting()

js_train_data_dir = base_setting.origin_train_data_dir
js_test_data_dir = base_setting.origin_test_data_dir

data_parameter_dir = 'js_dataset/rename_variable/rename_parameter.pkl'

sub_train_data_dir = 'js_dataset/rename_variable/train_data/'
sub_valid_data_dir = 'js_dataset/rename_variable/valid_data/'
sub_test_data_dir = 'js_dataset/rename_variable/test_data/'

sub_int_train_dir = 'js_dataset/rename_variable/train_data/int_format/'
sub_int_valid_dir = 'js_dataset/rename_variable/valid_data/int_format/'
sub_int_test_dir = 'js_dataset/rename_variable/test_data/int_format/'

num_sub_valid_data = base_setting.num_sub_valid_data
num_sub_train_data = base_setting.num_sub_train_data
num_sub_test_data = base_setting.num_sub_test_data

most_common_termial_num = 20000
unknown_token = base_setting.unknown_token
time_steps = base_setting.time_steps


def dataset_split(is_training=True, subset_size=5000):
    """读取原始AST数据集，并将其分割成多个subset data
    对每个AST，将其转换成二叉树的形式，然后进行中序遍历生成一个nt-sequence"""
    sys.setrecursionlimit(10000)  # 设置递归最大深度
    print('setrecursionlimit == 10000')

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
    nt_seq = []
    for i in range(1, total_size + 1):
        try:
            line = file.readline()  # read a lind from file(one ast)
            ast = json.loads(line)  # transform it to json format
            rename_ast = rename_variable(ast)
            binary_tree = bulid_binary_tree(rename_ast)  # AST to binary tree
            nt_seq = ast_to_seq(binary_tree)  # binary to nt_sequence
        except UnicodeDecodeError as error:  # arise by readline
            print(error)
        except JSONDecodeError as error:  # arise by json_load
            print(error)
        except RecursionError as error:
            print(error)
        except BaseException:
            print('other unknown error, plesae check the code')
        else:
            subset_list.append(nt_seq)  # 将生成的nt sequence加入到list中

        if i % subset_size == 0:  # 当读入的ast已经等于给定的subset的大小时
            sub_path = saved_to_path + \
                'part{}'.format(i // subset_size) + '.json'
            utils.pickle_save(sub_path, subset_list)  # 将subset dataset保存
            subset_list = []

    if is_training:  # 当处理训练数据集时，需要保存映射map，测试数据集则不需要
        save_string_int_dict()
        print('training data seperating finished...')
        print('encoding information has been saved in {}'.format(data_parameter_dir))
    else:
        print('testing data seperating finished...')


def bulid_binary_tree(ast):
    """transform the AST(one node may have several childNodes) to
    Left-Child-Right-Sibling(LCRS) binary tree."""
    brother_map = {0: -1}
    for index, node in enumerate(ast):  # 顺序遍历每个AST中的node

        if not isinstance(node, dict) and node == 0:  # AST中最后添加一个'EOF’标识
            ast[index] = 'EOF'
            break

        node['right'] = brother_map.get(node['id'], -1)

        if 'children' in node.keys():  # 表示该node为non-terminal
            node['isTerminal'] = False  # 存在四种token，有children list但是list的长度为0，暂时将其归为terminal
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
non_terminal_set = set()  # 统计non_termial token 种类


def ast_to_seq(binary_tree, run_or_process='process'):
    # 将一个ast首先转换成二叉树，然后对该二叉树进行中序遍历，得到nt_sequence
    temp_terminal_count = Counter()
    temp_non_terminal_set = set()

    def node_to_string(node):
        # 将node的所有信息进行编码，转换为string
        if node == 'EMPTY':
            string_node = 'EMPTY'
            temp_terminal_count[string_node] += 1
        elif node['isTerminal']:  # 如果node为terminal
            string_node = str(node['type'])
            if 'value' in node.keys():
                # Note:There are some tokens(like:break .etc）do not contains 'value'
                string_node += '=$$=' + str(node['value'])
            temp_terminal_count[string_node] += 1

        else:  # 如果是non-terminal

            string_node = str(node['type']) + '=$$=' + \
                str(node['hasSibling']) + '=$$=' + \
                str(node['hasNonTerminalChild'])  # 有些non-terminal包含value，探索该value的意义？（value种类非常多）
            temp_non_terminal_set.add(string_node)

        return string_node

    def in_order_traversal(bin_tree, index):
        # 对给定的二叉树进行中序遍历，并在中序遍历的时候，生成nt_pair
        node = bin_tree[index]
        if 'left' in node.keys():
            in_order_traversal(bin_tree, node['left'])

        if node == 'EMPTY' or node['isTerminal'] is True:  # 该token是terminal，只将其记录到counter中
            node_to_string(node)
        else:
            assert 'isTerminal' in node.keys() and node['isTerminal'] is False
            # 如果该node是non-terminal，并且包含一个terminal 子节点，则和该子节点组成nt_pair保存在output中
            # 否则将nt_pair的T设为字符串EMPTY
            n_pair = node_to_string(node)
            for child_index in node['children']:  # 遍历该non-terminal的所有child，分别用所有child构建NT-pair
                if bin_tree[child_index]['isTerminal']:
                    t_pair = node_to_string(bin_tree[child_index])
                else:
                    t_pair = node_to_string('EMPTY')
                nt_pair = (n_pair, t_pair)
                output.append(nt_pair)

            # #原处理方式会产生多余的 nt，empty。所以应该改成下面的代码
            # n_pair = node_to_string(node)
            # has_terminal_child = False
            # for child_index in node['children']:  # 遍历该non-terminal的所有child，分别用所有terminal child构建NT-pair
            #     if bin_tree[child_index]['isTerminal']:
            #         t_pair = node_to_string(bin_tree[child_index])
            #         has_terminal_child = True
            #         nt_pair = (n_pair, t_pair)
            #         output.append(nt_pair)
            # if not has_terminal_child: # 该nt node不包含任何terminal child
            #     t_pair = node_to_string('EMPTY')
            #     nt_pair = (n_pair, t_pair)
            #     output.append(nt_pair)

        if node['right'] != -1:  # 遍历right side
            in_order_traversal(bin_tree, node['right'])

    output = []
    in_order_traversal(binary_tree, 0)

    if run_or_process == 'run':
        return output

    if len(output) >= time_steps:  # note: 仅将长度大于阈值的ast产生的node统计到counter中
        terminal_count.update(temp_terminal_count)
        non_terminal_set.update(temp_non_terminal_set)
    else:
        output = []
    return output


def rename_variable(ast):
    """将terminal token中的variable rename成统一的arg1， arg2的形式，大大减少variable的种类"""
    rename_map = {}
    for node in ast:
        if node == 0:
            break
        if ('children' not in node.keys() or len(node['children']) == 0) and node['type'] == 'Identifier':
            terminal_value = node['value']
            if terminal_value in rename_map.keys():
                node['value'] = rename_map[terminal_value]
            else:
                order_value = 'arg' + str(len(rename_map.keys()) + 1)
                node['value'] = order_value
                rename_map[terminal_value] = order_value
    return ast




def save_string_int_dict():
    # 将nonterminal和terminal对应的映射字典保存并返回
    # 其中，对于terminal只选用most frequent的30000个token
    tt_token_to_int = {}
    tt_int_to_token = {}
    nt_token_to_int = {}
    nt_int_to_token = {}

    import pickle
    pickle.dump([terminal_count], open('js_dataset/rename_variable/terminal_counter.pkl', 'wb'))

    most_common_tuple = terminal_count.most_common(most_common_termial_num)
    for index, (token, times) in enumerate(most_common_tuple):
        tt_token_to_int[token] = index
        tt_int_to_token[index] = token
    for index, token in enumerate(list(non_terminal_set)):
        nt_token_to_int[token] = index
        nt_int_to_token[index] = token

    tt_int_to_token[len(tt_int_to_token)] = unknown_token  # terminal中添加UNK
    tt_token_to_int[unknown_token] = len(tt_token_to_int)

    utils.pickle_save(data_parameter_dir,
                [tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token])  # 将映射字典保存到本地
    return tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token



def nt_seq_to_int(time_steps=50, status='TRAIN'):
    # 对NT seq进行进一步的处理，首先将每个token转换为number，
    # 然后对于train data和valid data将所有ast-seq extend成一个list 便于训练时的格式转换
    # 对于test data，将所有ast-seq append，保留各个ast的独立seq
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
        pickle.load(open('js_dataset/rename_variable/rename_parameter.pkl', 'rb'))
    total_num_nt_pair = 0
    if status == 'TRAIN':
        sub_data_dir = sub_train_data_dir
        num_sub_data = num_sub_train_data
        sub_int_data_dir = sub_int_train_dir
    elif status == 'VALID':
        sub_data_dir = sub_valid_data_dir
        num_sub_data = num_sub_valid_data
        sub_int_data_dir = sub_int_valid_dir
    elif status == 'TEST':
        sub_data_dir = sub_test_data_dir
        num_sub_data = num_sub_test_data
        sub_int_data_dir = sub_int_test_dir
    else:
        print('ERROR! Unknown commend!!')
        sys.exit(1)

    def get_subset_data():  # 对每个part的nt_sequence读取并返回，等待进行处理
        for i in range(1, num_sub_data + 1):
            data_path = sub_data_dir + 'part{}.json'.format(i)
            data = utils.pickle_load(data_path)
            yield (i, data)

    subset_generator = get_subset_data()
    for index, data in subset_generator:
        data_seq = []
        for one_ast in data:  # 将每个nt_seq进行截取，并encode成integer，然后保存
            if len(one_ast) < time_steps:  # 该ast大小不足time step 舍去
                continue
            try:
                nt_int_seq = [(nt_token_to_int[n], tt_token_to_int.get(
                    t, tt_token_to_int[unknown_token])) for n, t in one_ast]
            except KeyError:
                print('key error')
                continue
            # 在train和valid中，是直接将所有ast-seq extend到一起，在test中，保留各个ast-seq的独立
            if status == 'TEST':
                data_seq.append(nt_int_seq)
                total_num_nt_pair += len(nt_int_seq)
            else:
                data_seq.extend(nt_int_seq)
                total_num_nt_pair += len(nt_int_seq)

        one_sub_int_data_dir = sub_int_data_dir + 'int_part{}.json'.format(index)
        utils.pickle_save(one_sub_int_data_dir, data_seq)
    # old:14,976,250  new:157,237,460  size of training dataset comparison
    # old: 1,557,285  new: 81,078,099  测试数据集数据量对比
    print('There are {} nt_pair in {} dataset...'.format(total_num_nt_pair, status))




if __name__ == '__main__':

    operation_list = ['TRAIN', 'TEST', 'VALID']
    data_process = operation_list[2]

    if data_process == 'TRAIN':
        dataset_split(is_training=True)
        nt_seq_to_int(status='TRAIN')
    elif data_process == 'TEST':
        #dataset_split(is_training=False)
        nt_seq_to_int(status='TEST')
    elif data_process == 'VALID':
        nt_seq_to_int(status='VALID')