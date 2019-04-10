"""提出的一个新的数据处理方法，该处理方法的优点是可以将生成的token sequence 转换回AST，
这样就实现了真正的code completion"""

import pickle
import json
import sys
from collections import Counter
from json.decoder import JSONDecodeError

from setting import Setting
import utils



base_setting = Setting()

js_test_data_dir = base_setting.origin_test_data_dir
js_train_data_dir = base_setting.origin_train_data_dir

data_parameter_dir = 'js_dataset/new_data_process/data_parameter.pkl'

sub_data_dir = 'js_dataset/new_data_process/'
sub_int_train_data_dir = 'js_dataset/new_data_process/int_format/train_data/'
sub_int_valid_data_dir = 'js_dataset/new_data_process/int_format/valid_data/'
sub_int_test_data_dir = 'js_dataset/new_data_process/int_format/test_data/'

num_sub_train_data = base_setting.num_sub_train_data
num_sub_valid_data = base_setting.num_sub_valid_data
num_sub_test_data = base_setting.num_sub_test_data

most_common_termial_num = 30000
unknown_token = base_setting.unknown_token
time_steps = base_setting.time_steps



def data_process(train_or_test, subset_size=5000):
    """读取原始AST数据集，并将其分割成多个subset data
    对每个AST，将其转换成二叉树的形式，然后进行中序遍历生成一个nt-sequence"""
    sys.setrecursionlimit(10000)  # 设置递归最大深度
    print('setrecursionlimit == 10000')
    saved_to_path = sub_data_dir

    if train_or_test == 'train':  # 对training数据集进行分割
        data_path = js_train_data_dir
        total_size = 100000
        base_num = 0
    elif train_or_test == 'test':  # 对test数据集进行分割
        data_path = js_test_data_dir
        total_size = 50000
        base_num = num_sub_train_data
    else:
        raise KeyError

    file = open(data_path, 'r')
    subset_list = []
    nt_seq = []
    for i in range(1, total_size + 1):
        try:
            line = file.readline()  # read a lind from file(one ast)
            ast = json.loads(line)  # transform it to json format
            binary_tree = bulid_binary_tree(ast)  # AST to binary tree
            nt_seq = ast_to_seq(binary_tree, 'process')  # binary to nt_sequence
        except UnicodeDecodeError as error:  # arise by readline
            print(error)
        except JSONDecodeError as error:  # arise by json_load
            print(error)
        except RecursionError as error:
            print(error)
        except BaseException as error:
            print('UNKNOWN ERROR', error)
        else:
            subset_list.append(nt_seq)  # 将生成的nt sequence加入到list中

        if i % subset_size == 0:  # 当读入的ast已经等于给定的subset的大小时
            sub_path = saved_to_path + \
                'sub_part{}'.format(base_num + (i // subset_size)) + '.json'
            utils.pickle_save(sub_path, subset_list)  # 将subset dataset保存
            subset_list = []

    if train_or_test == 'train':  # 当处理训练数据集时，需要保存映射map，测试数据集则不需要
        save_string_int_dict()
        print('training data seperating finished...')
        print('encoding information has been saved in {}'.format(data_parameter_dir))
    else:
        print('testing data seperating finished...')



def bulid_binary_tree(ast):
    """transform the AST(one node may have several childNodes) to
    Left-Child-Right-Sibling(LCRS) binary tree."""
    def is_terminal(node):
        if 'children' in node.keys() and len(node['children']) > 0:
            return False
        return True

    brother_map = {0: -1}
    for index, node in enumerate(ast):  # 顺序遍历每个AST中的node
        if not isinstance(node, dict) and node == 0:  # 删除AST中最后的0标识
            del ast[index]
            break

        if is_terminal(node):
            # 表示该node为terminal
            node['isTerminal'] = True
            continue
        else:
            # 注： 存在四种token，有children list但是list的长度为0，暂时将其归为terminal
            node['left'] = -1
            node['right'] = brother_map.get(node['id'], -1) # 只为non-terminal node构建左右child
            node['isTerminal'] = False
            add_two_bits_info(ast, node, brother_map)  # 向每个节点添加两bit的额外信息
            child_list = node['children']

            first_nt_i = None
            temp_nt_node_id = None
            for i, child_index in enumerate(child_list):
                # 找到该节点第一个non-terminal child node作为该node的left node
                if not is_terminal(ast[child_index]):
                    node['left'] = child_index
                    first_nt_i = i
                    temp_nt_node_id = child_list[first_nt_i]
                    break

            if first_nt_i != None:
                # 说明该node有non-terminal left child，所以为这个nt left child构建brother map
                assert isinstance(first_nt_i, int) \
                       and first_nt_i < len(child_list) \
                       and isinstance(temp_nt_node_id, int)

                for index in range(first_nt_i+1, len(child_list)):
                    next_node_id = child_list[index]
                    # 为该node的所有non-terminal children构建non-terminal right sibling
                    if not is_terminal(ast[next_node_id]):
                        #print('nt',next_node_id)
                        brother_map[temp_nt_node_id] = next_node_id
                        temp_nt_node_id = next_node_id
                        #print(brother_map)

            # 将转化生成的binary tree添加节点，组成完全二叉树
            if (node['left'] == -1) and (node['right'] != -1):
                node['left'] = 'PAD_EMPTY'
            if (node['left'] != -1) and(node['right'] == -1):
                node['right'] = 'PAD_EMPTY'

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
terminal_count[unknown_token] += 1
non_terminal_set = set()  # 统计non_termial token 种类
non_terminal_set.add('EMPTY')

def ast_to_seq(binary_tree, run_or_process):
    # 将一个ast首先转换成二叉树，然后对该二叉树进行中序遍历，得到nt_sequence

    temp_terminal_count = Counter()
    temp_non_terminal_set = set()
    def node_to_string(node):
        # 将一个node转换为string，并用map统计各个node的数量信息
        if node == 'EMPTY':
            string_node = 'EMPTY'
        elif node['isTerminal']:  # 如果node为terminal
            # string_node = str(node['id'])  test用
            string_node = str(node['type'])
            if 'value' in node.keys():
                # Note:There are some tokens(like:break .etc）do not contains 'value'
                string_node += '=$$=' + str(node['value'])
            temp_terminal_count[string_node] += 1

        else:  # 如果是non-terminal
            #string_node = str(node['id']) test用
            string_node = str(node['type']) + '=$$=' + \
                str(node['hasSibling']) + '=$$=' + \
                str(node['hasNonTerminalChild'])  # 有些non-terminal包含value，探索该value的意义？（value种类非常多）
            temp_non_terminal_set.add(string_node)

        return string_node

    def in_order_traversal(bin_tree, index, node_or_leaf, left_or_right):
        # 对给定的二叉树进行中序遍历，并在中序遍历的时候，生成nt_pair
        node = bin_tree[index]
        # 向左边递归
        if 'left' in node.keys() and not node['isTerminal']:

            # 说明该节点只有right child，添加一个EMPTY的 left child到sequence中
            if node['left'] == 'PAD_EMPTY':
                output.append(('EMPTY', 'EMPTY', 'leaf', 'left'))

            # 根据当前node是node还是leaf分别进行递归
            elif node['left'] != -1:
                next_node = bin_tree[node['left']]
                if (next_node['left'] == -1 and next_node['right'] == -1): # 说明next node是叶子节点
                    in_order_traversal(bin_tree, node['left'], 'leaf', 'left')
                else: # 说明next node是中间节点
                    in_order_traversal(bin_tree, node['left'], 'node', 'left')
            else: # 说明当前节点是叶子节点，递归回溯
                pass

        # 中序遍历中访问该节点
        if node == 'EMPTY' or node['isTerminal'] is True:
        # 该token是terminal，只将其记录到counter中
            node_to_string(node)
        else:
            assert 'isTerminal' in node.keys() and node['isTerminal'] is False
            # 如果该node是non-terminal，并且包含一个terminal 子节点，则和该子节点组成nt_pair保存在output中
            # 否则将nt_pair的T设为字符串EMPTY
            n_pair = node_to_string(node)
            has_terminal_child = False
            for child_index in node['children']:  # 遍历该non-terminal的所有child，分别用所有terminal child构建NT-pair
                if bin_tree[child_index]['isTerminal']:
                    t_pair = node_to_string(bin_tree[child_index])
                    has_terminal_child = True
                    nt_pair = (n_pair, t_pair, node_or_leaf, left_or_right)
                    output.append(nt_pair)
            if not has_terminal_child: # 该nt node不包含任何terminal child
                t_pair = node_to_string('EMPTY')
                nt_pair = (n_pair, t_pair, node_or_leaf, left_or_right)
                output.append(nt_pair)

        # 向右递归
        if node['right'] != -1:
            # 说明该节点只有right child，添加一个EMPTY的 left child到sequence中
            if node['right'] == 'PAD_EMPTY':
                output.append(('EMPTY', 'EMPTY', 'leaf', 'right'))
            elif node['right'] != -1: # 说明当前节点包含右子树
                next_node = bin_tree[node['right']]
            # 根据当前节点是node还是leaf分别进行递归
                if next_node['left'] == -1 and next_node['right'] == -1:
                    in_order_traversal(bin_tree, node['right'], 'leaf', 'right')
                else:
                    in_order_traversal(bin_tree, node['right'], 'node', 'right')

    output = []
    in_order_traversal(binary_tree, 0, 'node', 'left')

    if run_or_process == 'run':
        return output
    elif run_or_process == 'process':
        if len(output) >= time_steps:  # note: 仅将长度大于阈值的ast产生的node统计到counter中
            terminal_count.update(temp_terminal_count)
            non_terminal_set.update(temp_non_terminal_set)
        else:
            output = []
        return output
    else:
        raise KeyError


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
    for index, token in enumerate(list(non_terminal_set)):
        nt_token_to_int[token] = index
        nt_int_to_token[index] = token

    tt_int_to_token[len(tt_int_to_token)] = unknown_token  # terminal中添加UNK
    tt_token_to_int[unknown_token] = len(tt_token_to_int)

    utils.pickle_save(data_parameter_dir,
                [tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token])  # 将映射字典保存到本地
    return tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token


def seq_to_binary_tree(token_seq):
    """将seq转换回binary tree"""
    stack = []
    def reduce(stack):
        # 对stack进行递归reduce
        while stack[-1]['side'] == 'right':
            right_child = stack.pop()
            parent_node = stack.pop()
            left_child = stack.pop()
            parent_node['r_child'] = right_child
            parent_node['l_child'] = left_child
            stack.append(parent_node)

    nt_node, tt_node, node_or_leaf, left_or_right = token_seq[0]
    # 将第一个node单独拿出来，因为需要判断具有下一个token和上一个token是否具有相同的n_node，如果nt相同，则属于同一个nt_node
    node = {'nt_node': nt_node,
            'tt_node': [tt_node], 'type': node_or_leaf, 'side': left_or_right}
    # index = nt_node.split('=$$=')[0] # test用
    # node['id'] = index
    for i in range(1, len(token_seq)):
        nt_node, tt_node, node_or_leaf, left_or_right = token_seq[i]
        if nt_node == node['nt_node']: # 临近的两个token的nt_node相同，则只需要将tt_node加入到上一个token info中
            node['tt_node'].append(tt_node)
        else:
            stack.append(node)
            node = {'nt_node': nt_node,
                    'tt_node': [tt_node],'type':node_or_leaf, 'side': left_or_right}
            # index = nt_node.split('=$$=')[0] # test用
            # node['id'] = index

            if stack[-1]['type'] == 'leaf' and stack[-1]['side'] == 'right':
                reduce(stack)

    stack.append(node) # 将最后一个node放入stack中，并进行最后一次reduce
    reduce(stack)

    assert len(stack) == 1
    return stack[0]





def nt_seq_to_int(time_steps=50, status='train'):
    # 对NT seq进行进一步的处理，首先将每个token转换为number，
    # 然后对于train data和valid data将所有ast-seq extend成一个list 便于训练时的格式转换
    # 对于test data，将所有ast-seq append，保留各个ast的独立seq

    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = \
        pickle.load(open(data_parameter_dir, 'rb'))
    total_num_nt_pair = 0
    if status == 'train':
        num_sub_data = num_sub_train_data
        sub_int_data_dir = sub_int_train_data_dir
        base_num = 0
    elif status == 'valid':
        num_sub_data = num_sub_valid_data
        sub_int_data_dir = sub_int_valid_data_dir
        base_num = num_sub_train_data
    elif status == 'test':
        num_sub_data = num_sub_test_data
        sub_int_data_dir = sub_int_test_data_dir
        base_num = num_sub_train_data + num_sub_valid_data
    else:
        print('ERROR! Unknown commend!!')
        raise KeyError

    def get_subset_data():  # 对每个part的nt_sequence读取并返回，等待进行处理
        for i in range(base_num + 1, base_num + num_sub_data + 1):
            data_path = sub_data_dir + 'sub_part{}.json'.format(i)
            data = utils.pickle_load(data_path)
            yield (i, data)

    subset_generator = get_subset_data()
    for index, data in subset_generator:
        data_seq = []
        for one_ast in data:  # 将每个nt_seq进行截取，并encode成integer，然后保存
            if len(one_ast) < time_steps:  # 该ast大小不足time step 舍去
                continue
            try:
                nt_int_seq = []
                for n, t, node_or_leaf, left_or_right in one_ast:
                    int_n = nt_token_to_int.get(n, nt_token_to_int['EMPTY'])
                    int_t = tt_token_to_int.get(t, tt_token_to_int[unknown_token])
                    if node_or_leaf == 'node':
                        int_node_or_leaf = 0
                    else:
                        int_node_or_leaf = 1
                    if left_or_right == 'left':
                        int_left_or_right = 0
                    else:
                        int_left_or_right = 1
                    one_pair = (int_n, int_t, int_node_or_leaf, int_left_or_right)
                    nt_int_seq.append(one_pair)
            except KeyError as error:
                print(error)
                continue
            # 在train和valid中，是直接将所有ast-seq extend到一起，在test中，保留各个ast-seq的独立
            if status == 'test':
                data_seq.append(nt_int_seq)
                total_num_nt_pair += len(nt_int_seq)
            else:
                data_seq.extend(nt_int_seq)
                total_num_nt_pair += len(nt_int_seq)

        one_sub_int_data_dir = sub_int_data_dir + 'int_part{}.json'.format(index)
        utils.pickle_save(one_sub_int_data_dir, data_seq)
    # There are 168377411 nt_pair in train dataset...
    # There are 7962311 nt_pair in valid dataset...
    print('There are {} nt_pair in {} dataset...'.format(total_num_nt_pair, status))





if __name__ == '__main__':
    # 测试用
    # import examples
    # ast_example = examples.ast_example
    # binary_tree = bulid_binary_tree(ast_example)
    # token_seq = ast_to_seq(binary_tree, run_or_process='run')
    # for i in token_seq:
    #     print(i)
    #
    # rebuild_binary = seq_to_binary_tree(token_seq)
    # print('\n result')
    # print(rebuild_binary)

    data_process(train_or_test='train')
    data_process(train_or_test='test')
    nt_seq_to_int(status='test')


