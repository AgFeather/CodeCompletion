"""提出的一个新的数据处理方法，该处理方法的优点是可以将生成的token sequence 转换回AST，
这样就实现了真正的code completion"""


import json
import sys
from collections import Counter
from json.decoder import JSONDecodeError

from setting import Setting
import utils



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

most_common_termial_num = 30000
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
            binary_tree = bulid_binary_tree(ast)  # AST to binary tree
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




def bulid_binary_tree(ast):
    """transform the AST(one node may have several childNodes) to
    Left-Child-Right-Sibling(LCRS) binary tree."""
    brother_map = {0: -1}
    for index, node in enumerate(ast):  # 顺序遍历每个AST中的node

        if not isinstance(node, dict) and node == 0:  # AST中最后添加一个'EOF’标识
            del ast[index]
            break

        node['right'] = brother_map.get(node['id'], -1)
        node['left'] = -1

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
        # 将一个node转换为string
        if node == 'EMPTY':
            string_node = 'EMPTY'
            temp_terminal_count[string_node] += 1
        elif node['isTerminal']:  # 如果node为terminal
            string_node = str(node['id']) +'=$$=' + str(node['type'])
            if 'value' in node.keys():
                # Note:There are some tokens(like:break .etc）do not contains 'value'
                string_node += '=$$=' + str(node['value'])
            temp_terminal_count[string_node] += 1

        else:  # 如果是non-terminal

            string_node = str(node['id']) +'=$$='+ str(node['type']) + '=$$=' + \
                str(node['hasSibling']) + '=$$=' + \
                str(node['hasNonTerminalChild'])  # 有些non-terminal包含value，探索该value的意义？（value种类非常多）
            temp_non_terminal_set.add(string_node)

        return string_node

    def in_order_traversal(bin_tree, index, node_or_leaf, left_or_right):
        # 对给定的二叉树进行中序遍历，并在中序遍历的时候，生成nt_pair
        node = bin_tree[index]
        if 'left' in node.keys() and not node['isTerminal']:
            next_node = bin_tree[node['left']]
            left_child = bin_tree[next_node['left']]
            right_child = bin_tree[next_node['right']]

            if (next_node['left'] == -1 and next_node['right'] == -1) or (left_child['isTerminal'] and right_child['isTerminal']):
                in_order_traversal(bin_tree, node['left'], 'leaf', 'left')
            else:
                in_order_traversal(bin_tree, node['left'], 'node', 'left')

        if node == 'EMPTY' or node['isTerminal'] is True:  # 该token是terminal，只将其记录到counter中
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


        if node['right'] != -1:  # 遍历right side
            in_order_traversal(bin_tree, node['right'], 'node', 'right')

    output = []
    in_order_traversal(binary_tree, 0, 'node', 'left')

    if run_or_process == 'run':
        return output

    # if len(output) >= time_steps:  # note: 仅将长度大于阈值的ast产生的node统计到counter中
    #     terminal_count.update(temp_terminal_count)
    #     non_terminal_set.update(temp_non_terminal_set)
    # else:
    #     output = []
    return output


def seq_to_binary_tree(token_seq):
    """将seq转换回binary tree"""

    def reduce(stack):
        while stack[-1]['side'] == 'right':
            right_child = stack.pop()
            parent_node = stack.pop()
            left_child = stack.pop()
            parent_node['right'] = right_child
            parent_node['left'] = left_child
            stack.append(parent_node)

    stack = []
    for index, (nt_node, tt_node, node_or_leaf, left_or_right) in enumerate(token_seq):
        if node_or_leaf == 'node':
            node = {'id':index, 'node_info':nt_node, 'side':left_or_right}
            stack.append(node)
        if node_or_leaf == 'leaf':
            if left_or_right == 'left':
                node = {'node_info': nt_node, 'side': left_or_right}
                stack.append(node)
            elif left_or_right == 'right':
                node = {'node_info': nt_node, 'side': left_or_right}
                stack.append(node)
                reduce(stack)










if __name__ == '__main__':
    import examples
    ast_example = examples.ast_example
    binary_tree = bulid_binary_tree(ast_example)
    token_seq = ast_to_seq(binary_tree)

    for a in token_seq:
        print(a)
    # rebuild_binary = seq_to_binary_tree(token_seq)

