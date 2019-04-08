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
            # todo 暂时注释掉
            #add_two_bits_info(ast, node, brother_map)  # 向每个节点添加两bit的额外信息
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

                #print(node['id'])
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
non_terminal_set = set()  # 统计non_termial token 种类

def ast_to_seq(binary_tree, run_or_process='process'):
    # 将一个ast首先转换成二叉树，然后对该二叉树进行中序遍历，得到nt_sequence
    temp_terminal_count = Counter()
    temp_non_terminal_set = set()
    def node_to_string(node):
        # 将一个node转换为string，并用map统计各个node的数量信息
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

            string_node = str(node['id'])\
                #           +'=$$='+ str(node['type']) + '=$$=' + \
                # str(node['hasSibling']) + '=$$=' + \
                # str(node['hasNonTerminalChild'])  # 有些non-terminal包含value，探索该value的意义？（value种类非常多）
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

    # if len(output) >= time_steps:  # note: 仅将长度大于阈值的ast产生的node统计到counter中
    #     terminal_count.update(temp_terminal_count)
    #     non_terminal_set.update(temp_non_terminal_set)
    # else:
    #     output = []
    return output


def seq_to_binary_tree(token_seq):
    """将seq转换回binary tree"""
    stack = []
    def reduce(stack):
        # 对stack进行递归reduce
        #print('\nreduce')
        while stack[-1]['side'] == 'right':
            right_child = stack.pop()
            parent_node = stack.pop()
            left_child = stack.pop()
            parent_node['r_child'] = right_child
            parent_node['l_child'] = left_child
            stack.append(parent_node)

    nt_node, tt_node, node_or_leaf, left_or_right = token_seq[0]
    index = nt_node.split('=$$=')[0]
    node = {'id': index, 'nt_node': nt_node, 'tt_node': [tt_node], 'type': node_or_leaf, 'side': left_or_right}
    for i in range(1, len(token_seq)):
        nt_node, tt_node, node_or_leaf, left_or_right = token_seq[i]
        index = nt_node.split('=$$=')[0]
        if nt_node == node['nt_node']:
            node['tt_node'].append(tt_node)
        else:
            stack.append(node)
            node = {'id': index, 'nt_node': nt_node, 'tt_node': [tt_node],'type':node_or_leaf, 'side': left_or_right}

            if stack[-1]['type'] == 'leaf' and stack[-1]['side'] == 'right':
                reduce(stack)

   # 将最后一个node放入stack中，并进行最后一次reduce
    stack.append(node)
    reduce(stack)

    print(len(stack))
    # todo：检查正确性
    for s in stack:
        print(s)
    return stack[0]










if __name__ == '__main__':
    import examples
    ast_example = examples.ast_example
    binary_tree = bulid_binary_tree(ast_example)
    # for i in binary_tree:
    #     print(i)
    token_seq = ast_to_seq(binary_tree)
    #
    # for a in token_seq:
    #     print(a)
    rebuild_binary = seq_to_binary_tree(token_seq)
    #print(rebuild_binary)

