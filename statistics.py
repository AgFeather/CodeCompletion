"""通过遍历dataset，对dataset进行统计，并保存统计信息"""



import json
import sys
from collections import Counter
from json.decoder import JSONDecodeError
import pickle

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


def dataset_traversal(is_training=True):
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

    file = open(data_path, 'r')
    for i in range(1, total_size + 1):
        try:
            line = file.readline()  # read a lind from file(one ast)
            ast = json.loads(line)  # transform it to json format
            #count_tokens_types(ast)
            count_num_terminal(ast)
        except UnicodeDecodeError as error:  # arise by readline
            print(error)
        except JSONDecodeError as error:  # arise by json_load
            print(error)
        except RecursionError as error:
            print(error)
        except BaseException as error:
            print(error)
            print('other unknown error, plesae check the code')

    terminal_num_counter_path = 'terminal_num_counter.pkl'
    pickle.dump(terminal_count, open(terminal_num_counter_path, 'wb'))
    # count_info_path = 'count_statistic.pkl'
    # pickle.dump([nt_count, tt_count], open(count_info_path, 'wb'))




terminal_count = Counter()  # 统计每个terminal token的出现次数
def count_num_terminal(ast):
    def node_to_string(node):
        # 将一个node转换为string
        string_node = str(node['type'])
        if 'value' in node.keys():
            # Note:There are some tokens(like:break .etc）do not contains 'value'
            string_node += '=$$=' + str(node['value'])
        terminal_count[string_node] += 1

    for node in ast:
        if node == 0:
            break
        if 'children' not in node.keys() or len(node['children']) == 0:
            node_to_string(node)




tt_count = Counter()  # 统计每种terminal token的出现次数
nt_count = Counter()  # 统计每种non-terminal token的出现次数
def count_tokens_types(ast):
    """对输入进来的ast进行数据统计，统计每种terminal token的数量"""
    for node in ast:
        if node == 0:
            continue
        type_info = str(node['type'])
        if 'children' not in node.keys() or len(node['children']) == 0: # 说明该节点是non-terminal
            tt_count[type_info] += 1
        else:
            nt_count[type_info] += 1


def show_token_statistic(nt_clip=17, tt_clip=6):
    import matplotlib.pyplot as plt
    nt_counter, tt_counter = pickle.load(open('count_statistic.pkl', 'rb'))
    temp_nt_x = [k for k, v in nt_counter.most_common()]
    temp_nt_y = [v for k, v in nt_counter.most_common()]
    nt_x = temp_nt_x[:nt_clip]
    nt_y = temp_nt_y[:nt_clip]
    other_non_terminal = 0
    for i in range(nt_clip, len(temp_nt_y)):
        other_non_terminal += temp_nt_y[i]
    nt_x.append('other non-terminal')
    nt_y.append(other_non_terminal)

    temp_tt_x = [k for k, v in tt_counter.most_common()]
    temp_tt_y = [v for k, v in tt_counter.most_common()]
    tt_x = temp_tt_x[:tt_clip]
    tt_y = temp_tt_y[:tt_clip]
    other_terminal = 0
    for i in range(tt_clip, len(temp_tt_y)):
        other_terminal += temp_tt_y[i]
    tt_x.append('other terminal')
    tt_y.append(other_terminal)

    plt.figure(figsize=(20, 10))
    plt.bar(nt_x, nt_y)
    plt.xticks(rotation=60)
    plt.xlabel('non-terminal token')
    plt.ylabel('number of each token')

    plt.figure(figsize=(20, 10))
    plt.bar(tt_x, tt_y)
    plt.xlabel('terminal token')
    plt.ylabel('number of each token')
    plt.show()












if __name__ == '__main__':
    dataset_traversal()
    #show_token_statistic()
