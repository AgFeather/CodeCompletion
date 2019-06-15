import tensorflow as tf
import random
import sys
import json
from json.decoder import JSONDecodeError
import copy
import pickle

import data_process
from nn_model.lstm_model import RnnModel as orgin_model
from nn_model.lstm_node2vec import LSTM_Node_Embedding as embedding_model
from data_generator import DataGenerator
import utils
from setting import Setting

"""在一个ast中随机创建一个hole，然后分别交给两个lstm进行预测，找到两者不同的地方"""
"""Implement a pipeline: 
choose an AST from the original test dataset,
create a hole (non-terminal hole and its child terminal hole) in this AST, and record its index. 
convert it to a nt-sequence with a hole
and ask two models to predict, record their prediction """


test_setting = Setting()
current_time = test_setting.current_time
origin_trained_model_dir = 'trained_model/origin_lstm/'
embedding_trained_model_dir = 'trained_model/lstm_with_node2vec/'

n_incorrect_count = 0
t_incorrect_count = 0
n_pickle_save_index = 1
t_pickle_save_index = 1

class CompletionCompare(object):
    def __init__(self,
                 num_ntoken,
                 num_ttoken,):
        origin_graph = tf.Graph()
        embedding_graph = tf.Graph()
        with origin_graph.as_default():
            self.origin_model = orgin_model(num_ntoken, num_ttoken, is_training=False)
            origin_checkpoints_path = tf.train.latest_checkpoint(origin_trained_model_dir)
            saver = tf.train.Saver()
            self.origin_session = tf.Session(graph=origin_graph)
            saver.restore(self.origin_session, origin_checkpoints_path)
        with embedding_graph.as_default():
            self.embedding_model = embedding_model(num_ntoken, num_ttoken, is_training=False)
            self.embedding_session = tf.Session(graph=embedding_graph)
            embedding_checkpoints_path = tf.train.latest_checkpoint(embedding_trained_model_dir)
            saver = tf.train.Saver()
            saver.restore(self.embedding_session, embedding_checkpoints_path)

        self.generator = DataGenerator()
        self.tt_token_to_int, self.tt_int_to_token, self.nt_token_to_int, self.nt_int_to_token = \
            utils.load_dict_parameter(is_lower=False)

        self.n_incorrect = open('temp_data/predict_compare/nt_compare'+str(current_time)+'.txt', 'w')
        self.t_incorrect = open('temp_data/predict_compare/tt_compare'+str(current_time)+'.txt', 'w')

        self.n_incorrect_pickle_list = []
        self.t_incorrect_pickle_list = []

    def eval_origin_model(self, prefix):
        """ evaluate one source code file, return the top k prediction and it's possibilities,
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        """
        lstm_state = self.origin_session.run(self.origin_model.init_state)
        test_batch = self.generator.get_test_batch(prefix)
        n_prediction, t_prediction = None, None
        for nt_token, tt_token in test_batch:
            feed = {self.origin_model.n_input: nt_token,
                    self.origin_model.t_input: tt_token,
                    self.origin_model.keep_prob: 1.,
                    self.origin_model.lstm_state: lstm_state}
            n_prediction, t_prediction, lstm_state = self.origin_session.run(
                [self.origin_model.n_output, self.origin_model.t_output, self.origin_model.final_state],
                feed_dict=feed)

        assert n_prediction is not None and t_prediction is not None
        n_prediction = n_prediction[-1, :]
        t_prediction = t_prediction[-1, :]
        n_topk_pred = (-n_prediction).argsort()[0]
        t_topk_pred = (-t_prediction).argsort()[0]

        return n_topk_pred, t_topk_pred


    def eval_embedding_model(self, prefix):
        lstm_state = self.embedding_session.run(self.embedding_model.init_state)
        test_batch = self.generator.get_test_batch(prefix)
        n_prediction, t_prediction = None, None
        for nt_token, tt_token in test_batch:
            feed = {self.embedding_model.n_input: nt_token,
                    self.embedding_model.t_input: tt_token,
                    self.embedding_model.keep_prob: 1.,
                    self.embedding_model.lstm_state: lstm_state}
            n_prediction, t_prediction, lstm_state = self.embedding_session.run(
                [self.embedding_model.n_output, self.embedding_model.t_output, self.embedding_model.final_state],
                feed_dict=feed)

        assert n_prediction is not None and t_prediction is not None
        n_prediction = n_prediction[-1, :]
        t_prediction = t_prediction[-1, :]
        n_topk_pred = (-n_prediction).argsort()[0]
        t_topk_pred = (-t_prediction).argsort()[0]

        return n_topk_pred, t_topk_pred

    def query(self,prefix):
        origin_n_pred, origin_t_pred = self.eval_origin_model(prefix)
        embedding_n_pred, embedding_t_pred = self.eval_embedding_model(prefix)
        return origin_n_pred, origin_t_pred, embedding_n_pred, embedding_t_pred

    def completion_test(self):
        """Test model with the whole test dataset
        first, it will create a hole in the test ast_nt_sequence randomly,
        then, it will call self.query() for each test case"""
        test_times = 10000
        ast_generator = get_one_test_ast()
        for i, ast, prefix, expectation_token, expectation_index in ast_generator:  # 遍历该subset中每个nt token sequence
            # 对一个ast sequence进行test
            origin_n_pred, origin_t_pred, embedding_n_pred, embedding_t_pred = self.query(prefix)
            n_expectation_token, t_expectation_token = expectation_token
            n_expectation_index, t_expectation_index = expectation_index

            # 当两个模型预测不准确时，记录。格式为：(ast, index, ori_pred, embed_pred, expect)
            if origin_n_pred != embedding_n_pred:
                origin_n_pred_token = self.nt_int_to_token[origin_n_pred]
                embedding_n_pred_token = self.nt_int_to_token[embedding_n_pred]

                temp_n_incorrect = {'ast':ast,
                                    'expect_index':n_expectation_index,
                                    'ori_pred':origin_n_pred_token,
                                    'embed_pred':embedding_n_pred_token,
                                    'expectation':n_expectation_token}
                self.n_incorrect_pickle_list.append(temp_n_incorrect)
                global n_incorrect_count
                n_incorrect_count+=1
                if n_incorrect_count % 50 == 0:
                    global n_pickle_save_index
                    with open('temp_data/predict_compare/n_incorrect{}.pkl'.format(n_pickle_save_index), 'wb') as file:
                        pickle.dump(self.n_incorrect_pickle_list, file)
                        n_pickle_save_index += 1
                        self.n_incorrect_pickle_list = []

                info = 'ast;' + str(ast) + '\n' + \
                       'expect_token_index;' + str(n_expectation_index) + '\n' + \
                       'ori_pred;' + str(origin_n_pred_token) + '\n' + \
                       'embed_pred;' + str(embedding_n_pred_token) + '\n' + \
                       'expectation;' + str(n_expectation_token) + '\n'
                self.n_incorrect.write(info)
                self.n_incorrect.flush()
                #print('There is a N token predict wrong, ast index:{}'.format(i), end='   ')
                print('There is a N token predict wrong', end='   ')
                print('Ori predict:{}; Embed predict:{}'.format(origin_n_pred_token, embedding_n_pred_token))

            if origin_t_pred != embedding_t_pred:
                origin_t_pred_token = self.tt_int_to_token[origin_t_pred]
                embedding_t_pred_token = self.tt_int_to_token[embedding_t_pred]

                temp_t_correct = {'ast':ast,
                                  'expect_index':t_expectation_index,
                                  'ori_pred':origin_t_pred_token,
                                  'embed_pred':embedding_t_pred_token,
                                  'expectation':t_expectation_token}
                self.t_incorrect_pickle_list.append(temp_t_correct)
                global t_incorrect_count
                t_incorrect_count += 1
                if t_incorrect_count % 50 == 0:
                    global t_pickle_save_index
                    with open('temp_data/predict_compare/t_incorrect{}.pkl'.format(t_pickle_save_index), 'wb') as file:
                        pickle.dump(self.t_incorrect_pickle_list, file)
                        t_pickle_save_index += 1
                        self.t_incorrect_pickle_list = []

                info = 'ast;' + str(ast) + '\n' + \
                       'expect_token_index;' + str(t_expectation_index) + '\n' + \
                       'ori_pred;' + str(origin_t_pred_token) + '\n' + \
                       'embed_pred;' + str(embedding_t_pred_token) + '\n' + \
                       'expectation;' + str(t_expectation_token) + '\n'
                self.t_incorrect.write(info)
                self.t_incorrect.flush()
                print('There is a T token predict wrong', end='   ')
                print('Ori predict:{}; Embed predict:{}'.format(origin_t_pred_token, embedding_t_pred_token))
            # if i > 10:
            #     break


def get_one_test_ast():
    """"""
    sys.setrecursionlimit(10000)  # 设置递归最大深度
    print('setrecursionlimit == 10000')
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = utils.load_dict_parameter(
        is_lower=False)

    def nt_seq_to_int(nt_sequence):
        """"""
        try:
            int_seq = [(nt_token_to_int[n], tt_token_to_int.get(
                t, tt_token_to_int[unknown_token])) for n, t in nt_sequence]
        except KeyError:
            print('key error')
        else:
            return int_seq

    unknown_token = test_setting.unknown_token
    data_path = test_setting.origin_test_data_dir
    total_size = 50000
    file = open(data_path, 'r')
    for i in range(1, total_size + 1):
        try:
            line = file.readline()  # read a lind from file(one ast)
            ast = json.loads(line)  # transform it to json format
            ori_ast = copy.deepcopy(ast)
            binary_tree = data_process.bulid_binary_tree(ast)  # AST to binary tree
            prefix, expectation, predict_token_index = ast_to_seq(binary_tree)  # binary to nt_sequence
        except UnicodeDecodeError as error:  # arise by readline
            print(error)
        except JSONDecodeError as error:  # arise by json_load
            print(error)
        except RecursionError as error:
            print(error)
        except BaseException:
            pass
            #print('other unknown error, plesae check the code')
        else:
            int_prefix = nt_seq_to_int(prefix)
            if len(int_prefix) != 0:
                yield i, ori_ast, int_prefix, expectation, predict_token_index

def ast_to_seq(binary_tree):
    # 将一个ast首先转换成二叉树，然后对该二叉树进行中序遍历，得到nt_sequence
    def node_to_string(node):
        # 将一个node转换为string
        if node == 'EMPTY':
            string_node = 'EMPTY'
        elif node['isTerminal']:  # 如果node为terminal
            string_node = str(node['type'])
            if 'value' in node.keys():
                # Note:There are some tokens(like:break .etc）do not contains 'value'
                string_node += '=$$=' + str(node['value'])
        else:  # 如果是non-terminal
            # str(node['id']) + 添加上对应的index用以测试
            string_node =  str(node['type']) + '=$$=' + \
                str(node['hasSibling']) + '=$$=' + \
                str(node['hasNonTerminalChild'])  # 有些non-terminal包含value，探索该value的意义？（value种类非常多）

        return string_node

    def in_order_traversal(bin_tree, index):
        # 对给定的二叉树进行中序遍历，并在中序遍历的时候，生成nt_pair
        node = bin_tree[index]
        if 'left' in node.keys() and node['left'] != -1:
            in_order_traversal(bin_tree, node['left'])

        if node == 'EMPTY' or node['isTerminal'] is True:  # 该token是terminal，只将其记录到counter中
            node_to_string(node)
        else:
            assert 'isTerminal' in node.keys() and node['isTerminal'] is False
            n_pair = node_to_string(node)
            has_terminal_child = False
            for child_index in node['children']:  # 遍历该non-terminal的所有child，分别用所有terminal child构建NT-pair
                if bin_tree[child_index]['isTerminal']:
                    t_pair = node_to_string(bin_tree[child_index])
                    has_terminal_child = True
                    nt_pair = (n_pair, t_pair)
                    nt_index = (node['id'], bin_tree[child_index]['id'])
                    nt_sequence.append(nt_pair)
                    index_sequence.append(nt_index)
            if not has_terminal_child: # 该nt node不包含任何terminal child
                t_pair = node_to_string('EMPTY')
                nt_pair = (n_pair, t_pair)
                nt_index = (node['id'], 'EMPTY')
                nt_sequence.append(nt_pair)
                index_sequence.append(nt_index)

        if node['right'] != -1:  # 遍历right side
            in_order_traversal(bin_tree, node['right'])

    nt_sequence = []
    index_sequence = [] # 用以记录nt_sequence中每个token对应在原origin ast 中的索引
    in_order_traversal(binary_tree, 0)
    assert len(nt_sequence) == len(index_sequence)
    hole_index = random.randint(10, len(nt_sequence)-1)
    prefix = nt_sequence[:hole_index]
    expectation = nt_sequence[hole_index]
    suffix = nt_sequence[hole_index:]
    predict_token_index = index_sequence[hole_index]
    return prefix, expectation, predict_token_index



if __name__ == '__main__':
    num_ntoken = test_setting.num_non_terminal
    num_ttoken = test_setting.num_terminal
    test_model = CompletionCompare(num_ntoken, num_ttoken)
    test_model.completion_test()

    # func = get_one_test_ast()
    # for a, b, c in func:
    #     if a > 10:
    #         break
    #     print(c)
