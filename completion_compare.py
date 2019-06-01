import tensorflow as tf
import random
from nn_model.lstm_model import RnnModel as orgin_model
from nn_model.lstm_node2vec import LSTM_Node_Embedding as embedding_model
from data_generator import DataGenerator
import utils
from setting import Setting

"""在一个ast中随机创建一个hole，然后分别交给两个lstm进行预测，找到两者不同的地方"""


test_setting = Setting()
current_time = test_setting.current_time
origin_trained_model_dir = 'trained_model/origin_lstm/'
embedding_trained_model_dir = 'trained_model/lstm_with_node2vec/'


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

    def query(self,ast_index, token_sequence):
        prefix, expectation, suffix = self.create_hole(token_sequence)  # 随机在sequence中创建一个hole
        n_expectation, t_expectation = expectation[0]
        origin_n_pred, origin_t_pred = self.eval_origin_model(prefix)
        embedding_n_pred, embedding_t_pred = self.eval_embedding_model(prefix)
        if origin_n_pred != embedding_n_pred:
            # 两个模型对nt token的预测不相同，记录
            self.record_incorrect('n', ast_index, token_sequence,
                                  len(prefix), origin_n_pred, embedding_n_pred, n_expectation)
            print('There is a n-token predict wrong, ast index:{}'.format(ast_index))
        if origin_t_pred != embedding_t_pred:
            self.record_incorrect('t', ast_index, token_sequence,
                                  len(prefix), origin_t_pred, embedding_t_pred, t_expectation)
            print('There is a t-token predict wrong, ast index:{}'.format(ast_index))

    def record_incorrect(self,n_or_t, ast_index, nt_sequence, token_index,
                         origin_pred, embed_pred, expectation):
        # 记录预测失误的信息到本地
        # 整个文件的保存格式为：（ast_index,nt_sequence, node_index, origin_pred, embed_pred, true_label）
        nt_sequence = [(self.nt_int_to_token[n], self.tt_int_to_token[t]) for n, t in nt_sequence]
        if n_or_t == 'n':
            origin_pred = self.nt_int_to_token[origin_pred]
            embed_pred = self.nt_int_to_token[embed_pred]
            expectation = self.nt_int_to_token[expectation]
            info = 'ast_index;' + str(ast_index) + '\n' + \
                   'nt_sequence;' + str(nt_sequence) + '\n' + \
                   'token_index;' + str(token_index) + '\n' + \
                   'ori_pred;' + str(origin_pred) + '\n' + \
                   'embed_pred;' + str(embed_pred) + '\n' + \
                   'expectation;' + str(expectation) + '\n'
            self.n_incorrect.write(info)
            self.n_incorrect.flush()
        elif n_or_t == 't':
            origin_pred = self.tt_int_to_token[origin_pred]
            embed_pred = self.tt_int_to_token[embed_pred]
            expectation = self.tt_int_to_token[expectation]
            info = 'ast_index;' + str(ast_index) + '\n' + \
                   'nt_sequence;' + str(nt_sequence) + '\n' + \
                   'token_index;' + str(token_index) + '\n' + \
                   'ori_pred;' + str(origin_pred) + '\n' + \
                   'embed_pred;' + str(embed_pred) + '\n' + \
                   'expectation;' + str(expectation) + '\n'
            self.t_incorrect.write(info)
            self.t_incorrect.flush()
        else:
            raise KeyError

    def completion_test(self):
        """Test model with the whole test dataset
        first, it will create a hole in the test ast_nt_sequence randomly,
        then, it will call self.query() for each test case"""
        test_times = 10000
        test_step = 0
        self.generator = DataGenerator()
        sub_data_generator = self.generator.get_test_subset_data()

        for index, subset_test_data in sub_data_generator:  # 遍历每个sub test dataset
            for token_sequence in subset_test_data:  # 遍历该subset中每个nt token sequence
                # 对一个ast sequence进行test
                self.query(test_step, token_sequence)
                test_step += 1

    def create_hole(self, nt_token_seq, hole_size=1):
        hole_start_index = random.randint(
            len(nt_token_seq) // 2, len(nt_token_seq) - hole_size)
        hole_end_index = hole_start_index + hole_size
        prefix = nt_token_seq[0:hole_start_index]
        expectation = nt_token_seq[hole_start_index:hole_end_index]
        suffix = nt_token_seq[hole_end_index:-1]
        return prefix, expectation, suffix

    def int_to_token(self, n_int, t_int):
        """将以int形式表示的n_token和t_token还原成对应的token信息"""
        n_token = self.nt_int_to_token[n_int].split(test_setting.split_token)
        t_token = self.tt_int_to_token[t_int].split(test_setting.split_token)
        n_token_present = {}
        t_token_present = {}
        for index, value in enumerate(n_token):
            n_token_present[index] = value
        if t_token[0] == test_setting.unknown_token:
            t_token_present[0] = 'Unknown Token'
        else:
            for index, value in enumerate(t_token):
                t_token_present[index] = value
        return n_token_present, t_token_present




if __name__ == '__main__':
    num_ntoken = test_setting.num_non_terminal
    num_ttoken = test_setting.num_terminal
    test_model = CompletionCompare(num_ntoken, num_ttoken)
    test_model.completion_test()
