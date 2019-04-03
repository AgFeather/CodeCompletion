import tensorflow as tf
import random
import time
import pickle

from nn_model.lstm_model import RnnModel
from data_generator import DataGenerator
import utils
from setting import Setting


test_setting = Setting()
model_save_dir = test_setting.lstm_model_save_dir
show_every_n = test_setting.test_show
completion_log_dir = test_setting.lstm_completion_log_dir
unknown_token = test_setting.unknown_token
define_topk = test_setting.define_topk


class CodeCompletion(object):
    """test the performance of code completion, Creating a random hole in the given nt-sequence,
    and the model will return it's prediction, then calculate the accuracy"""

    def __init__(self,
                 num_ntoken,
                 num_ttoken,):
        self.model = RnnModel(num_ntoken, num_ttoken, is_training=False)
        self.log_file = open(completion_log_dir, 'w')
        self.tt_token_to_int, self.tt_int_to_token, self.nt_token_to_int, self.nt_int_to_token = \
            utils.load_dict_parameter(is_lower=False)
        self.session = tf.Session()
        checkpoints_path = tf.train.latest_checkpoint(model_save_dir)
        saver = tf.train.Saver()
        self.generator = DataGenerator()
        saver.restore(self.session, checkpoints_path)
        self.test_log(checkpoints_path + ' is using...')

        self.get_identifer_set()

    def get_identifer_set(self):
        """统计所有Identifier terminal tokens对应的index"""
        self.identifier_set = set()
        for index, token in self.tt_int_to_token.items():
            type_info = token.split('=$$=')[0]
            if type_info == 'Identifier':
                self.identifier_set.add(index)
        print('There are {} kinds of identifier in vocabulary'.format(len(self.identifier_set)))

    def predict(self, int_nt_seq, topk=3, next_n=1):
        """对外接口，先将所有pre-context转换为int，然后使用trained model进行evaluate"""
        if topk == define_topk:
            eval_func = self.eval
        else:
            eval_func = self.eval_without_define_k

        topk_token_pairs = []
        topk_pairs_poss = []
        for _ in range(next_n):
            n_topk_pred, n_topk_poss, t_topk_pred, t_topk_poss = eval_func(int_nt_seq, topk)

            topk_token_pairs.append([self.int_to_token(n_int ,t_int)
                                for n_int, t_int in zip(n_topk_pred, t_topk_pred)])
            topk_pairs_poss.append([(n_poss, t_poss)
                               for n_poss, t_poss in zip(n_topk_poss, t_topk_poss)])
            int_nt_seq = [(n_topk_poss[0], t_topk_pred[0])]

        return topk_token_pairs, topk_pairs_poss

    def eval(self, prefix, next_n=5):
        """ evaluate one source code file, return the top k prediction and it's possibilities,
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        """
        lstm_state = self.session.run(self.model.init_state)
        test_batch = self.generator.get_test_batch(prefix)
        n_topk_pred, n_topk_poss, t_topk_pred, t_topk_poss = [], [], [], []
        for nt_token, tt_token in test_batch:
            feed = {self.model.n_input: nt_token,
                    self.model.t_input: tt_token,
                    self.model.keep_prob: 1.,
                    self.model.lstm_state: lstm_state}
            n_pred, n_poss, t_pred, t_poss, lstm_state = self.session.run([
                self.model.n_topk_pred, self.model.n_topk_poss, self.model.t_topk_pred,
                self.model.t_topk_poss, self.model.final_state], feed_dict=feed)

            n_topk_pred = n_pred[-1, :]
            n_topk_poss = n_poss[-1, :]
            t_topk_pred = t_pred[-1, :]
            t_topk_poss = t_poss[-1, :]

        return n_topk_pred, n_topk_poss, t_topk_pred, t_topk_poss

    def eval_without_define_k(self, prefix, topk):
        """ evaluate one source code file, return the top k prediction and it's possibilities,
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        """
        lstm_state = self.session.run(self.model.init_state)
        test_batch = self.generator.get_test_batch(prefix)
        n_prediction, t_prediction = None, None
        for nt_token, tt_token in test_batch:
            feed = {self.model.n_input: nt_token,
                    self.model.t_input: tt_token,
                    self.model.keep_prob: 1.,
                    self.model.lstm_state: lstm_state}
            n_prediction, t_prediction, lstm_state = self.session.run(
                [self.model.n_output, self.model.t_output, self.model.final_state], feed_dict=feed)

        assert n_prediction is not None and t_prediction is not None
        n_prediction = n_prediction[-1, :]
        t_prediction = t_prediction[-1, :]
        n_topk_pred = (-n_prediction).argsort()[:topk]
        t_topk_pred = (-t_prediction).argsort()[:topk]
        n_topk_poss = n_prediction[n_topk_pred]
        t_topk_poss = t_prediction[t_topk_pred]

        return n_topk_pred, n_topk_poss, t_topk_pred, t_topk_poss

    def query(self, token_sequence, topk=3):
        prefix, expectation, suffix = self.create_hole(token_sequence)  # 随机在sequence中创建一个hole
        print(expectation)
        n_expectation, t_expectation = expectation[0]
        if define_topk == topk:
            n_topk_pred, n_topk_poss, t_topk_pred, t_topk_poss = self.eval(prefix)
        else:
            n_topk_pred, n_topk_poss, t_topk_pred, t_topk_poss = self.eval_without_define_k(prefix, topk)

        if self.top_one_equal(n_topk_pred, n_expectation):
            self.nt_correct_count += 1

        if self.top_one_equal(t_topk_pred, t_expectation):
            self.tt_correct_count += 1

        if self.topk_equal(n_topk_pred, n_expectation):
            self.topk_nt_correct_count += 1
        if self.topk_equal(t_topk_pred, t_expectation):
            self.topk_tt_correct_count += 1

        if t_expectation in self.identifier_set:
            # 如果target terminal token是identifier，就进行统计
            if self.top_one_equal(t_topk_pred, t_expectation):
                self.identifier_correct_count += 1
            else:
                self.identifier_incorrect_count += 1

    def completion_test(self, topk=3):
        """Test model with the whole test dataset
        first, it will create a hole in the test ast_nt_sequence randomly,
        then, it will call self.query() for each test case,
        finally, the statistical accuracy will be update
        """
        self.test_log('test phase is beginning...')
        start_time = time.time()
        self.tt_correct_count = 0.0
        self.nt_correct_count = 0.0
        self.topk_nt_correct_count = 0.0
        self.topk_tt_correct_count = 0.0
        self.identifier_correct_count = 0 # 用以计算对Identifier的预测准确率
        self.identifier_incorrect_count = 0
        self.identifier_accu_list = []
        test_times = 10000
        test_step = 0
        self.generator = DataGenerator()
        sub_data_generator = self.generator.get_test_subset_data()

        for index, subset_test_data in sub_data_generator:  # 遍历每个sub test dataset
            one_test_start_time = time.time()
            for token_sequence in subset_test_data:  # 遍历该subset中每个nt token sequence
                test_step += 1
                # 对一个ast sequence进行test
                self.query(token_sequence, topk=topk)
                if test_step % show_every_n == 0:
                    one_test_end_time = time.time()
                    duration = (one_test_end_time - one_test_start_time) / show_every_n
                    one_test_start_time = one_test_end_time
                    identifier_accu = (self.identifier_correct_count+1) / (self.identifier_correct_count + self.identifier_incorrect_count+1)
                    self.identifier_accu_list.append(identifier_accu)
                    nt_accu = self.nt_correct_count / test_step
                    tt_accu = self.tt_correct_count / test_step
                    topk_nt_accu = self.topk_nt_correct_count / test_step
                    topk_tt_accu = self.topk_tt_correct_count / test_step
                    log_info = 'test step:{}  '.format(test_step) + \
                               'nt_accuracy:{:.2f}%  '.format(nt_accu * 100) + \
                               'tt_accuracy:{:.2f}%  '.format(tt_accu * 100) + \
                               'nt_top{}_accuracy:{:.2f}%  '.format(define_topk, topk_nt_accu * 100) + \
                               'tt_top{}_accuracy:{:.2f}%  '.format(define_topk, topk_tt_accu * 100) + \
                               'identifier accuracy:{:.2f}% '.format(identifier_accu * 100) + \
                               'average time cost:{:.2f}s  '.format(duration)
                    self.test_log(log_info)

                if test_step >= test_times:
                    break

            nt_accuracy = self.nt_correct_count / test_step
            tt_accuracy = self.tt_correct_count / test_step
            topk_nt_accu = self.topk_nt_correct_count / test_step
            topk_tt_accu = self.topk_tt_correct_count / test_step
            end_time = time.time()
            log_info = '{}th subset of test data  '.format(index) + \
                       'there are {} nt_sequence to test  '.format(test_step) + \
                       'accuracy of non-terminal token: {:.2f}%  '.format(nt_accuracy * 100) + \
                       'accuracy of terminal token: {:.2f}%  '.format(tt_accuracy * 100) + \
                       'top{} accuracy of non-terminal:{:.2f}%  '.format(define_topk, topk_nt_accu * 100) + \
                       'top{} accuracy of terminal:{:.2f}%  '.format(define_topk, topk_tt_accu * 100) + \
                       'total time cost of this subset: {:.2f}s  '.format(end_time - start_time) + \
                       'average time cost per case: {:.2f}s  '.format((end_time - start_time) / test_step)
            self.test_log(log_info)

            file = open('identifier_prediction_accu.pkl', 'wb')
            pickle.dump(self.identifier_accu_list, file)



    def top_one_equal(self, prediction, expectation):
        if prediction[0] == expectation:
            return True
        return False

    def topk_equal(self, prediction, expectation):
        if expectation in prediction:
            return True
        return False

    def create_hole(self, nt_token_seq, hole_size=1):
        hole_start_index = random.randint(
            len(nt_token_seq) // 2, len(nt_token_seq) - hole_size)
        hole_end_index = hole_start_index + hole_size
        prefix = nt_token_seq[0:hole_start_index]
        expectation = nt_token_seq[hole_start_index:hole_end_index]
        suffix = nt_token_seq[hole_end_index:-1]
        return prefix, expectation, suffix

    def test_log(self, log_info):
        self.log_file.write(log_info)
        self.log_file.write('\n')
        print(log_info)

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
    test_model = CodeCompletion(num_ntoken, num_ttoken)
    test_model.completion_test(topk=3)
