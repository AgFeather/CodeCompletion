import tensorflow as tf
import numpy as np
import time
import random
import pickle

from lstm_model import RnnModel
from data_generator import DataGenerator
from setting import Setting
import utils

test_setting = Setting()
model_save_dir = test_setting.lstm_model_save_dir
test_log_dir = test_setting.lstm_test_log_dir

num_subset_test_data = test_setting.num_sub_test_data
seq_per_subset = test_setting.num_seq_per_subset
show_every_n = test_setting.test_show


class CodeCompletion(object):
    """test code completion performance"""
    def __init__(self,
                 num_ntoken,
                 num_ttoken,):
        self.model = RnnModel(num_ntoken, num_ttoken, is_training=False)
        self.sess = tf.Session()
        self.last_chackpoints = tf.train.latest_checkpoint(
            checkpoint_dir=model_save_dir)
        self.test_log(self.last_chackpoints + 'has been used...')
        saver = tf.train.Saver()
        saver.restore(self.sess, self.last_chackpoints)
        self.log_file = open(test_log_dir, 'w')

    # query test
    def eval(self, prefix, suffix):
        """ Query one source code file,
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        """
        new_state = self.sess.run(self.model.init_state)
        test_batch = self.generator.get_test_batch(prefix)
        n_prediction, t_prediction = None, None
        for nt_token, tt_token in test_batch:
            feed = {self.model.n_input: nt_token,
                    self.model.t_input: tt_token,
                    self.model.keep_prob: 1.,
                    self.model.lstm_state: new_state}
            n_prediction, t_prediction, new_state = self.sess.run(
                [self.model.n_output, self.model.t_output, self.model.final_state], feed_dict=feed)

        assert n_prediction is not None and t_prediction is not None
        n_prediction = n_prediction[-1, :]
        t_prediction = t_prediction[-1, :]
        n_prediction = np.argmax(n_prediction)
        t_prediction = np.argmax(t_prediction)
        return n_prediction, t_prediction

    def query(self, token_sequence):
        prefix, expectation, suffix = self.create_hole(token_sequence)  # 随机在sequence中创建一个hole
        n_expectation, t_expectation = expectation[0]
        n_prediction, t_prediction = self.eval(prefix, suffix)

        if self.token_equal(n_prediction, n_expectation):
            self.nt_correct_count += 1
        else:
            temp_error_dic = {}
            temp_error_dic['ast_prefix'] = prefix
            temp_error_dic['ast_expectation'] = expectation
            temp_error_dic['ast_suffix'] = suffix
            temp_error_dic['expectation'] = n_expectation
            temp_error_dic['prediction'] = n_prediction
            self.nt_error_log.append(temp_error_dic)

        if self.token_equal(t_prediction, t_expectation):
            self.tt_correct_count += 1
        else:
            temp_error_dic = {}
            temp_error_dic['ast_prefix'] = prefix
            temp_error_dic['ast_expectation'] = expectation
            temp_error_dic['ast_suffix'] = suffix
            temp_error_dic['expectation'] = t_expectation
            temp_error_dic['prediction'] = t_prediction
            self.tt_error_log.append(temp_error_dic)

    def test_model(self):
        """Test model with the whole test dataset, it will call self.query() for each test case"""
        self.test_log('test phase is beginning...')
        start_time = time.time()
        total_tt_accuracy = 0.0
        total_nt_accuracy = 0.0
        self.nt_error_log = []
        self.tt_error_log = []
        test_times = 2000
        test_step = 0
        self.generator = DataGenerator()
        sub_data_generator = self.generator.get_test_subset_data()
        for index, subset_test_data in sub_data_generator:  # 遍历每个sub test dataset
            self.tt_correct_count = 0.0
            self.nt_correct_count = 0.0
            one_test_start_time = time.time()
            for token_sequence in subset_test_data:  # 遍历该subset中每个nt token sequence
                test_step += 1
                self.query(token_sequence)

                if test_step % show_every_n == 0:
                    one_test_end_time = time.time()
                    duration = (one_test_end_time - one_test_start_time) / show_every_n
                    one_test_start_time = one_test_end_time
                    sub_nt_accuracy = self.nt_correct_count / test_step
                    sub_tt_accuracy = self.tt_correct_count / test_step
                    log_info = 'test step:{}  '.format(test_step) + \
                               'nt_accuracy:{:.2f}%  '.format(sub_nt_accuracy * 100) + \
                               'tt_accuracy:{:.2f}%  '.format(sub_tt_accuracy * 100) + \
                               'average time cost:{:.2f}s  '.format(duration)
                    self.test_log(log_info)

                if test_step >= test_times:
                    break

            sub_nt_accuracy = self.nt_correct_count / test_step
            sub_tt_accuracy = self.tt_correct_count / test_step
            total_nt_accuracy += sub_nt_accuracy
            total_tt_accuracy += sub_tt_accuracy

            end_time = time.time()
            log_info = '{}th subset of test data  '.format(index) + \
                'there are {} nt_sequence to test  '.format(len(subset_test_data)) + \
                'total time cost of this subset: {:.2f}s  '.format(end_time - start_time) + \
                'average time cost per case: {:.2f}s  '.format((end_time-start_time)/seq_per_subset) + \
                'accuracy of non-terminal token: {:.2f}%  '.format(sub_nt_accuracy*100) + \
                'accuracy of terminal token: {:.2f}%  '.format(sub_tt_accuracy*100)
            self.test_log(log_info)

        error_prediction_dir = 'error_prediction_info.p'
        file = open(error_prediction_dir, 'wb')
        pickle.load((self.nt_error_log, self.tt_error_log), file)
        print(error_prediction_dir, 'has been saved...')
        return total_nt_accuracy, total_tt_accuracy

    def top_k_predict(self, prefix, define_k=5):
        """给出top k预测的index，以及对应的概率"""
        # todo 没有测试
        new_state = self.sess.run(self.model.init_state)
        test_batch = self.generator.get_test_batch(prefix)
        n_prediction, t_prediction = None, None
        for nt_token, tt_token in test_batch:
            feed = {self.model.n_input: nt_token,
                    self.model.t_input: tt_token,
                    self.model.keep_prob: 1.,
                    self.model.lstm_state: new_state}
            n_prediction, t_prediction, new_state = self.sess.run(
                [self.model.n_output, self.model.t_output, self.model.final_state], feed_dict=feed)

        assert n_prediction is not None and t_prediction is not None
        n_prediction = n_prediction[-1, :]
        t_prediction = t_prediction[-1, :]
        n_top_k_prediction = (-n_prediction).argsort()[:define_k]
        t_top_k_prediction = (-t_prediction).argsort()[:define_k]
        n_top_k_possiblity = n_prediction[n_top_k_prediction]
        t_top_k_possiblity = t_prediction[t_top_k_prediction]
        return n_top_k_prediction, n_top_k_possiblity, t_top_k_prediction, t_top_k_possiblity

    def int_to_token(self, n_int, t_int):
        """将以int形式表示的n_token和t_token还原成对应的token信息"""
        # todo 没有测试
        tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = utils.load_dict_parameter()
        n_token = nt_int_to_token[n_int].split(test_setting.split_token)
        t_token = tt_int_to_token[t_int].split(test_setting.split_token)
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

    def token_equal(self, prediction, expectation):
        if(prediction == expectation):
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



if __name__ == '__main__':
    num_ntoken = test_setting.num_non_terminal
    num_ttoken = test_setting.num_terminal
    test_model = CodeCompletion(num_ntoken, num_ttoken)
    nt_accuracy, tt_accuracy = test_model.test_model()