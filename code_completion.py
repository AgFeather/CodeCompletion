import tensorflow as tf
import numpy as np
import time
import random
import pickle

import utils
from lstm_model import RnnModel


test_subset_data_dir = 'split_js_data/eval_data/int_format/'
model_save_dir = 'trained_model/lstm_model/'
curr_time = time.strftime('_%Y_%M_%d_%H', time.localtime())
test_log_dir = 'test_log/lstm_log' + str(curr_time) + '.txt'


num_subset_test_data = 10
seq_per_subset = 5000
num_terminal = 30000


class CodeCompletion(object):
    def __init__(self,
                 num_ntoken,
                 num_ttoken):
        self.model = RnnModel(num_ntoken, num_ttoken, is_training=False)
        self.last_chackpoints = tf.train.latest_checkpoint(
            checkpoint_dir=model_save_dir)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self.last_chackpoints)
        self.log_file = open(test_log_dir, 'w')

    # query test
    def query(self, prefix, suffix):
        '''
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        '''
        new_state = self.sess.run(self.model.init_state)
        n_prediction = None
        t_prediction = None
        for i, (nt_token, tt_token) in enumerate(prefix):
            nt_x = np.zeros((1, 1), dtype=np.int32)
            tt_x = np.zeros((1, 1), dtype=np.int32)
            nt_x[0, 0] = nt_token
            tt_x[0, 0] = tt_token
            feed = {self.model.n_input: nt_x,
                    self.model.t_input: tt_x,
                    self.model.keep_prob: 1.,
                    self.model.init_state: new_state}
            n_prediction, t_prediction, new_state = self.sess.run(
                [self.model.n_output, self.model.t_output, self.model.final_state], feed_dict=feed)

        assert n_prediction is not None and t_prediction is not None
        n_prediction = np.argmax(n_prediction)
        t_prediction = np.argmax(t_prediction)
        return n_prediction, t_prediction

    def subset_generator(self):
        for index in range(1, num_subset_test_data+1):
            with open(test_subset_data_dir + 'int_part{}.json'.format(index), 'rb') as file:
                subset_data = pickle.load(file)
                yield index, subset_data

    def test_model(self):
        self.test_log('test step is beginning..')
        start_time = time.time()
        total_tt_accuracy = 0.0
        total_nt_accuracy = 0.0
        subdata_generator = self.subset_generator()
        for index, subset_test_data in subdata_generator:  # 遍历每个sub test dataset
            sub_tt_correct = 0.0
            sub_nt_correct = 0.0

            for token_sequence in subset_test_data:  # 遍历该subset中每个nt token sequence
                n_expection, t_expection = expection
                prefix, expection, suffix = self.create_hole(token_sequence)  # 随机在sequence中创建一个hole
                n_prediction, t_prediction = self.query(prefix, suffix)

                if self.token_equal(n_prediction, n_expection):
                    sub_nt_correct += 1
                if self.token_equal(t_prediction, t_expection):
                    sub_tt_correct += 1

            sub_nt_accuracy = sub_nt_correct / len(subset_test_data)
            sub_tt_accuracy = sub_tt_correct / len(subset_test_data)
            total_nt_accuracy += sub_nt_accuracy
            total_tt_accuracy += sub_tt_accuracy
            end_time = time.time()
            log_info = '{}th subset of test data  '.format(index) + \
                'average time cost per case: {:.2f}  '.format((end_time-start_time)/seq_per_subset) + \
                'accuracy of non-terminal token: {:.2f}  '.format(sub_nt_accuracy*100) + \
                'accuracy of terminal token: {:.2f}  '.format(sub_tt_accuracy*100)
            self.test_log(log_info)

        total_nt_accuracy /= num_subset_test_data
        total_tt_accuracy /= num_subset_test_data
        log_info = 'test finished  ' + \
            'accuracy of non-terminal token: {:.2f}  '.format(total_nt_accuracy) + \
            'accuracy of terminal token: {:.2f}  '.format(total_tt_accuracy)
        self.test_log(log_info)
        return total_nt_accuracy, total_tt_accuracy

    def token_equal(self, prediction, expection):
        if type(prediction) is not int or type(expection) is not int:
            print('ERROR: the format of token is not int')
            return False
        elif(prediction == expection):
            return True
        return False

    def create_hole(self, nt_token_seq, hole_size=1):
        hole_start_index = random.randint(
            len(nt_token_seq) // 2,
            len(nt_token_seq) - hole_size)
        hole_end_index = hole_start_index + hole_size
        prefix = nt_token_seq[0:hole_start_index]
        expection = nt_token_seq[hole_start_index:hole_end_index]
        suffix = nt_token_seq[hole_end_index:-1]
        return prefix, expection, suffix

    def test_log(self, log_info):
        self.log_file.write(log_info)
        self.log_file.write('\n')
        print(log_info)


if __name__ == '__main__':
    # test step
    terminalToken2int, terminalInt2token, nonTerminalToken2int, nonTerminalInt2token = utils.load_dict_parameter()
    num_ntoken = len(nonTerminalInt2token)
    num_ttoken = len(terminalInt2token)
    test_model = CodeCompletion(num_ntoken, num_ttoken)
    nt_accuracy, tt_accuracy = test_model.test_model()
