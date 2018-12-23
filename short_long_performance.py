import tensorflow as tf
import numpy as np
import time
import pickle
from collections import Counter

import utils
from basic_lstm import RnnModel
from setting import Setting


test_setting = Setting()
test_subset_data_dir = test_setting.sub_int_test_dir
model_save_dir = test_setting.lstm_model_save_dir
test_log_dir = test_setting.lstm_test_log_dir


num_subset_test_data = test_setting.num_sub_test_data
seq_per_subset = 5000
show_every_n = test_setting.test_show
num_terminal = test_setting.num_terminal
test_time_step = 50




class ShortLongTest(object):
    """Record the performance with different length"""
    def __init__(self,
                 num_ntoken,
                 num_ttoken,):
        self.model = RnnModel(num_ntoken, num_ttoken, is_training=False)
        self.sess = tf.Session()
        self.last_chackpoints = tf.train.latest_checkpoint(
            checkpoint_dir=model_save_dir)

        saver = tf.train.Saver()
        saver.restore(self.sess, self.last_chackpoints)

    def subset_generator(self):
        for index in range(1, num_subset_test_data +1):
            with open(test_subset_data_dir + 'int_part{}.json'.format(index), 'rb') as file:
                subset_data = pickle.load(file)
                yield index, subset_data

    def find_long_seq(self, length_define=5000, saved_info=False):
        """pick up longer test cases in test dataset"""
        long_case = []
        subdata_generator = self.subset_generator()
        length_counter = Counter()
        for index, subset_test_data in subdata_generator:
            for token_seq in subset_test_data:
                length_counter[len(token_seq)] += 1
                if len(token_seq) >= length_define:
                    token_seq = token_seq[:length_define+1]
                    long_case.append(token_seq)
        sorted_counter = sorted(length_counter.items(), key=lambda x: x[0] ,reverse=True)
        if saved_info:
            pickle.dump(sorted_counter, open('longth_count_info.p', 'wb'))
        return long_case

    def short_long_performance(self):
        length_define = 5000
        long_case = self.find_long_seq(length_define)
        num_test_case = len(long_case)
        long_case = np.array(long_case)  # long_case.shape = (258, 5001, 2)
        test_epoch = 1
        length_nt_correct = np.zeros(length_define, dtype=np.float32)
        length_tt_correct = np.zeros(length_define, dtype=np.float32)

        for i in range(test_epoch):
            lstm_state = self.sess.run(self.model.init_state)
            for test_case in long_case:
                nt_token_input = test_case[:length_define, 0].reshape([1, length_define])
                tt_token_input = test_case[:length_define, 1].reshape([1, length_define])
                nt_token_target = test_case[1:length_define +1, 0]
                tt_token_target = test_case[1:length_define +1, 1]
                feed = {self.model.lstm_state: lstm_state,
                        self.model.n_input :nt_token_input,
                        self.model.t_input :tt_token_input,
                        self.model.keep_prob :1.0}
                lstm_state, n_prediction, t_prediction = self.sess.run(
                    [self.model.final_state, self.model.n_output, self.model.t_output], feed)
                n_prediction = np.argmax(n_prediction, axis=1)
                t_prediction = np.argmax(t_prediction, axis=1)
                nt_result = np.equal(n_prediction, nt_token_target).astype(np.float32).reshape(length_define)
                tt_result = np.equal(t_prediction, tt_token_target).astype(np.float32).reshape(length_define)
                length_nt_correct += nt_result
                length_tt_correct += tt_result

        nt_accuracy = length_nt_correct / (test_epoch * num_test_case)
        tt_accuracy = length_tt_correct / (test_epoch * num_test_case)
        file = open('short_long_performance.p', 'wb')
        pickle.dump([nt_accuracy, tt_accuracy], file)

        return nt_accuracy, tt_accuracy

    def plot_performance(self):
        import matplotlib.pyplot as plt
        file = open('short_long_performance.p', 'rb')
        nt_accuracy, tt_accuracy = pickle.load(file)
        plt.figure(figsize=(40, 12))
        plt.plot(nt_accuracy, label='non-terminal')
        plt.plot(tt_accuracy, label='terminal')
        plt.xlabel('time step')
        plt.ylabel('accuracy')
        plt.title('performance with length')
        plt.grid()
        plt.show()





if __name__ == '__main__':
    # test step
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = utils.load_dict_parameter()
    num_ntoken = len(nt_token_to_int)
    num_ttoken = len(tt_token_to_int)
    model = ShortLongTest(num_ntoken, num_ttoken)
    model.short_long_performance()