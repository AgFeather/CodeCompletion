import tensorflow as tf
import numpy as np
import time

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
define_topk =  test_setting.define_topk

class RnnModelTest(object):
    """test code completion performance"""
    def __init__(self,
                 num_ntoken,
                 num_ttoken,):
        self.model = RnnModel(num_ntoken, num_ttoken, is_training=False)
        self.log_file = open(test_log_dir, 'w')
        self.session = tf.Session()
        self.time_steps = 50
        checkpoints_path = tf.train.latest_checkpoint(model_save_dir)
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoints_path)
        self.test_log(checkpoints_path + ' is using...')

    def test_model(self):
        """Test model with the whole test dataset, it will call self.query() for each test case"""
        self.test_log('test phase is beginning...')
        start_time = time.time()

        tt_correct_count = 0.0
        nt_correct_count = 0.0
        topk_nt_correct_count = 0.0
        topk_tt_correct_count = 0.0
        test_times = 2000
        test_step = 0
        self.generator = DataGenerator()
        sub_data_generator = self.generator.get_test_subset_data()

        for index, subset_test_data in sub_data_generator:  # 遍历每个sub test dataset
            one_test_start_time = time.time()
            for token_sequence in subset_test_data:  # 遍历该subset中每个nt token sequence
                test_step += 1
                # 对一个ast sequence进行test
                if len(token_sequence) < self.time_steps:
                    continue
                nt_accu, tt_accu, topk_nt_accu, topk_tt_accu = self.valid_query(token_sequence)
                nt_correct_count += nt_accu
                tt_correct_count += tt_accu
                topk_nt_correct_count += topk_nt_accu
                topk_tt_correct_count += topk_tt_accu

                if test_step % show_every_n == 0:
                    one_test_end_time = time.time()
                    duration = (one_test_end_time - one_test_start_time) / show_every_n
                    one_test_start_time = one_test_end_time
                    show_nt_accu = nt_correct_count / test_step
                    show_tt_accu = tt_correct_count / test_step
                    show_topk_nt_accu = topk_nt_correct_count / test_step
                    show_topk_tt_accu = topk_tt_correct_count / test_step
                    log_info = 'test step:{}  '.format(test_step) + \
                               'nt_accuracy:{:.2f}%  '.format(show_nt_accu * 100) + \
                               'tt_accuracy:{:.2f}%  '.format(show_tt_accu * 100) + \
                               'nt_top{}_accuracy:{:.2f}%  '.format(define_topk, show_topk_nt_accu * 100) + \
                               'tt_top{}_accuracy:{:.2f}%  '.format(define_topk, show_topk_tt_accu * 100) + \
                               'average time cost:{:.2f}s  '.format(duration)
                    self.test_log(log_info)

                if test_step >= test_times:
                    break

            show_nt_accu = nt_correct_count / test_step
            show_tt_accu = tt_correct_count / test_step
            show_topk_nt_accu = topk_nt_correct_count / test_step
            show_topk_tt_accu = topk_tt_correct_count / test_step
            end_time = time.time()
            log_info = '{}th subset of test data  '.format(index) + \
                'there are {} nt_sequence to test  '.format(test_step) + \
                'accuracy of non-terminal token: {:.2f}%  '.format(show_nt_accu*100) + \
                'accuracy of terminal token: {:.2f}%  '.format(show_tt_accu*100) + \
                'top{} accuracy of non-terminal:{:.2f}%  '.format(define_topk, show_topk_nt_accu * 100) + \
                'top{} accuracy of terminal:{:.2f}%  '.format(define_topk, show_topk_tt_accu * 100) + \
                'total time cost of this subset: {:.2f}s  '.format(end_time - start_time) + \
                'average time cost per case: {:.2f}s  '.format((end_time - start_time) / seq_per_subset)
            self.test_log(log_info)


    def valid_query(self, nt_seq):
        """如同valid方法一样，不再是对每个ast随机创建一个hole，而是对每一个token都进行测试"""
        batch_step = 0
        test_nt_topk_accu = 0.0
        test_tt_topk_accu = 0.0
        test_nt_top1_accu = 0.0
        test_tt_top1_accu = 0.0

        lstm_state = self.session.run(self.model.init_state)
        batch_generator = self.generator.get_valid_batch(nt_seq)
        for b_nt_x, b_nt_y, b_tt_x, b_tt_y in batch_generator:
            batch_step += 1

            feed = {self.model.n_input: b_nt_x,
                    self.model.t_input: b_tt_x,
                    self.model.keep_prob: 1.,
                    self.model.n_target: b_nt_y,
                    self.model.t_target: b_tt_y,
                    self.model.lstm_state: lstm_state}
            n_topk_accu, t_topk_accu, n_accu, t_accu, lstm_state = self.session.run([
                self.model.n_top_k_accu, self.model.t_top_k_accu,
                self.model.n_accu, self.model.t_accu, self.model.final_state], feed_dict=feed)

            # if batch_step <= 1:
            #     continue
        # test_nt_top1_accu /= (batch_step-1)
        # test_tt_top1_accu /= (batch_step-1)
        # test_nt_topk_accu /= (batch_step-1)
        # test_tt_topk_accu /= (batch_step-1)
            test_nt_top1_accu += n_accu
            test_tt_top1_accu += t_accu
            test_nt_topk_accu += n_topk_accu
            test_tt_topk_accu += t_topk_accu

        test_nt_top1_accu /= batch_step
        test_tt_top1_accu /= batch_step
        test_nt_topk_accu /= batch_step
        test_tt_topk_accu /= batch_step

        return test_nt_top1_accu, test_tt_top1_accu, test_nt_topk_accu, test_tt_topk_accu

    def test_log(self, log_info):
        self.log_file.write(log_info)
        self.log_file.write('\n')
        print(log_info)

    def top_one_equal(self, prediction, expectation):
        if prediction[0] == expectation:
            return True
        return False

    def topk_equal(self, prediction, expectation):
        if expectation in prediction:
            return True
        return False




if __name__ == '__main__':
    num_ntoken = test_setting.num_non_terminal
    num_ttoken = test_setting.num_terminal
    test_model = RnnModelTest(num_ntoken, num_ttoken)
    test_model.test_model()