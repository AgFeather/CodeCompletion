import tensorflow as tf
import random
import time

from lstm_model import RnnModel
from data_generator import DataGenerator
from setting import Setting
import utils

test_setting = Setting()
model_save_dir = test_setting.lstm_model_save_dir
show_every_n = test_setting.test_show


class CodeCompletion(object):
    """test code completion performance"""
    def __init__(self,
                 num_ntoken,
                 num_ttoken,):
        self.model = RnnModel(num_ntoken, num_ttoken, is_training=False)
        self.sess = tf.Session()
        checkpoints_path = tf.train.latest_checkpoint(model_save_dir)
        self.define_topk = 3
        self.generator = DataGenerator()
        saver = tf.train.Saver()
        _, self.tt_int_to_token, __, self.nt_int_to_token = utils.load_dict_parameter()
        saver.restore(self.sess, checkpoints_path)

    def eval(self, prefix):
        """ evaluate one source code file, return the top k prediction and it's possibilities,
        Input: all tokens before the hole token(prefix) and all tokens after the hole token,
        ML model will predict the most probable token in the hole
        """
        new_state = self.sess.run(self.model.init_state)
        test_batch = self.generator.get_test_batch(prefix)
        n_topk_pred, n_topk_poss, t_topk_pred, t_topk_poss = None, None, None, None
        for nt_token, tt_token in test_batch:
            feed = {self.model.n_input: nt_token,
                    self.model.t_input: tt_token,
                    self.model.keep_prob: 1.,
                    self.model.lstm_state: new_state}
            n_topk_pred, n_topk_poss, t_topk_pred, t_topk_poss, new_state = self.sess.run([
                self.model.n_topk_pred, self.model.n_topk_poss, self.model.t_topk_pred,
                self.model.t_topk_poss, self.model.final_state], feed_dict=feed)

        assert n_topk_pred is not None and n_topk_poss is not None and \
            t_topk_pred is not None and t_topk_poss is not None
        n_topk_pred = n_topk_pred[-1, :]
        n_topk_poss = n_topk_poss[-1, :]
        t_topk_pred = t_topk_pred[-1, :]
        t_topk_poss = t_topk_poss[-1, :]
        topk_token_pairs = [self.int_to_token(n_int ,t_int)
                            for n_int, t_int in zip(n_topk_pred, t_topk_pred)]
        topk_pairs_poss = [(n_poss, t_poss)
                           for n_poss, t_poss in zip(n_topk_poss, t_topk_poss)]

        print('the token you may want to write is:')
        print(topk_token_pairs)
        print('with possibilities:')
        print(topk_pairs_poss)
        return topk_token_pairs, topk_pairs_poss

    def eval_without_define_k(self, prefix, topk):
        """ evaluate one source code file, return the top k prediction and it's possibilities,
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
        n_topk_pred = (-n_prediction).argsort()[:topk]
        t_topk_pred = (-t_prediction).argsort()[:topk]
        n_topk_poss = n_prediction[n_topk_pred]
        t_topk_poss = t_prediction[t_topk_pred]

        topk_token_pairs = [self.int_to_token(n_int, t_int)
                            for n_int, t_int in zip(n_topk_pred, t_topk_pred)]
        topk_pairs_poss = [(n_poss, t_poss)
                           for n_poss, t_poss in zip(n_topk_poss, t_topk_poss)]
        print('the token you may want to write is:')
        print(topk_token_pairs)
        print('with possibilities:')
        print(topk_pairs_poss)
        return topk_token_pairs, topk_pairs_poss

    def query(self, token_sequence, topk=3):
        prefix, expectation, suffix = self.create_hole(token_sequence)  # 随机在sequence中创建一个hole
        n_expectation, t_expectation = expectation[0]
        if self.define_topk == topk:
            self.eval(prefix)
        else:
            self.eval_without_define_k(prefix, topk)


    def test_model(self):
        """Test model with the whole test dataset, it will call self.query() for each test case"""
        start_time = time.time()
        test_times = 2000
        test_step = 0
        self.generator = DataGenerator()
        sub_data_generator = self.generator.get_test_subset_data()

        for index, subset_test_data in sub_data_generator:  # 遍历每个sub test dataset
            one_test_start_time = time.time()
            for token_sequence in subset_test_data:  # 遍历该subset中每个nt token sequence
                test_step += 1
                # 对一个ast sequence进行test
                self.query(token_sequence)
                if test_step % show_every_n == 0:
                    one_test_end_time = time.time()
                    duration = (one_test_end_time - one_test_start_time) / show_every_n
                    one_test_start_time = one_test_end_time

                    log_info = 'test step:{}  '.format(test_step) + \
                               'average time cost:{:.2f}s  '.format(duration)
                    print(log_info)

                if test_step >= test_times:
                    break

            end_time = time.time()
            log_info = '{}th subset of test data  '.format(index) + \
                'there are {} nt_sequence to test  '.format(test_step) + \
                'total time cost of this subset: {:.2f}s  '.format(end_time - start_time) + \
                'average time cost per case: {:.2f}s  '.format((end_time - start_time) / test_step)
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

    def create_hole(self, nt_token_seq, hole_size=1):
        hole_start_index = random.randint(
            len(nt_token_seq) // 2, len(nt_token_seq) - hole_size)
        hole_end_index = hole_start_index + hole_size
        prefix = nt_token_seq[0:hole_start_index]
        expectation = nt_token_seq[hole_start_index:hole_end_index]
        suffix = nt_token_seq[hole_end_index:-1]
        return prefix, expectation, suffix




if __name__ == '__main__':
    num_ntoken = test_setting.num_non_terminal
    num_ttoken = test_setting.num_terminal
    test_model = CodeCompletion(num_ntoken, num_ttoken)
    test_model.test_model()