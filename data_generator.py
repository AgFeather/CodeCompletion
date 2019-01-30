import numpy as np
import pickle
from setting import Setting



class DataGenerator():
    """a generator class for data generation"""
    def __init__(self, batch_size=50, time_steps=50):
        self.time_steps = time_steps
        self.batch_size = batch_size
        model_setting = Setting()
        self.num_subset_train_data = model_setting.num_sub_train_data
        self.num_subset_test_data = model_setting.num_sub_test_data
        self.sub_int_train_dir = model_setting.sub_int_train_dir
        self.sub_int_valid_dir = model_setting.sub_int_valid_dir
        self.sub_int_test_dir = model_setting.sub_int_test_dir
        #self.sub_int_test_dir = model_setting.sub_int_valid_dir

    def get_batch(self, data_seq):
        """Generator for training and valid phase,
        each time it will return (non-terminal x, non-terminal y, terminal x, terminal y)
        shape of both x and y is [batch_size, time_step]"""
        data_seq = np.array(data_seq)
        total_length = self.time_steps * self.batch_size
        n_batches = len(data_seq) // total_length
        data_seq = data_seq[:total_length * n_batches]
        nt_x = data_seq[:, 0]
        tt_x = data_seq[:, 1]
        nt_y = np.zeros_like(nt_x)
        tt_y = np.zeros_like(tt_x)
        nt_y[:-1], nt_y[-1] = nt_x[1:], nt_x[0]
        tt_y[:-1], tt_y[-1] = tt_x[1:], tt_x[0]

        nt_x = nt_x.reshape((self.batch_size, -1))
        tt_x = tt_x.reshape((self.batch_size, -1))
        nt_y = nt_y.reshape((self.batch_size, -1))
        tt_y = tt_y.reshape((self.batch_size, -1))
        data_seq = data_seq.reshape((self.batch_size, -1))
        for n in range(0, data_seq.shape[1], self.time_steps):
            batch_nt_x = nt_x[:, n:n + self.time_steps]
            batch_tt_x = tt_x[:, n:n + self.time_steps]
            batch_nt_y = nt_y[:, n:n + self.time_steps]
            batch_tt_y = tt_y[:, n:n + self.time_steps]
            if batch_nt_x.shape[1] == 0:
                break
            yield batch_nt_x, batch_nt_y, batch_tt_x, batch_tt_y

    def get_valid_batch(self, nt_seq):
        """Generator for valid test during test phase"""
        nt_seq = np.array(nt_seq)
        nt_x = nt_seq[:-1, 0]
        tt_x = nt_seq[:-1, 1]
        nt_y = nt_seq[1:, 0]
        tt_y = nt_seq[1:, 1]
        for n in range(0, len(nt_seq), self.time_steps):
            batch_nt_x = nt_x[n:n + self.time_steps].reshape([1, -1])
            batch_tt_x = tt_x[n:n + self.time_steps].reshape([1, -1])
            batch_nt_y = nt_y[n:n + self.time_steps].reshape([1, -1])
            batch_tt_y = tt_y[n:n + self.time_steps].reshape([1, -1])
            yield batch_nt_x, batch_nt_y, batch_tt_x, batch_tt_y

    def get_test_batch(self, prefix):
        prefix = np.array(prefix)
        for index in range(0, len(prefix), self.time_steps):
            nt_token = prefix[index: index+self.time_steps, 0].reshape([1, -1])
            tt_token = prefix[index: index+self.time_steps, 1].reshape([1, -1])
            yield nt_token, tt_token

    def get_train_subset_data(self):
        """yield sub training dataset"""
        for i in range(1, self.num_subset_train_data + 1):
            data_path = self.sub_int_train_dir + 'int_part{}.json'.format(i)
            with open(data_path, 'rb') as file:
                data = pickle.load(file)
                yield data

    def get_valid_subset_data(self):
        valid_dir = self.sub_int_valid_dir + 'int_part1.json'
        with open(valid_dir, 'rb') as f:
            valid_data = pickle.load(f)
        return valid_data

    def get_test_subset_data(self):
        for index in range(1, self.num_subset_test_data+1):
            with open(self.sub_int_test_dir + 'int_part{}.json'.format(index), 'rb') as file:
                subset_data = pickle.load(file)
                yield index, subset_data