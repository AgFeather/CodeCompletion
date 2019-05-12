import tensorflow as tf
import time
import os
from nn_model.lstm_model import RnnModel
from nn_model.lstm_node2vec import LSTM_Node_Embedding
from setting import Setting
from data_generator import DataGenerator

"""训练LSTM模型，根据指定data_type选择是对原始数据集训练还是对rename数据集训练。
根据指定model_type选择使用LSTM模型还是LSTM_with_Node2Vec模型"""

#data_type = 'rename'
data_type = 'origin'
model_type = 'with_embedding'
# model_type = 'lstm'
base_setting = Setting()


model_save_dir = 'trained_model/' + data_type + '_' + model_type + '/'
tensorboard_log_dir = 'log_info/tensorboard_log/' + data_type + model_type + '/'
curr_time = time.strftime('_%Y_%m_%d_%H_%M', time.localtime())  # 年月日时分
training_log_dir = 'log_info/training_log/' + data_type + '_' + model_type + str(curr_time) + '.txt'
valid_log_dir = 'log_info/valid_log/lstm_valid_log/' + data_type + '_' + model_type + str(curr_time) + '.txt'

training_accu_log = 'log_info/accu_log/' + data_type + '_' + model_type + str(curr_time) + '.txt'

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
if not os.path.exists(tensorboard_log_dir):
    os.makedirs(tensorboard_log_dir)

show_every_n = base_setting.show_every_n
save_every_n = base_setting.save_every_n
valid_every_n = base_setting.valid_every_n


class TrainModel(object):
    def __init__(self,
                 num_ntoken, num_ttoken,
                 batch_size=50,
                 num_epochs=5,
                 time_steps=50,):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if model_type == 'with_embedding':
            self.model_name = "LSTM with Node2Vec"

            self.model = LSTM_Node_Embedding(num_ntoken, num_ttoken, is_training=True)
        else:
            self.model_name = "original LSTM"
            self.model = RnnModel(num_ntoken, num_ttoken, is_training=True)
        print("Using", self.model_name)


    def train(self):
        model_info = self.model_name + \
                     ' time_step:{},  batch_size:{} is training...'.format(
                         self.time_steps, self.batch_size)
        self.print_and_log(model_info)
        saver = tf.train.Saver(max_to_keep=self.num_epochs + 1)
        session = tf.Session()
        self.generator = DataGenerator(self.batch_size, self.time_steps)
        tb_writer = tf.summary.FileWriter(tensorboard_log_dir, session.graph)
        global_step = 0
        session.run(tf.global_variables_initializer())

        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            batch_step = 0
            subset_generator = self.generator.get_train_subset_data(train_type=data_type)

            for data in subset_generator:
                batch_generator = self.generator.get_batch(data_seq=data)
                lstm_state = session.run(self.model.init_state)

                for b_nt_x, b_nt_y, b_t_x, b_t_y in batch_generator:
                    batch_step += 1
                    global_step += 1

                    feed = {self.model.t_input: b_t_x,
                            self.model.n_input: b_nt_x,
                            self.model.n_target: b_nt_y,
                            self.model.t_target: b_t_y,
                            self.model.keep_prob: 0.5,
                            self.model.lstm_state: lstm_state,
                            self.model.decay_epoch: (epoch-1)}
                    loss, n_loss, t_loss, \
                    n_accu, t_accu, \
                    _, learning_rate, summary_str = \
                        session.run([
                            self.model.loss,
                            self.model.n_loss,
                            self.model.t_loss,
                            self.model.n_accu,
                            self.model.t_accu,
                            self.model.optimizer,
                            self.model.decay_learning_rate,
                            self.model.merged_op], feed_dict=feed)

                    self.save_accu_log(global_step, n_accu, t_accu)

                    tb_writer.add_summary(summary_str, global_step)
                    tb_writer.flush()

                    if global_step % show_every_n == 0:
                        log_info = 'epoch:{}/{}  '.format(epoch, self.num_epochs) + \
                                   'global_step:{}  '.format(global_step) + \
                                   'loss:{:.2f}(n_loss:{:.2f} + t_loss:{:.2f})  '.format(loss, n_loss, t_loss) + \
                                   'nt_accu:{:.2f}%  '.format(n_accu * 100) + \
                                   'tt_accu:{:.2f}%  '.format(t_accu * 100) + \
                                   'learning_rate:{:.4f}  '.format(learning_rate)
                        self.print_and_log(log_info)

                    # if global_step % valid_every_n == 0:
                    #     self.valid(session, epoch, global_step)

            # valid_n_accu, valid_t_accu = self.valid(session, epoch, global_step)
            epoch_end_time = time.time()
            epoch_cost_time = epoch_end_time - epoch_start_time

            epoch_log = 'EPOCH:{}/{}  '.format(epoch, self.num_epochs) + \
                        'time cost this epoch:{:.2f}/s  '.format(epoch_cost_time) + '\n'
            # 'epoch valid nt_accu:{:.2f}%  '.format(valid_n_accu) + \
            # 'epoch valid tt_accu:{:.2f}%  '.format(valid_t_accu) + \
            saver.save(session, model_save_dir + 'EPOCH{}.ckpt'.format(epoch))
            self.print_and_log(epoch_log)
            self.print_and_log('EPOCH{} model saved'.format(epoch))


        saver.save(session, model_save_dir + 'lastest_model.ckpt')
        self.print_and_log('model training finished...')
        session.close()

    def valid(self, session, epoch, global_step):
        """valid model when it is trained"""
        valid_data = self.generator.get_valid_subset_data(train_type=data_type)
        batch_generator = self.generator.get_batch(valid_data)
        valid_step = 0
        valid_n_accuracy = 0.0
        valid_t_accuracy = 0.0
        valid_times = 400
        valid_start_time = time.time()
        lstm_state = session.run(self.model.init_state)
        for b_nt_x, b_nt_y, b_t_x, b_t_y in batch_generator:
            valid_step += 1
            feed = {self.model.t_input: b_t_x,
                    self.model.n_input: b_nt_x,
                    self.model.n_target: b_nt_y,
                    self.model.t_target: b_t_y,
                    self.model.keep_prob: 1.0,
                    self.model.lstm_state: lstm_state}
            n_accuracy, t_accuracy, lstm_state = session.run(
                [self.model.n_accu, self.model.t_accu, self.model.final_state], feed)
            valid_n_accuracy += n_accuracy
            valid_t_accuracy += t_accuracy
            if valid_step >= valid_times:
                break

        valid_n_accuracy = (valid_n_accuracy*100) / valid_step
        valid_t_accuracy = (valid_t_accuracy*100) / valid_step
        valid_end_time = time.time()
        valid_log = "VALID epoch:{}/{}  ".format(epoch, self.num_epochs) + \
                    "global step:{}  ".format(global_step) + \
                    "valid_nt_accu:{:.2f}%  ".format(valid_n_accuracy) + \
                    "valid_tt_accu:{:.2f}%  ".format(valid_t_accuracy) + \
                    "valid time cost:{:.2f}s".format(valid_end_time - valid_start_time)
        if not os.path.exists(valid_log_dir):
            self.valid_file = open(valid_log_dir, 'w')
        valid_info = '{} {} {}\n'.format(global_step, valid_n_accuracy, valid_t_accuracy)
        self.valid_file.write(valid_info)
        self.print_and_log(valid_log)
        return valid_n_accuracy, valid_t_accuracy

    def print_and_log(self, info):
        if not os.path.exists(training_log_dir):
            self.log_file = open(training_log_dir, 'w')
        self.log_file.write(info)
        self.log_file.write('\n')
        print(info)

    def save_accu_log(self, global_step, n_accu, t_accu):
        if not os.path.exists(training_accu_log):
            self.accu_file = open(training_accu_log, 'w')
        accu_info = str(global_step) + ';' + str(n_accu) + ';' + str(t_accu) + '\n'
        self.accu_file.write(accu_info)
        if global_step % 1000 == 0:
            self.accu_file.flush()



if __name__ == '__main__':
    num_terminal = base_setting.num_terminal
    num_non_terminal = base_setting.num_non_terminal
    model = TrainModel(num_non_terminal, num_terminal)
    model.train()
