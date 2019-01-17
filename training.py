import tensorflow as tf
import time
import os

from lstm_model import RnnModel
from setting import Setting
from data_generator import DataGenerator

base_setting = Setting()

model_save_dir = base_setting.lstm_model_save_dir
tensorboard_log_dir = base_setting.lstm_tb_log_dir
training_log_dir = base_setting.lstm_train_log_dir
valid_log_dir = base_setting.lstm_valid_log_dir

show_every_n = base_setting.show_every_n
save_every_n = base_setting.save_every_n
valid_every_n = base_setting.valid_every_n


class TrainModel(object):
    """Train rnn model"""
    def __init__(self,
                 num_ntoken, num_ttoken, kernel='LSTM',
                 batch_size=50,
                 num_epochs=10,
                 time_steps=50,):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = RnnModel(num_ntoken, num_ttoken, is_training=True, kernel=kernel)

    def train(self):
        model_info = 'basic lstm model  ' + \
                     'time_step:{},  batch_size:{} is training...'.format(
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
            loss_per_epoch = 0.0
            n_accu_per_epoch = 0.0
            t_accu_per_epoch = 0.0
            subset_generator = self.generator.get_train_subset_data()

            for data in subset_generator:
                batch_generator = self.generator.get_batch(data_seq=data)
                lstm_state = session.run(self.model.init_state)

                for b_nt_x, b_nt_y, b_t_x, b_t_y in batch_generator:
                    batch_step += 1
                    global_step += 1
                    batch_start_time = time.time()

                    feed = {self.model.t_input: b_t_x,
                            self.model.n_input: b_nt_x,
                            self.model.n_target: b_nt_y,
                            self.model.t_target: b_t_y,
                            self.model.keep_prob: 0.5,
                            self.model.lstm_state: lstm_state,
                            self.model.decay_epoch: (epoch-1)}
                    loss, n_loss, t_loss, \
                    n_accu, t_accu, topk_n_accu, topk_t_accu, \
                    _, learning_rate, summary_str = \
                        session.run([
                            self.model.loss,
                            self.model.n_loss,
                            self.model.t_loss,
                            self.model.n_accu,
                            self.model.t_accu,
                            self.model.n_topk_accu,
                            self.model.t_topk_accu,
                            self.model.optimizer,
                            self.model.decay_learning_rate,
                            self.model.merged_op], feed_dict=feed)

                    tb_writer.add_summary(summary_str, global_step)
                    tb_writer.flush()

                    loss_per_epoch += loss
                    n_accu_per_epoch += n_accu
                    t_accu_per_epoch += t_accu
                    batch_end_time = time.time()

                    if global_step % show_every_n == 0:
                        log_info = 'epoch:{}/{}  '.format(epoch, self.num_epochs) + \
                                   'global_step:{}  '.format(global_step) + \
                                   'loss:{:.2f}(n_loss:{:.2f} + t_loss:{:.2f})  '.format(loss, n_loss, t_loss) + \
                                   'nt_accu:{:.2f}%  '.format(n_accu * 100) + \
                                   'tt_accu:{:.2f}%  '.format(t_accu * 100) + \
                                   'top3_nt_accu:{:.2f}%  '.format(topk_n_accu * 100) + \
                                   'top3_tt_accu:{:.2f}%  '.format(topk_t_accu * 100) + \
                                   'learning_rate:{:.4f}  '.format(learning_rate) + \
                                   'time cost per batch:{:.2f}/s'.format(batch_end_time - batch_start_time)
                        self.print_and_log(log_info)

                    if global_step % valid_every_n == 0:
                        self.valid(session, epoch, global_step)

                    # if global_step % save_every_n == 0:
                    #     saver.save(session, model_save_dir + 'e{}_b{}.ckpt'.format(epoch, batch_step))
                    #     print('model saved: epoch:{} global_step:{}'.format(epoch, global_step))

            valid_n_accu, valid_t_accu = self.valid(session, epoch, global_step)
            epoch_end_time = time.time()
            epoch_cost_time = epoch_end_time - epoch_start_time

            epoch_log = 'EPOCH:{}/{}  '.format(epoch, self.num_epochs) + \
                        'epoch average loss:{:.2f}  '.format(loss_per_epoch / batch_step) + \
                        'epoch average nt_accu:{:.2f}%  '.format(100 * n_accu_per_epoch / batch_step) + \
                        'epoch average tt_accu:{:.2f}%  '.format(100 * t_accu_per_epoch / batch_step) + \
                        'epoch valid nt_accu:{:.2f}%  '.format(valid_n_accu) + \
                        'epoch valid tt_accu:{:.2f}%  '.format(valid_t_accu) + \
                        'time cost this epoch:{:.2f}/s  '.format(epoch_cost_time) + '\n'
            saver.save(session, model_save_dir + 'EPOCH{}.ckpt'.format(epoch, batch_step))
            self.print_and_log(epoch_log)
            self.print_and_log('EPOCH{} model saved'.format(epoch))


        saver.save(session, model_save_dir + 'lastest_model.ckpt')
        self.print_and_log('model training finished...')
        session.close()

    def valid(self, session, epoch, global_step):
        """valid model when it is trained"""
        valid_data = self.generator.get_valid_subset_data()
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


if __name__ == '__main__':
    num_terminal = base_setting.num_terminal
    num_non_terminal = base_setting.num_non_terminal
    model = TrainModel(num_non_terminal, num_terminal, kernel='LSTM')
    model.train()
