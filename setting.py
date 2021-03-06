import time

"""保存整个项目的一些设定参数"""

class Setting(object):
    '''保存各种路径，以及其它的一些参数'''

    def __init__(self):

        # 数据相关的路径变量
        self.data_path = 'js_dataset/'  # 最原始数据集路径
        self.origin_test_data_dir = self.data_path + 'js_programs_eval.json'  # 原始AST测试数据路径
        self.origin_train_data_dir = self.data_path + 'js_programs_training.json'  # 原始AST训练数据路径

        self.sub_train_data_dir = self.data_path + 'split_js_data/train_data/'  # 原始AST训练数据被分成成多个小数据集的保存路径
        self.sub_valid_data_dir = self.data_path + 'split_js_data/valid_data/'
        self.sub_test_data_dir = self.data_path + 'split_js_data/eval_data/'  # 原始AST测试数据被分成成多个小数据集的保存路径

        # int2token的映射字典保存路径
        self.data_parameter_dir = self.data_path + 'split_js_data/parameter.p'  # 生成的int-token映射字典保存路径

        # 按照映射字典转换成int值的seq表示
        self.sub_int_train_dir = self.data_path + 'split_js_data/train_data/int_format/'
        self.sub_int_valid_dir = self.data_path + 'split_js_data/valid_data/int_format/'
        self.sub_int_test_dir = self.data_path + 'split_js_data/eval_data/int_format/'

        # 模型log相关路径
        curr_time = time.strftime('_%Y_%m_%d_%H_%M', time.localtime())  # 年月日时分
        self.current_time = curr_time
        self.lstm_model_save_dir = 'trained_model/lstm_model/'  # 训练好的模型的保存路径
        self.double_lstm_save_dir = 'trained_model/double_model/'
        self.lstm_tb_log_dir = 'log_info/tensorboard_log/lstm' + str(curr_time) + '/'  # 训练时tensorboard的log
        self.lstm_train_log_dir = 'log_info/training_log/lstm_train_log' + str(curr_time) + '.txt'  # 模型训练时的log
        self.lstm_completion_log_dir = 'log_info/test_log/lstm_completion_log' + str(curr_time) + '.txt'  # 模型completion时的log
        self.lstm_test_log_dir = 'log_info/test_log/lstm_test_log' + str(curr_time) + '.txt'  # 模型测试时的log
        self.lstm_valid_log_dir = 'log_info/valid_log/lstm_valid_log' + str(curr_time) + '.txt'

        self.node2vec_train_log_dir = '../log_info/training_log/node2vec_train_log' + str(curr_time) + '.txt'  # node2vec 模型训练时的log
        self.node2vec_tt_train_log_dir = '../log_info/training_log/node2vec_tt_train_log' + str(curr_time) + '.txt'  # node2vec 模型训练时的log



        # 暂时存放一些信息
        self.temp_info = 'temp_info/'

        # 数据集特性相关
        self.num_sub_train_data = 20  # change to 28 for more data to train
        self.num_sub_valid_data = 1
        self.num_sub_test_data = 9
        self.unknown_token = 'UNK'
        self.split_token = '=$$='
        self.num_non_terminal = 123
        self.num_terminal = 30000 + 1
        self.num_seq_per_subset = 5000

        # 学习模型相关
        self.time_steps = 50
        self.show_every_n = 1  # 为了快速debug
        self.valid_every_n = 2
        self.save_every_n = 1
        self.test_show = 1
        self.define_topk = 3

        # word2vec相关
        self.word2vec_embed_dim = 300
        self.word2vec_save_path = 'trained_model/token2vec.model'