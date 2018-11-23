import time

class Setting(object):
    '''保存各种路径，以及其它的一些参数'''

    def __init__(self):

        # 数据相关的路径变量
        self.origin_test_data_dir = 'js_dataset/js_programs_eval.json'  # 原始AST测试数据路径
        self.origin_train_data_dir = 'js_dataset/js_programs_training.json'  # 原始AST训练数据路径

        self.sub_train_data_dir = 'split_js_data/train_data/'  # 原始AST训练数据被分成成多个小数据集的保存路径
        self.sub_test_data_dir = 'split_js_data/eval_data/'  # 原始AST测试数据被分成成多个小数据集的保存路径

        self.data_parameter_dir = 'split_js_data/parameter.p'  # 生成的int-token映射字典保存路径
        self.sub_int_train_dir = 'split_js_data/train_data/int_format/'
        self.sub_int_test_dir = 'split_js_data/eval_data/int_format/'

        self.lstm_model_save_dir = 'trained_model/lstm_model/'
        self.lstm_tb_log_dir = 'tensorboard_log/lstm/'
        curr_time = time.strftime('_%Y_%H_%d_%M', time.localtime())
        self.lstm_train_log_dir = 'training_log/lstm_log' + str(curr_time) + '.txt'

        self.num_sub_train_data = 20
        self.num_sub_test_data = 10
        self.unknown_token = 'UNK'
        self.num_terminal = 30000

        # 模型相关
        self.show_every_n = 1
        self.save_every_n = 1500