from training import TrainModel
from lstm_test import RnnModelTest
from code_completion import CodeCompletion
from short_long_performance import ShortLongTest
import utils
from setting import Setting

"""The pipeline and overflow of code completion system"""

base_setting = Setting()
num_ntoken = base_setting.num_non_terminal
num_ttoken = base_setting.num_terminal

def data_processing():

    print('data processing begin...')
    utils.dataset_split(is_training=True)
    utils.nt_seq_to_int(status='TRAIN')
    utils.dataset_split(is_training=False)
    utils.nt_seq_to_int(status='VALID')
    utils.nt_seq_to_int(status='VALID')
    print('data processing finished...')


def model_training():
    model = TrainModel(num_ntoken, num_ttoken)
    model.train()


def model_evaluation():
    test_model = RnnModelTest(num_ntoken, num_ttoken)
    test_model.test_model()

def code_completion():
    model = CodeCompletion(num_ntoken, num_ttoken)
    model.test_model()

def length_performance():
    model = ShortLongTest(num_ntoken, num_ttoken)
    model.short_long_performance()


if __name__ == '__main__':
    steps = ['data processing', 'model training', 'model evaluation', 'code completion']
    data_processing()
    model_training()
    model_evaluation()
    length_performance()