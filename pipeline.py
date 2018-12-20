from basic_lstm import RnnModel
from code_completion import CodeCompletion
import utils

def data_processing():

    print('data processing begin...')
    utils.dataset_split(is_training=True)
    utils.train_nt_seq_to_int(train_or_valid='TRAIN')
    utils.dataset_split(is_training=False)
    utils.test_nt_seq_to_int()
    utils.train_nt_seq_to_int(train_or_valid='VALID')
    print('data processing finished...')


def model_training():

    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = utils.load_dict_parameter()
    n_ntoken = len(nt_int_to_token)
    n_ttoken = len(tt_int_to_token)
    model = RnnModel(n_ntoken, n_ttoken, saved_model=True)
    model.train()


def model_evaluation():

    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = utils.load_dict_parameter()
    num_ntoken = len(nt_token_to_int)
    num_ttoken = len(tt_token_to_int)
    test_model = CodeCompletion(num_ntoken, num_ttoken)
    nt_accuracy, tt_accuracy = test_model.test_model()


if __name__ == '__main__':
    steps = ['data processing', 'model training', 'model evaluation']
    data_processing()
    model_training()
    model_evaluation()