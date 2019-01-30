import pickle

from setting import Setting

base_setting = Setting()

data_parameter_dir = base_setting.data_parameter_dir




def pickle_save(path, data):
    """使用pickle将给定数据保存到给定路径中"""
    file = open(path, 'wb')
    pickle.dump(data, file)
    print(path + ' has been saved...')


def pickle_load(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    return data


def load_dict_parameter(is_lower=True):
    # 加载terminal和nonterminal对应的映射字典
    if is_lower:
        path = '../' + data_parameter_dir
    else:
        path = data_parameter_dir
    file = open(path, 'rb')
    tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token = pickle.load(file)
    return tt_token_to_int, tt_int_to_token, nt_token_to_int, nt_int_to_token