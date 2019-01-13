import os
from urllib.request import urlretrieve
from tqdm import tqdm

def mkdir(path):
    if os.path.exists(path):
        print(path, 'already exists')
    else:
        os.mkdir(path)


def js_dataset_download():
    data_url = "http://files.srl.inf.ethz.ch/data/js_dataset.tar.gz"
    data_save_path = "js_dataset/data.tar.gz"
    if os.path.exists(data_save_path):
        print('dataset existing...')
    else:
        with DLProgress(unit='B', unit_scale=True, mininters=1, desc='Download {}'.format(data_save_path)) as pbar:
            urlretrieve(data_url, data_save_path, pbar.hook())


class DLProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def dataset_unzip():
    pass

if __name__ == '__main__':
    # path_setting = setting.Setting()
    log_path = 'log_info/'
    mkdir(log_path)
    mkdir(log_path + 'tensorboard_log/')
    mkdir(log_path + 'test_log/')
    mkdir(log_path + 'training_log/')
    mkdir(log_path + 'valid_log/')
    mkdir('temp_info/')
    mkdir('trained_model')

# 创建项目需要的各个路径，加上数据集下载，解压（improve）