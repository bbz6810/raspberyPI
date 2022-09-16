'''
加载语音数据生成器
'''

import os
import random
import numpy as np
from asrt.utils.config import load_config_file, DEFAULT_CONFIG_FILENAME, load_pinyin_dict
from asrt.utils.ops import read_wav_data


class DataLoader:
    '''
    数据加载器
    参数：
        config：配置信息字典
        dataset_type: 要加载的数据集类型（train，test，dev）三种
    '''

    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.data_list = list()
        self.wav_dict = dict()
        self.label_dict = dict()
        self.pinyin_list = list()
        self.pinyin_dict = dict()
        self._load_data()

    def _load_data(self):
        config = load_config_file(DEFAULT_CONFIG_FILENAME)
        self.pinyin_list, self.pinyin_dict = load_pinyin_dict(config['dict_filename'])

        for index in range(len(config['dataset'][self.dataset_type])):
            filename_datalist = config['dataset'][self.dataset_type][index]['data_list']
            filename_datapath = config['dataset'][self.dataset_type][index]['data_path']
            with open(filename_datalist, 'r', encoding='utf-8') as fr:
                lines = fr.read().split('\n')
                for line in lines:
                    if len(line) == 0:
                        continue
                    tokens = line.split(' ')
                    self.data_list.append(tokens[0])
                    self.wav_dict[tokens[0]] = os.path.join(filename_datapath, tokens[1])

            filename_labellist = config['dataset'][self.dataset_type][index]['label_list']
            with open(filename_labellist, 'r', encoding='utf-8') as fr:
                lines = fr.read().split('\n')
                for line in lines:
                    if len(line) == 0:
                        continue
                    tokens = line.split(' ')
                    self.label_dict[tokens[0]] = tokens[1:]

    def get_data_count(self):
        return len(self.data_list)

    def get_data(self, index):
        mark = self.data_list[index]

        wav_signal, sample_rate, _, _ = read_wav_data(self.wav_dict[mark])
        labels = list()
        for item in self.label_dict[mark]:
            if len(item) == 0:
                continue
            labels.append(self.pinyin_dict[item])
        data_label = np.array(labels)
        return wav_signal, sample_rate, data_label

    def shuffle(self):
        random.shuffle(self.data_list)
