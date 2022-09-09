'''
声学模型调用类
'''

import os
import time
import random
import numpy as np

from asrt.utils.ops import get_edit_distance, read_wav_data
from asrt.utils.config import load_config_file, DEFAULT_CONFIG_FILENAME, load_pinyin_dict
from asrt.utils.thread import threadsafe_generator


class ModelSpeech:
    '''
    语音模型类
    '''

    def __init__(self, speech_model, speech_features, max_label_length=64):
        '''
        :param speech_model: 声学模型类实例对象 basemodel
        :param speech_features: 声学特征提取实例对象
        :param max_label_length:
        '''
        self.data_loader = None
        self.speech_model = speech_model
        self.trained_model, self.base_model = speech_model.get_model()
        self.speech_features = speech_features
        self.max_label_length = max_label_length

    @threadsafe_generator
    def _data_generator(self, batch_size, data_loader):
        '''
        数据生成器函数
        :param batch_size:
        :param data_loader:
        :return:
        '''

        labels = np.zeros((batch_size, 1), dtype=np.float)
        data_count = data_loader.get_data_count()
        index = 0

        while True:
            x = np.zeros((batch_size,) + self.speech_model.input_shape, dtype=np.float)
            y = np.zeros((batch_size, self.max_label_length), dtype=np.int16)
            input_length = []
            label_length = []

            for i in range(batch_size):
                wavdata, sample_rate, data_labels = data_loader.get_data(index)
                data_input = self.speech_features.run(wavdata, sample_rate)
                data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                pool_size = self.speech_model.input_shape[0] // self.speech_model.output_shape[0]
                inlen = min(data_input.shape[0] // pool_size + data_input.shape[0] % pool_size,
                            self.speech_model.output_shape[0])
                input_length.append(inlen)

                x[i, 0:len(data_input)] = data_input
                y[i, 0:len(data_labels)] = data_labels
                label_length.append(len(data_labels))

                index = (index + 1) % data_count

            label_length = np.array(label_length)
            input_length = np.array([input_length]).T
            yield [x, y, input_length, label_length], labels

    def train_model(self, optimizer, data_loader, epochs=1, save_step=1, batch_size=16, last_epoch=0, call_back=None):
        '''
        训练模型
        :param optimizer: tensorflow.keras.optimizers 优化器对象
        :param data_loader: 数据加载器 SpeechData 实例对象
        :param epochs: 迭代轮次
        :param save_step: 没多少轮次保存一次
        :param batch_size:
        :param last_epoch: 上次轮次的编号，可用于断点继续训练
        :param call_back:
        :return:
        '''

        save_filename = os.path.join('save_model', self.speech_model.get_model_name(),
                                     self.speech_model.get_model_name())

        self.trained_model.compile(loss=self.speech_model.get_loss_function(), optimizer=optimizer)
        print('[asrt] compiles model successfully')

        yielddata = self._data_generator(batch_size, data_loader)
        data_count = data_loader.get_data_count()
        num_iterate = data_count // batch_size
        iter_start = last_epoch
        iter_end = last_epoch + epochs

        for epoch in range(iter_start, iter_end):
            epoch += 1
            print('[asrt training] train epoch {}/{}'.format(epoch, iter_end))
            data_loader.shuffle()
            self.trained_model.fit_generator(yielddata, num_iterate, callbacks=call_back)

            if epoch % save_step == 0:
                if not os.path.exists('save_model'):
                    os.makedirs('save_model')
                if not os.path.exists(os.path.join('save_model', self.speech_model.get_model_name())):
                    os.makedirs(os.path.join('save_model', self.speech_model.get_model_name()))

                self.save_model(save_filename + '_epoch' + str(epoch))
        print('[asrt info] model training complete')

    def load_model(self, filename):
        self.speech_model.load_weights(filename)

    def save_model(self, filename):
        pass

