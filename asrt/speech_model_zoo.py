'''
声学模型的定义
'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization, Lambda, Activation, Conv2D, \
    MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np

from asrt.utils.ops import ctc_decode_delete_tail_blank


class BaseModel:
    '''
    定义声学模型基类
    '''

    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def get_model(self):
        return self.model, self.model_base

    def get_train_model(self):
        return self.model

    def get_eval_model(self):
        return self.model_base

    def get_model_name(self):
        return self._model_name

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def save_weights(self, filename):
        self.model.save_weights(filename + '.model.h5')
        self.model_base.save_weights(filename + '.model.base.h5')

    def get_loss_function(self):
        raise Exception('method not implemented')

    def forward(self, x):
        raise Exception('method not implemented')


def ctc_lambda_func(args):
    y_pred, labels, input_len, label_len = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_len, label_len)


class SpeechModel251BN(BaseModel):
    '''
    定义CNN+ctc模型，使用函数式编程
    输入层：200维的特征值序列，一条语音数据的最大长度设为1600
    隐藏层：卷积池化层，卷积核3*3，池化窗口为2
    隐藏层：全联结网络
    输出层：全联结层，使用softmax作为激活函数
    ctc层：使用ctc的loss函数，实现连接性时序多输出

    参数：
        input_shape: tuple (1600, 200, 1)
        output_shape: tuple (200, 1428)
        1428：是根据全部拼音的个数定义的，可改变
    '''

    def __init__(self, input_shape=(1600, 200, 1), output_size=1428):
        super(SpeechModel251BN, self).__init__()
        self.input_shape = input_shape
        self._pool_size = 8
        self.output_shape = (input_shape[0] // self._pool_size, output_size)
        self._model_name = 'SpeechModel251BN'
        self.model, self.model_base = self._define_model(self.input_shape, self.output_shape[1])

    def _define_model(self, input_shape, output_size):
        label_max_string_length = 64

        input_data = Input(name='the_input', shape=input_shape)

        layer_h = Conv2D(32, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv0')(
            input_data)
        layer_h = BatchNormalization(epsilon=0.0002, name='BN0')(layer_h)
        layer_h = Activation('relu', name='Act0')

        layer_h = Conv2D(32, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv1')(
            layer_h)
        layer_h = BatchNormalization(epsilon=0.0002, name='BN1')(layer_h)
        layer_h = Activation('relu', name='Act1')(layer_h)

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding='valid', name='Pool1')(layer_h)

        layer_h = Conv2D(64, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv2')(
            layer_h)
        layer_h = BatchNormalization(epsilon=0.0002, name='BN2')(layer_h)
        layer_h = Activation('relu', name='Act2')(layer_h)

        layer_h = Conv2D(64, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv3')(
            layer_h)
        layer_h = BatchNormalization(epsilon=0.0002, name='BN3')(layer_h)
        layer_h = Activation('relu', name='Act3')(layer_h)

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding='valid', name='Pool3')(layer_h)

        layer_h = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv4')(
            layer_h)
        layer_h = BatchNormalization(epsilon=0.0002, name='BN4')(layer_h)
        layer_h = Activation('relu', name='Act4')(layer_h)

        layer_h = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv5')(
            layer_h)
        layer_h = BatchNormalization(epsilon=0.0002, name='BN5')(layer_h)
        layer_h = Activation('relu', name='Act5')(layer_h)

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding='valid', name='Pool5')(layer_h)

        layer_h = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv6')(
            layer_h)
        layer_h = BatchNormalization(epsilon=0.0002, name='BN6')(layer_h)
        layer_h = Activation('relu', name='Act6')(layer_h)

        layer_h = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv7')(
            layer_h)
        layer_h = BatchNormalization(epsilon=0.0002, name='BN7')(layer_h)
        layer_h = Activation('relu', name='Act7')(layer_h)

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding='valid', name='Pool7')(layer_h)

        layer_h = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv8')(
            layer_h)
        layer_h = BatchNormalization(epsilon=0.0002, name='BN8')(layer_h)
        layer_h = Activation('relu', name='Act8')(layer_h)

        layer_h = Conv2D(128, (3, 3), use_bias=True, padding='same', kernel_initializer='he_normal', name='Conv9')(
            layer_h)
        layer_h = BatchNormalization(epsilon=0.0002, name='BN9')(layer_h)
        layer_h = Activation('relu', name='Act9')(layer_h)

        layer_h = MaxPooling2D(pool_size=2, strides=None, padding='valid', name='Pool9')(layer_h)

        layer_h = Reshape((self.output_shape[0], input_shape[1] // self._pool_size * 128), name='Reshape0n')(layer_h)

        layer_h = Dense(128, activation='relu', use_bias=True, kernel_initializer='he_normal', name='Dense0')(layer_h)

        layer_h = Dense(output_size, use_bias=True, kernel_initializer='he_normal', name='Dense1')(layer_h)
        y_pred = Activation('softmax', name='Activation0')(layer_h)

        model_base = Model(inputs=input_data, outputs=y_pred)

        labels = Input(name='the_labels', shape=[label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        return model, model_base

    def get_loss_function(self):
        return {'ctc': lambda y_true, y_pred: y_pred}

    def forward(self, data_input):
        batch_size = 1
        in_len = np.zeros((batch_size), dtype=np.int32)
        in_len[0] = self.output_shape[0]
        x_in = np.zeros((batch_size,) + self.input_shape, dtype=np.float)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.model_base.predict(x=x_in)
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)

        r1 = r[0][0].numpy()

        speech_result = ctc_decode_delete_tail_blank(r1[0])
        return speech_result
