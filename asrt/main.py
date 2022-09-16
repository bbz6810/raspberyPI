'''
训练，测试代码
'''

import os
import sys
from tensorflow.keras.optimizers import Adam

from asrt.model_speech import ModelSpeech
from asrt.speech_model_zoo import SpeechModel251BN
from asrt.data_loader import DataLoader
from asrt.speech_features import SpecAugment, Spectrogram
from asrt.language_model import ModelLanguage

audio_length = 1600
audio_feature_length = 200
channels = 1

# 默认输出的拼音的表示大小是1428，即1427个拼音 + 1个空白块
output_size = 1428

op = sys.argv[1]
fn = sys.argv[2]

if op == 'train':
    sm251bn = SpeechModel251BN(
        input_shape=(audio_length, audio_feature_length, channels),
        output_size=output_size
    )
    feat = SpecAugment()
    train_data = DataLoader('train')
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)
    model_speech = ModelSpeech(sm251bn, feat, max_label_length=64)
    model_speech.train_model(optimizer=opt, data_loader=train_data, epochs=30, save_step=1, batch_size=16, last_epoch=0)
    model_speech.save_model('save_model/' + sm251bn.get_model_name())

elif op == 'dev':
    sm251bn = SpeechModel251BN(
        input_shape=(audio_length, audio_feature_length, channels),
        output_size=output_size
    )
    feat = Spectrogram()
    evalue_data = DataLoader('dev')
    model_speech = ModelSpeech(sm251bn, feat, max_label_length=64)
    model_speech.load_model('save_model/' + sm251bn.get_model_name() + '.model.h5')
    model_speech.evaluate_model(data_loader=evalue_data, data_count=-1, out_report=True, show_ratio=True,
                                show_per_step=100)

elif op == 'test':
    sm251bn = SpeechModel251BN(
        input_shape=(audio_length, audio_feature_length, channels),
        output_size=output_size
    )
    feat = Spectrogram()
    model_speech = ModelSpeech(sm251bn, feat, max_label_length=64)
    model_speech.load_model('save_model/' + sm251bn.get_model_name() + '.model.h5')
    res = model_speech.recognize_speech_from_file(filename=fn)
    print('[声学模型语音识别结果:{}]'.format(res))

    model_language = ModelLanguage('model_language')
    model_language.load_model()
    res = model_language.pinyin_to_text(res)
    print('[语音识别最终结果:{}]'.format(res))
