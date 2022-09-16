'''
常用操作函数定义
'''

import wave
import difflib
# import matplotlib.pyplot as plt
import numpy as np


def read_wav_data(filename):
    wav = wave.open(filename, 'rb')
    num_frame = wav.getnframes()  # 帧数
    num_channel = wav.getnchannels()  # 声道
    framerate = wav.getframerate()  # 帧率
    num_sample_width = wav.getsampwidth()  # 获取实际的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    wav.close()

    wave_data = np.fromstring(str_data, dtype=np.short)  # 将声音文件数据转换成数组矩阵形式
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形
    wave_data = wave_data.T
    return wave_data, framerate, num_channel, num_sample_width


def ctc_decode_delete_tail_blank(ctc_decode_list):
    '''
    删除ctc解码后的空白元素【-1】
    :param ctc_decode_list:
    :return:
    '''
    p = 0
    while p < len(ctc_decode_list) and ctc_decode_list[p] != -1:
        p += 1
    return ctc_decode_list[:p]


def get_edit_distance(str1, str2):
    '''
    计算两个串的编辑距离
    :param str1:
    :param str2:
    :return:
    '''
    leven_cost = 0
    sequence_match = difflib.SequenceMatcher(None, str1, str2)
    for tag, index1, index2, indexj1, indexj2 in sequence_match.get_opcodes():
        if tag == 'replace':
            leven_cost += max(index2 - index1, indexj2 - indexj1)
        elif tag == 'insert':
            leven_cost += (indexj2 - indexj1)
        elif tag == 'delete':
            leven_cost += (index2 - index1)
    return leven_cost


def get_symbol_dict(dict_filename):
    '''
    读取拼音汉字的字典文件
    返回读取后的字典
    :param dict_filename:
    :return:
    '''

    txt_obj = open(dict_filename, 'r', encoding='utf-8')
    txt_text = txt_obj.read()
    txt_obj.close()
    txt_lines = txt_text.split('\n')
    dic_symbol = {}
    for i in txt_lines:
        list_symbol = []
        if i != '':
            txt_l = i.split('\t')
            pinyin = txt_l[0]
            for word in txt_l[1]:
                list_symbol.append(word)
            dic_symbol[pinyin] = list_symbol
    return dic_symbol


def get_language_model(model_language_filename):
    '''
    读取语言模型文件
    返回读取后的模型
    :param model_language_filename:
    :return:
    '''

    txt_obj = open(model_language_filename, 'r', encoding='utf-8')
    txt_text = txt_obj.read()
    txt_obj.close()
    txt_lines = txt_text.split('\n')

    dic_model = {}
    for i in txt_lines:
        if i != '':
            txt_l = i.split('\t')
            if len(txt_l) == 1:
                continue
            dic_model[txt_l[0]] = txt_l[1]
    return dic_model
