'''
加载配置文件
'''

import json

DEFAULT_CONFIG_FILENAME = 'asrt_config.json'
_config_dict = None
_pinyin_dict = None
_pinyin_list = None


def load_config_file(filename):
    global _config_dict
    if _config_dict is not None:
        return _config_dict

    with open(filename, 'r', encoding='utf-8') as fr:
        _config_dict = json.load(fr)
    return _config_dict


def load_pinyin_dict(filename):
    '''
    加载拼音字典和拼音列表
    :param filename:
    :return:
    '''

    global _pinyin_list, _pinyin_dict
    if _pinyin_dict is not None and _pinyin_list is not None:
        return _pinyin_list, _pinyin_dict

    _pinyin_list = list()
    _pinyin_dict = dict()
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.read().split('\n')

    for line in lines:
        if len(line) == 0:
            continue
        tokens = line.split('\t')
        _pinyin_list.append(tokens[0])
        _pinyin_dict[tokens[0]] = len(_pinyin_list) - 1
    return _pinyin_list, _pinyin_dict


if __name__ == '__main__':
    l, d = load_pinyin_dict('../dict.txt')
    print(l)
    print(d)