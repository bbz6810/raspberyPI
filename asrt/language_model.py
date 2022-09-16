'''
asrt语音识别的语言模型
N-gram模型
'''

import os
from asrt.utils.ops import get_symbol_dict, get_language_model


class ModelLanguage:
    '''
    asrt专用N-gram语言模型
    '''

    def __init__(self, model_path):
        self.model_path = model_path
        self.dict_pinyin = dict()
        self.model1 = dict()
        self.model2 = dict()

    def load_model(self):
        '''
        加载N-gram模型到内存
        :return:
        '''
        self.dict_pinyin = get_symbol_dict('dict.txt')
        self.model1 = get_language_model(os.path.join(self.model_path, 'language_model1.txt'))
        self.model2 = get_language_model(os.path.join(self.model_path, 'language_model2.txt'))
        model = (self.dict_pinyin, self.model1, self.model2)
        return model

    def pinyin_to_text(self, list_pinyin, beam_size=100):
        '''
        拼音转文本，一次性取得全部结果
        :param list_pinyin:
        :param beam_size:
        :return:
        '''
        result = []
        tmp_result_last = []
        for item_pinyin in list_pinyin:
            tmp_result = self.pinyin_stream_decode(tmp_result_last, item_pinyin, beam_size)
            if len(tmp_result) == 0 and len(tmp_result_last) > 0:
                result.append(tmp_result_last[0][0])
                tmp_result = self.pinyin_stream_decode([], item_pinyin, beam_size)
                if len(tmp_result) > 0:
                    result.append(tmp_result[0][0])
                tmp_result = []
            tmp_result_last = tmp_result

        if len(tmp_result_last) > 0:
            result.append(tmp_result_last[0][0])
        return ''.join(result)

    def pinyin_stream_decode(self, temple_result, item_pinyin, beam_size):
        '''
        拼音流式解码，逐字转换，每次返回中间结果
        :param temple_result:
        :param item_pinyin:
        :param beam_size:
        :return:
        '''
        if item_pinyin not in self.dict_pinyin:
            return []
        cur_words = self.dict_pinyin[item_pinyin]
        if len(temple_result) == 0:
            lst_result = list()
            for word in cur_words:
                lst_result.append([word, 1.0])
            return lst_result

        new_result = list()
        for sequence in temple_result:
            for cur_word in cur_words:
                tuple2_word = sequence[0][-1] + cur_word
                if tuple2_word not in self.model2:
                    continue

                prob_origin = sequence[1]
                count_two_word = float(self.model2[tuple2_word])
                count_one_word = float(self.model1[tuple2_word[-2]])
                cur_probility = prob_origin * count_two_word / count_one_word
                new_result.append([sequence[0] + cur_word, cur_probility])

        new_result = sorted(new_result, key=lambda x: x[1], reverse=True)
        if len(new_result) > beam_size:
            return new_result[:beam_size]
        return new_result


if __name__ == '__main__':
    ml = ModelLanguage('model_language')
    ml.load_model()

    _str = ['xiao3', 'du4', 'xiao3', 'du4']
    _result = ml.pinyin_to_text(_str)
    print('拼音转文字结果', _result)
