import collections

import torch
from torch.autograd import Variable


class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()

        self.alphabet = alphabet
        self.alphabet.append(ord('_'))

        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def encode(self, text):
        try:
            if isinstance(text, str):
                text_ = []
                for char in text:
                    if self._ignore_case:
                        char.lower()
                    char = ord(char)
                    if char in self.dict.keys():
                        text_.append(self.dict[char])
                    else:
                        text_.append(0)
                text = text_
                length = [len(text)]
            elif isinstance(text, collections.Iterable):
                length = [len(s) for s in text]
                text = ''.join(text)
                text, _ = self.encode(text)
        except KeyError as e:
            print(e)
            for ch in text:
                if ord(ch) not in self.dict.keys():
                    print('Not Found Char: {} - {}'.format(ch, ord(ch)))
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):

        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length
            if raw:
                return ''.join([chr(self.alphabet[i - 1]) for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(chr(self.alphabet[t[i] - 1]))
                return ''.join(char_list)
        else:
            assert t.numel() == length.sum()
            text = []
            index = 0
            _ = length.numel()
            for i in range(length.numel()):
                l = length[i]
                text.append(
                    self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw)
                )
                index += l
            return text


class averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
