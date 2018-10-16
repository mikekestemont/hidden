import json
from collections import Counter


class Dictionary(object):

    def __init__(self, idx2char=None, char2idx=None, min_freq=0):
        if idx2char is None:
            self.char2idx = {'<UNK>': 0}
            self.idx2char = ['<UNK>']
        else:
            self.char2idx = char2idx
            self.idx2char = idx2char

        self.min_freq = int(min_freq)

    def fit(self, text):
        cnt = Counter(text)

        for char in sorted(cnt.keys()):
            if char not in self.char2idx and cnt[char] > self.min_freq:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

        return self

    def transform(self, text):
        return [self.char2idx.get(char, 0) for char in text]

    def __len__(self):
        return len(self.char2idx)

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump({'idx2char': self.idx2char,
                       'char2idx': self.char2idx},
                       f, indent=4)

    @classmethod
    def load(self, path):
        with open(path, 'r') as f:
            params = json.load(f)

        return Dictionary(**params)
