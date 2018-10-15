import shutil
import os
import glob
import json
import random
from collections import Counter

import torch


class Dictionary(object):

    def __init__(self, idx2char=None, char2idx=None, min_freq=0):
        if idx2char is None:
            self.char2idx = {'<UNK>': 0}
            self.idx2char = ['<UNK>']
        else:
            self.char2idx = char2idx
            self.idx2char = idx2char

        self.min_freq = min_freq

    def fit(self, text):
        cnt = Counter(text)

        for char in sorted(cnt.keys()):
            if char not in self.char2idx and cnt[char] > self.min_freq:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

        return self

    def __len__(self):
        return len(self.char2idx)

    def dump(self, model_prefix):
        with open(model_prefix + '_dict.json', 'w') as f:
            json.dump({'idx2char': self.idx2char,
                       'char2idx': self.char2idx},
                       f, indent=4)

    @classmethod
    def load(self, model_prefix):
        with open(model_prefix + '_dict.json', 'r') as f:
            params = json.load(f)

        return Dictionary(**params)


class Corpus:

    def __init__(self, input_dir, split_dir,
                 min_freq=2,
                 seed=1345, train_size=0.9, make_splits=False):
        self.input_dir = input_dir
        self.split_dir = split_dir
        self.seed = seed
        self.dictionary = Dictionary(min_freq=min_freq)
        
        if make_splits:
            self.make_splits(train_size)
            with open(os.sep.join((self.split_dir, 'train.txt'))) as f:
                self.dictionary.fit(f.read())

    def make_splits(self, train_size):
        try:
            shutil.rmtree(self.split_dir)
        except FileNotFoundError:
            pass
        os.mkdir(self.split_dir)

        self.streams = {}
        for sp in 'train dev test'.split():
            fn = os.sep.join((self.split_dir, sp)) + '.txt'
            self.streams[sp] = open(fn, 'w')

        random.seed(self.seed)

        for inf in sorted(glob.glob(self.input_dir + '/*.txt')):
            with open(inf, 'r') as f:
                text = f.read()

            rest_size = int((len(text) - len(text) * train_size) / 2)
            random_start = random.randint(0, len(text) - rest_size * 2)

            dev_start, dev_end = random_start, random_start + int(rest_size)
            test_start, test_end = dev_end + 1, dev_end + int(rest_size) + 1

            self.streams['dev'].write(text[dev_start : dev_end])
            self.streams['test'].write(text[test_start : test_end])

            if dev_start < 0:
                self.streams['train'].write(text[:dev_start])
            if test_end < len(text) - 1:
                self.streams['train'].write(text[test_end:])

        for fn in self.streams:
            self.streams[fn].close()

    def get_split(self, stream):
        with open(os.sep.join((self.split_dir, stream)) + '.txt') as f:
            text = f.read()

        ints = []
        for char in text:
            try:
                ints.append(self.dictionary.char2idx[char])
            except KeyError:
                ints.append(0)

        return torch.LongTensor(ints)
