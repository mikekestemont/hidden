import json
import re
from collections import Counter

from bs4 import BeautifulSoup as soup
from lxml import etree


spaces = re.compile(r' +')

def ced_reader(fn):
    print(fn)

    with open(fn) as inf:
        s = etree.XML(inf.read().encode('ascii', 'xmlcharrefreplace'))

    for comment in s.xpath('.//comment'):
        comment.getparent().remove(comment)

    body = list(s.findall('.//dialogueText'))[0]

    samples = list(body.findall('.//sample'))
    if not samples:
        samples = [body]

    characters, labels = '', ''

    for sample in samples:
        for child in sample:
            if child.tag in ('pagebreak', 'omission'):
                pass
            elif child.tag in ('head', 'nonSpeech', 'font', 'foreign', 'emendation'):
                text = ''.join(child.itertext(with_tail=True))
                text = re.sub(spaces, ' ', text)
                if text:
                    characters += text
                    labels += ('O' * len(text))
            elif child.tag == 'dialogue':
                text = ''.join(child.itertext(with_tail=True))
                text = re.sub(spaces, ' ', text)
                if text:
                    characters += text
                    labels += 'B' + 'I' * (len(text) - 1)

    assert len(characters) == len(labels)

    formatted = '\n'.join([json.dumps((c, l)) for c, l in list(zip(characters, labels))])
    return formatted + '\n'

def de_reader(fn):
    print(fn)

    characters, labels = [], []

    with open(fn, 'r') as f:
        lines = list(f.readlines())[1:10000]

    for idx, line in enumerate(lines):
        try:
            token, speech, pos, *_ = line.strip().split('\t')
        except ValueError:
            continue

        if speech == '0':
            label = 'O'
        elif speech == '1':
            label = 'I'

        characters_ = token
        labels_ = ''.join([label] * len(characters_))
        
        if not pos.startswith('$'):
            characters_ = ' ' + characters_
            labels_ = labels_[0] + labels_

        characters += characters_
        labels += labels_

    assert len(characters) == len(labels)

    #for a, b in zip(characters, labels):
    #    print(a, b)

    formatted = '\n'.join([json.dumps((c, l)) for c, l in list(zip(characters, labels))])
    return formatted + '\n'

readers = {'CED-en': ced_reader, 'kern_rich-de': de_reader}

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

