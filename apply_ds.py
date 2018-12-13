# coding: utf-8
import argparse
import time
import glob
import math
import os
import random
import json

import numpy as np
import torch
from sklearn.metrics import classification_report

from hidden import data
from hidden import utils
from hidden import modelling
from hidden.encoding import LabelEncoder


def main():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument('--infile', type=str, default='./assets/test.txt',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--split_dir', type=str, default='./assets/annotated/kern_splits',
                        help='location of the data corpus')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--model_prefix', type=str, default='base',
                        help='path to save the final model')
    parser.add_argument('--direction', type=str, default='both',
                        help='path to save the final model')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold to select I-tag')
    args = parser.parse_args()
    print(args)

    assert args.direction in ('both', 'left', 'right')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')

    encoder = LabelEncoder.load(args.model_prefix + '_labeldict.json')

    device = torch.device('cuda' if args.cuda else 'cpu')
    random.seed(args.seed)

    dictionary = data.Dictionary.load(args.model_prefix+'_chardict.json')

    test = [json.loads(l) for l in open(os.sep.join((args.split_dir, 'test.txt')), 'r')]
    test_text, test_labels = zip(*test)

    test_text = test_text[:10000]
    test_labels = test_labels[:10000]

    def predict_proba(test_text, reverse=False):
        if reverse:
            p = args.model_prefix + '-rev_dsm.pt'
            test_text = test_text[::-1]
        else:
            p = args.model_prefix + '_dsm.pt'

        with open(p, 'rb') as f:
            dsm = torch.load(f)
            dsm.rnn.flatten_parameters()
        dsm = dsm.to(device)

        ints = torch.LongTensor([dictionary.char2idx.get(c, 0) for c in test_text])
        hid_ = None

        dsm.eval()
        output = np.zeros((len(test_text), len(encoder.classes_)))

        with torch.no_grad():
            for idx, i in enumerate(ints):
                i = i.to(device)
                out_, hid_ = dsm(i.view(1, -1), hid_)
                output[idx, :] = out_.squeeze().cpu().numpy()

        if reverse:
            return output[::-1]
        else:
            return output

    if args.direction == 'both':
        proba_left = predict_proba(test_text)
        proba_right = predict_proba(test_text, reverse=True)
        proba = (proba_left + proba_right) / 2

    elif args.direction == 'left':
        proba = predict_proba(test_text)

    elif args.direction == 'right':
        proba = predict_proba(test_text, reverse=True)

    if args.threshold:    
        idx = list(encoder.classes_).index('I')
        output = ['I' if o >= args.threshold else 'O' for o in proba[:, idx]]
    else:
        output = np.argmax(proba, axis=-1)
        output = encoder.inverse_transform(output)
    
    report = classification_report(test_labels, output)
    print(report)

    opened = False
    with open('output.html', 'w') as f:
        for char, out in zip(test_text, output):
            if out == 'I' and opened == False:
                f.write('<b>')
                opened = True

            f.write(char)

            if out == 'O' and opened == True:
                f.write('</b>')
                opened = False


if __name__ == '__main__':
    main()