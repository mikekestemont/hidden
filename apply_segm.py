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
    parser.add_argument('--split_dir', type=str, default='data/eltec_gt_splits/eng',
                        help='location of the data corpus')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--model_prefix', type=str, default='data/segm_models/eng/',
                        help='path to save the final model')
    parser.add_argument('--lm_model_prefix', type=str, default='data/lm_models/ELTeC-eng/',
                        help='path to save the final model')
    parser.add_argument('--results_dir', type=str, default='data/results',
                        help='path to save the final model')
    parser.add_argument('--direction', type=str, default='both',
                        help='path to save the final model')
    args = parser.parse_args()
    print(args)

    assert args.direction in ('both', 'left', 'right')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')

    encoder = LabelEncoder.load(args.model_prefix + 'labeldict.json')

    device = torch.device('cuda' if args.cuda else 'cpu')
    random.seed(args.seed)

    dictionary = data.Dictionary.load(args.lm_model_prefix + 'chardict.json')

    test = [l.rstrip().split('\t') for l in open(os.sep.join((args.split_dir, 'test.txt')))]
    test_text, test_labels = zip(*test)

    test_text = test_text
    test_labels = test_labels

    def predict_proba(test_text, reverse=False):
        if reverse:
            p = args.model_prefix + 'rev_dsm.pt'
            test_text = test_text[::-1]
        else:
            p = args.model_prefix + 'dsm.pt'

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

    output = np.argmax(proba, axis=-1)
    output = encoder.inverse_transform(output)
    
    report = classification_report(test_labels, output)
    print(report)

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    
    outfn = f'{args.results_dir}/{os.path.basename(args.split_dir)}_ours.txt'
    print('-> writing output to', outfn)
    with open(outfn, 'w') as f:
        for c, l in zip(test_text, output):
            f.write(f'{c}\t{l}\n')

if __name__ == '__main__':
    main()