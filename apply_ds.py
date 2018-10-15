# coding: utf-8
import argparse
import time
import glob
import math
import os
import random

import numpy as np
import torch

from hidden import data
from hidden import utils
from hidden import modelling
from hidden.encoding import LabelEncoder


def to_ints(text, dictionary):
    ints = []
    for char in text:
        try:
            ints.append(dictionary.char2idx[char])
        except KeyError:
            ints.append(0)
    return torch.LongTensor(ints)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument('--infile', type=str, default='./assets/test.txt',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--model_prefix', type=str, default='base',
                        help='path to save the final model')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    encoder = LabelEncoder.load(args.model_prefix+'_labeldict.json')

    device = torch.device("cuda" if args.cuda else "cpu")
    random.seed(args.seed)

    with open(args.infile, 'r') as f:
        text = f.read()

    dictionary = data.Dictionary.load(args.model_prefix+'_chardict.json')

    with open(args.model_prefix + '_dsm.pt', 'rb') as f:
        dsm = torch.load(f)
        dsm.rnn.flatten_parameters()
    dsm = dsm.to(device)

    ints = to_ints(text, dictionary)
    preds = []
    hid_ = None

    dsm.eval()
    output = np.zeros((len(text), len(encoder.classes_)))
    with torch.no_grad():
        for idx, i in enumerate(ints):
            i = i.to(device)
            out_, hid_ = dsm(i.view(1, -1), hid_)
            output[idx, :] = out_.squeeze().cpu().numpy()
    
    output = np.argmax(output, axis=-1)
    output = encoder.inverse_transform(output)

    for char, out in zip(text, output):
        print('  '.join((repr(char), out)))


if __name__ == '__main__':
    main()