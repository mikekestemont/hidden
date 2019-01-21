# coding: utf-8
import argparse
import time
import glob
import math
import os
import random
import json
from glob import glob

import numpy as np
from sklearn.metrics import classification_report

from hidden.encoding import LabelEncoder

def main():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument('--gt_file', type=str, default='data/eltec_gt_splits/eng/test.txt',
                        help='random seed')
    parser.add_argument('--results_dir', type=str, default='data/results',
                        help='use CUDA')
    args = parser.parse_args()
    print(args)

    gold = [l.rstrip().split('\t') for l in open(args.gt_file)]
    _, gold = zip(*gold)
    
    for fn in glob(f'{args.results_dir}/*.txt'):
        print('=' * 64)
        print(f'-> results for {fn}')

        silver = [l.rstrip().split('\t') for l in open(fn)]
        _, silver = zip(*silver)

        report = classification_report(gold, silver)
        print(report)
        

if __name__ == '__main__':
    main()