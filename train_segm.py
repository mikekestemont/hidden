import argparse
import time
import glob
import math
import os
import shutil
import json

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split as split

from hidden import data
from hidden import utils
from hidden import modelling

import numpy as np
from lxml import etree

from hidden.encoding import LabelEncoder

def to_ints(text, dictionary):
    return torch.LongTensor([dictionary.char2idx.get(c, 0) for c in text])


def main():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument('--split_dir', type=str, default='data/eltec_gt_splits/eng',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=75,
                        help='sequence length')
    parser.add_argument('--train_size', type=   float, default=.8,
                        help='sequence length')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='sequence length')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--lm_model_dir', type=str, default='data/lm_models/ELTeC-eng',
                        help='path to save the final model')
    parser.add_argument('--model_dir', type=str, default='data/segm_models',
                        help='path to save the final model')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--no_finetune', action='store_true', default=False,
                        help='use CUDA')
    parser.add_argument('--reverse', default=False, action='store_true',
                        help='backwards LM')

    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    
    args.model_prefix = f'{args.model_dir}/{os.path.basename(args.split_dir)}'

    if not args.reverse:
        try:
            shutil.rmtree(args.model_prefix)
        except FileNotFoundError:
            pass
        os.mkdir(args.model_prefix)
        args.model_prefix = f'{args.model_prefix}/'
        args.lm_model_prefix = f'{args.lm_model_dir}/'
    else:
        args.model_prefix = f'{args.model_dir}/{os.path.basename(args.split_dir)}/rev_'
        args.lm_model_prefix = f'{args.lm_model_dir}/rev_'

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')

    device = torch.device('cuda' if args.cuda else 'cpu')

    train = [l.rstrip().split('\t') for l in open(f'{args.split_dir}/train.txt')]
    if args.reverse:
        train = train[::-1]
    train_text, train_labels = zip(*train)

    dev = [l.rstrip().split('\t') for l in open(f'{args.split_dir}/dev.txt')]
    if args.reverse:
        dev = dev[::-1]
    dev_text, dev_labels = zip(*dev)

    test = [l.rstrip().split('\t') for l in open(f'{args.split_dir}/test.txt')]
    if args.reverse:
        test = test[::-1]
    test_text, test_labels = zip(*test)

    dictionary = data.Dictionary.load(args.lm_model_dir+'/chardict.json')
    train_text = to_ints(train_text, dictionary)
    dev_text = to_ints(dev_text, dictionary)
    test_text = to_ints(test_text, dictionary)

    train_X = utils.batchify(train_text, args.batch_size).to(device)
    dev_X = utils.batchify(dev_text, args.batch_size).to(device)
    test_X = utils.batchify(test_text, args.batch_size).to(device)

    if not args.reverse:
        encoder = LabelEncoder().fit(list(train_labels))
        encoder.save(args.model_prefix + 'labeldict.json')
    else:
        encoder = LabelEncoder.load(args.model_prefix.replace('rev_', '')+'labeldict.json')
    
    train_Y = encoder.transform(list(train_labels))
    dev_Y = encoder.transform(list(dev_labels))
    test_Y = encoder.transform(list(test_labels))

    train_Y = torch.LongTensor(train_Y).to(device)
    dev_Y = torch.LongTensor(dev_Y).to(device)
    test_Y = torch.LongTensor(test_Y).to(device)

    train_Y = utils.batchify(train_Y, args.batch_size).to(device)
    dev_Y = utils.batchify(dev_Y, args.batch_size).to(device)
    test_Y = utils.batchify(test_Y, args.batch_size).to(device)

    print(args.lm_model_prefix + 'lm_model.pt')
    with open(args.lm_model_prefix + 'lm_model.pt', 'rb') as f:
        dsm = torch.load(f)

    nclasses = len(encoder.classes_)
    dsm.decoder = torch.nn.Linear(dsm.nhid, nclasses)
    print(dsm)

    dsm = dsm.to(device)

    lr = args.lr
    best_val_loss = None
    criterion = nn.CrossEntropyLoss()

    def get_batch(source, i, bptt):
        seq_len = min(bptt, len(source) - i)
        return source[i : i+seq_len]

    def train(current_epoch, lr):
        dsm.train()
        total_loss = 0.
        start_time = time.time()
        hidden = None

        for g in optimizer.param_groups:
            g['lr'] = lr

        for batch, i in enumerate(range(0, train_X.size(0) - 1, args.bptt)):            
            data = get_batch(train_X, i, args.bptt)
            targets = get_batch(train_Y, i, args.bptt)

            dsm.zero_grad()
            output, hidden = dsm(data, hidden)

            output = output.view(-1, nclasses)
            targets = targets.view(-1)

            loss = criterion(output, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dsm.parameters(), args.clip)
            optimizer.step()
            total_loss += loss.item()
            hidden = modelling.repackage_hidden(hidden)

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.5f} | '
                        'loss {:5.6f}'.format(
                    current_epoch, batch * args.batch_size, len(train_X) // args.bptt * args.batch_size, lr,
                    elapsed * 1000 / args.log_interval, cur_loss))
                total_loss = 0
                start_time = time.time()

    def evaluate(X, Y):
        dsm.eval()
        total_loss = 0.
        hidden = None

        with torch.no_grad():

            for i in range(0, X.size(0) - 1, args.bptt):
                data = get_batch(X, i, args.bptt)
                targets = get_batch(Y, i, args.bptt)
                output, hidden = dsm(data, hidden)
                output = output.view(-1, nclasses)
                targets = targets.view(-1)
                total_loss += len(data) * criterion(output, targets).item()
                hidden = modelling.repackage_hidden(hidden)

        return total_loss / len(X)

    def train_loop():
        lr = args.lr
        best_val_loss = None
        
        try:
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                train(epoch, lr)
                val_loss = evaluate(dev_X, dev_Y)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | '.format(epoch, (time.time() - epoch_start_time), val_loss))
                
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.model_prefix + 'dsm.pt', 'wb') as f:
                        torch.save(dsm, f)
                    print('>>> saving model')    
                    best_val_loss = val_loss
                elif val_loss >= best_val_loss:
                    lr *= 0.5
                    print(f'>>> lowering learning rate to {lr}')
                    print('-' * 89)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')


    # 1. freeze the rnn:
    for param in dsm.parameters():
        param.requires_grad = False
    dsm.decoder.weight.requires_grad = True
    dsm.decoder.bias.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dsm.parameters()),
                                 lr=args.lr, weight_decay=1e-5)
    train_loop()

    # Load the best saved model.
    with open(args.model_prefix + 'dsm.pt', 'rb') as f:
        model = torch.load(f)
        model.rnn.flatten_parameters()

    # 2. unfreeze the entire net:
    if not args.no_finetune:
        print('Started full finetune')
        for param in dsm.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(dsm.parameters(),
                                 lr=args.lr, weight_decay=1e-5)
        train_loop()

    # Load the best saved model.
    with open(args.model_prefix + 'dsm.pt', 'rb') as f:
        model = torch.load(f)
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_X, test_Y)
    print('=' * 89)
    print('| End of training | test loss {:5.6f}'.format(test_loss))
    print('=' * 89)

if __name__ == '__main__':
    main()