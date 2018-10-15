# coding: utf-8
import argparse
import time
import glob
import math
import os
import random

import torch
import torch.nn as nn
import torch.onnx

from hidden import data
from hidden import utils
from hidden import modelling

import numpy as np
from lxml import etree

from hidden.encoding import LabelEncoder

import xml.dom.minidom
from xml.dom.minidom import Node
node_types = {1: 'ELEMENT', 2: "ATTRIBUTE", 3: 'TEXT'}

def load_file(path):
    text, labels = '', ''
    book = xml.dom.minidom.parse(path)
    items = list(book.getElementsByTagName('chapter'))
    if not items:
        items = list(book.getElementsByTagName('text'))
    for chapter in items:
        t_ = ''
        for element in chapter.childNodes:
            t_ = ''
            t = node_types[element.nodeType]
            if t == 'TEXT':
                t_ = element.nodeValue
                text += t_
                labels += ('O' * len(t_))
            elif t == 'ELEMENT':
                t_ = element.firstChild.nodeValue
                if t_ and element.tagName in ('quote', 'mention'):
                    text += t_
                    if element.tagName == 'quote':
                        labels += 'B' + 'I' * (len(t_) - 1)
                    elif element.tagName == 'mention':
                        labels += ('O' * len(t_))
    assert len(labels) == len(text)
    return text, labels

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
    parser.add_argument('--input_dir', type=str, default='./assets/annotated',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--train_size', type=float, default=.9,
                        help='sequence length')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='sequence length')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--model_prefix', type=str, default='base',
                        help='path to save the final model')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--full_finetune', action='store_true', default=False,
                        help='use CUDA')


    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")
    random.seed(args.seed)

    train_text, train_labels = '', ''
    dev_text, dev_labels = '', ''
    test_text, test_labels = '', ''

    for path in glob.glob(f'{args.input_dir}/*.xml'):
        text, labels = load_file(path)
        
        rest_size = int((len(text) - len(text) * args.train_size) / 2)
        random_start = random.randint(0, len(text) - rest_size * 2)

        dev_start, dev_end = random_start, random_start + int(rest_size)
        test_start, test_end = dev_end + 1, dev_end + int(rest_size) + 1

        dev_text += text[dev_start : dev_end]
        dev_labels += labels[dev_start : dev_end]

        test_text += text[test_start : test_end]
        test_labels += labels[test_start : test_end]

        if dev_start < 0:
            train_text += text[:dev_start]
            train_labels += labels[:dev_start]
        if test_end < len(text) - 1:
            train_text += text[test_end:]
            train_labels += labels[test_end:]

        assert len(train_text) == len(train_labels)
        assert len(dev_text) == len(dev_labels)
        assert len(test_text) == len(test_labels)
    
    print('train chars:', len(train_text))
    print('dev chars:', len(dev_text))
    print('test chars:', len(test_text))

    dictionary = data.Dictionary.load(args.model_prefix+'_chardict.json')
    train_text = to_ints(train_text, dictionary)
    dev_text = to_ints(dev_text, dictionary)
    test_text = to_ints(test_text, dictionary)

    train_X = utils.batchify(train_text, args.batch_size, device)
    dev_X = utils.batchify(dev_text, args.batch_size, device)
    test_X = utils.batchify(test_text, args.batch_size, device)

    encoder = LabelEncoder().fit(list(train_labels))
    encoder.save(args.model_prefix+'_labeldict.json')
    train_Y = encoder.transform(list(train_labels))
    dev_Y = encoder.transform(list(dev_labels))
    test_Y = encoder.transform(list(test_labels))

    train_Y = torch.LongTensor(train_Y).to(device)
    dev_Y = torch.LongTensor(dev_Y).to(device)
    test_Y = torch.LongTensor(test_Y).to(device)

    train_Y = utils.batchify(train_Y, args.batch_size, device)
    dev_Y = utils.batchify(dev_Y, args.batch_size, device)
    test_Y = utils.batchify(test_Y, args.batch_size, device)

    with open(args.model_prefix + '_model.pt', 'rb') as f:
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

    def train(current_epoch):
        dsm.train()
        total_loss = 0.
        start_time = time.time()
        hidden = None

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
                    current_epoch, batch * args.batch_size, len(train_X) // args.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss))
                total_loss = 0
                start_time = time.time()

                # remove later!
                #with open(args.model_prefix + '_dsm.pt', 'wb') as f:
                #    torch.save(dsm, f)

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
                train(epoch)
                val_loss = evaluate(dev_X, dev_Y)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | '.format(epoch, (time.time() - epoch_start_time), val_loss))
                print('-' * 89)
                
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.model_prefix + '_dsm.pt', 'wb') as f:
                        torch.save(dsm, f)
                    best_val_loss = val_loss
                else:
                    lr *= 0.5
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

    # 2. unfreeze the entire net:
    if args.full_finetune:
        for param in dsm.parameters():
            param.requires_grad = True
        optimizer = torch.optim.SGD(dsm.parameters(), lr=args.lr * 0.1, weight_decay=1e-5)
        train_loop()

    # Load the best saved model.
    with open(args.model_prefix + '_dsm.pt', 'rb') as f:
        model = torch.load(f)
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_X, test_Y)
    print('=' * 89)
    print('| End of training | test loss {:5.6f}'.format(test_loss))
    print('=' * 89)

if __name__ == '__main__':
    main()