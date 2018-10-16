import argparse
import time
import math
import os

import torch
import torch.nn as nn
import numpy as np

from hidden import data
from hidden import utils
from hidden import modelling


def main():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument('--split_dir', type=str, default='./assets/unannotated/EEBO_splits',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--min_char_freq', type=int, default=100,
                        help='size of word embeddings')
    parser.add_argument('--emsize', type=int, default=64,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=30,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=254,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--make_splits', action='store_true', help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--model_prefix', type=str, default='base',
                        help='path to save the final model')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')

    # preprocess
    dictionary = data.Dictionary(min_freq=args.min_char_freq)
    with open(os.sep.join((args.split_dir, 'train.txt')), 'r') as f:
        dictionary.fit(f.read())
    dictionary.dump(args.model_prefix + '_chardict.json')
    del dictionary

    dictionary = data.Dictionary.load(args.model_prefix + '_chardict.json')

    print('char vocab:', dictionary.idx2char)

    with open(os.sep.join((args.split_dir, 'train.txt')), 'r') as f:
        train = torch.LongTensor(dictionary.transform(f.read()))
    with open(os.sep.join((args.split_dir, 'dev.txt')), 'r') as f:
        dev = torch.LongTensor(dictionary.transform(f.read()))
    with open(os.sep.join((args.split_dir, 'test.txt')), 'r') as f:
        test = torch.LongTensor(dictionary.transform(f.read()))

    print(f'# of characters in train: {len(train)}')
    print(f'# of characters in dev: {len(dev)}')
    print(f'# of characters in test: {len(test)}')

    device = torch.device('cuda' if args.cuda else 'cpu')

    train = utils.batchify(train, args.batch_size).to(device)
    dev = utils.batchify(dev, args.batch_size).to(device)
    test = utils.batchify(test, args.batch_size).to(device)

    # set up model
    vocab_size = len(dictionary)
    lm = modelling.RNNModel(args.model, vocab_size, args.emsize,
                            args.nhid, args.nlayers, args.dropout).to(device)
    optimizer = torch.optim.Adam(lm.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    print(lm)

    def epoch():
        lm.train()

        total_loss = 0.
        start_time = time.time()
        hidden = None

        for batch, i in enumerate(range(0, train.size(0) - 1, args.bptt)):
            data, targets = utils.get_batch(train, i, args.bptt)

            lm.zero_grad()
            output, hidden = lm(data, hidden)
            loss = criterion(output.view(-1, vocab_size), targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.item()
            hidden = modelling.repackage_hidden(hidden)

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                       'loss {:5.6f} | ppl {:8.2f}'.format(
                    epoch_idx, batch * args.batch_size, train.size(0) // args.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

                with torch.no_grad():
                    hid_ = None
                    in_ = torch.randint(vocab_size, (1, 1), dtype=torch.long).to(device)

                    for i in range(90):
                        output, hid_ = lm(in_, hid_)
                        char_weights = output.squeeze().div(args.temperature).exp().cpu()
                        char_idx = torch.multinomial(char_weights, 1)[0]
                        in_.fill_(char_idx)
                        char = dictionary.idx2char[char_idx]
                        print(char, end='')

                print('\n')

                # remove!
                with open(args.model_prefix + '_model.pt', 'wb') as f:
                    torch.save(lm, f)

    def evaluate(data_source):
        lm.eval()
        total_loss = 0.
        hidden = None

        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = utils.get_batch(data_source, i, args.bptt)
                output, hidden = lm(data, hidden)
                output_flat = output.view(-1, vocab_size)
                total_loss += len(data) * criterion(output_flat, targets).item()
                hidden = modelling.repackage_hidden(hidden)

        return total_loss / len(data_source)

    lr = args.lr
    best_val_loss = None

    try:
        for epoch_idx in range(1, args.epochs+1):
            epoch_start_time = time.time()
            epoch()
            val_loss = evaluate(dev)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.6f}'.format(epoch_idx, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.model_prefix + '_model.pt', 'wb') as f:
                    torch.save(lm, f)
                best_val_loss = val_loss
            else:
                lr *= 0.5
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    with open(args.model_prefix + '_model.pt', 'rb') as f:
        lm = torch.load(f)
        lm.rnn.flatten_parameters()

    test_loss = evaluate(test)
    print('=' * 89)
    print('| End of training | test loss {:5.6f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

if __name__ == '__main__':
    main()