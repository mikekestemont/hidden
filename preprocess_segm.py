import argparse
import os
import glob
import re
import shutil

from sklearn.model_selection import train_test_split as split
from bs4 import BeautifulSoup as soup


def main():
    parser = argparse.ArgumentParser(description='Preprocess a directory of XML files')
    parser.add_argument('--indir', type=str, default='data/eltec_gt/eng',
                        help='location of the original Eltec XML-files')
    parser.add_argument('--outdir', type=str, default='data/eltec_gt_splits',
                        help='location of the train, dev, and test spit')
    parser.add_argument('--seed', type=int, default=12345,
                        help='random seed')
    parser.add_argument('--train_size', type=float, default=.9,
                        help='Proportion of training data')

    args = parser.parse_args()
    print(args)

    filenames = glob.glob(os.sep.join((f'{args.indir}', '**', '*.tsv')), recursive=True)

    train, rest = split(filenames,
                        train_size=args.train_size,
                        shuffle=True,
                        random_state=args.seed)
    dev, test = split(rest,
                      train_size=0.5,
                      shuffle=True,
                      random_state=args.seed)

    print(f'# train files: {len(train)}')
    print(f'# dev files: {len(dev)}')
    print(f'# test files: {len(test)}')

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    out_folder = f'{args.outdir}/{os.path.basename(args.indir)}'
    try:
        shutil.rmtree(out_folder)
    except FileNotFoundError:
        pass
    os.mkdir(out_folder)

    with open(f'{out_folder}/train.txt', 'w') as f:
        for fn in train:
            for line in open(fn):
                f.write(line)

    with open(f'{out_folder}/dev.txt', 'w') as f:
        for fn in dev:
            for line in open(fn):
                f.write(line)

    with open(f'{out_folder}/test.txt', 'w') as f:
        for line in open(fn):
                f.write(line)


if __name__ == '__main__':
    main()