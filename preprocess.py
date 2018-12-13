import argparse
import os
import glob
import re
import shutil

from sklearn.model_selection import train_test_split as split
from bs4 import BeautifulSoup as soup

"""
python --indir="assets/unannotated/DTA" --outdir="assets/unannotated/DTA_splits"
"""

def plain_text(fn, n=3):
    print(fn)
    with open(fn) as inf:
        s = soup(inf.read(), 'lxml')
    for script in s(['script', 'style']):
        script.extract()
    text = s.find('text').text

    text = re.sub(r'oͤ', 'ö', text)
    text = re.sub(r'uͤ', 'ü', text)
    text = re.sub(r'aͤ', 'ä', text)

    text = re.sub(r'\-\s*\n+\s*', '', text)
    text = re.sub(r'¬\s*\n+\s*', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text


def main():
    parser = argparse.ArgumentParser(description='Preprocess a directory of XML files')
    parser.add_argument('--indir', type=str, default='assets/unannotated/DTA',
                        help='location of the original TEI-files')
    parser.add_argument('--outdir', type=str, default='assets/unannotated/DTA_splits',
                        help='location of the train, dev, and test spit')
    parser.add_argument('--seed', type=int, default=12345,
                        help='random seed')
    parser.add_argument('--train_size', type=float, default=.9,
                        help='Proportion of training data')

    args = parser.parse_args()
    print(args)

    filenames = glob.glob(os.sep.join((args.indir, '**', '*.xml')), recursive=True)

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

    try:
        shutil.rmtree(args.outdir)
    except FileNotFoundError:
        pass
    os.mkdir(args.outdir)

    with open(os.sep.join((args.outdir, 'train.txt')), 'w') as f:
        for fn in train:
            f.write(plain_text(fn) + '\n')

    with open(os.sep.join((args.outdir, 'dev.txt')), 'w') as f:
        for fn in dev:
            f.write(plain_text(fn) + '\n')

    with open(os.sep.join((args.outdir, 'test.txt')), 'w') as f:
        for fn in test:
            f.write(plain_text(fn) + '\n')


if __name__ == '__main__':
    main()