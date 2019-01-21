import argparse

from hidden.baselines import NLTK_segmenter, Spacy_segmenter
import hidden.utils as u

def main():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument('--language', type=str, default='eng',
                        help='random seed')
    args = parser.parse_args()
    print(args)
    
    test_tuples = u.tuples_from_file(f'data/eltec_gt_splits/{args.language}/test.txt')
    test_text = u.string_from_tuples(test_tuples)
    print(len(test_text))

    """
    # NLTK:
    nltk = NLTK_segmenter()
    segmented = nltk.segment(test_text)

    with open(f'data/results/{args.language}_nltk.txt', 'w') as f:
        for x, y in u.char_tuples(test_text, segmented):
            f.write(f'{x}\t{y}\n')
    """

    spacy = Spacy_segmenter('en')
    segmented = spacy.segment(test_text)
    
    tuples = u.char_tuples(test_text, segmented)
    print(len(tuples))
    with open(f'data/results/{args.language}_spacy.txt', 'w') as f:
        for x, y in tuples:
            f.write(f'{x}\t{y}\n')


if __name__ == '__main__':
    main()