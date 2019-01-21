from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import spacy

class NLTK_segmenter:
    def __init__(self):
        pass
    
    def segment(self, paragraph):
        return tuple(tuple(word_tokenize(s)) for s in sent_tokenize(paragraph))

class Spacy_segmenter:
    def __init__(self, language='en'):
        self.nlp = spacy.load('en')
    
    def segment(self, paragraph):
        sentences = []
        for sent in self.nlp(paragraph).sents:
            sentences.append([t.text.strip() for t in sent])
        return sentences

