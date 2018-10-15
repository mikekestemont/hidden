#!usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
from sklearn.preprocessing import LabelEncoder as ScikitEncoder


class LabelEncoder(ScikitEncoder):
    """
    Wrapper around scikit's LabelEncoder to
    enable saving and loading of a fitted
    encoder.
    """

    def __init__(self):
        super(LabelEncoder, self).__init__()

    def save(self, p):
        with open(p, 'w') as f:
            f.write(json.dumps(list(self.classes_)))

    @classmethod
    def load(self, p):
        encoder = ScikitEncoder()
        with open(p, 'r') as f:
            encoder.classes_ = np.array(json.loads(f.read()))
        return encoder
