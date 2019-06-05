#!/usr/bin/env python3

# Predict with multilingual text classifier.

import sys
import os
import json

import numpy as np

from logging import warning, error

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score

from common import load_fasttext_data_multi
from common import text_to_indices, label_to_index
from common import NullOutput


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-q', '--quiet', default=False, action='store_true',
                    help='suppress messages')
    ap.add_argument('model', metavar='MODEL',
                    help='path to trained model')
    ap.add_argument('data', metavar='LANG:PATH',
                    help='data in language LANG (fastText format)')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    log = sys.stderr if not args.quiet else NullOutput()
    modelfn = '{}.h5'.format(args.model)
    metafn = '{}.json'.format(args.model)
    model = load_model(modelfn)
    print('Loaded model from {}'.format(modelfn), file=log)
    with open(metafn, encoding='utf-8') as f:
        meta = json.load(f)
    try:
        cased = meta['cased']
        max_length = meta['max_length']
        label_to_idx = meta['label_to_idx']
        token_to_idx = meta['token_to_idx']
    except KeyError:
        raise ValueError('incomplete metadata in {}'.format(metafn))
    print('Loaded metadata from {}'.format(metafn), file=log)

    lang, path = args.data.split(':')
    texts, labels = load_fasttext_data_multi([lang], [path], not cased, log)
    text_indices = [text_to_indices(t, token_to_idx, False) for t in texts]
    label_indices = [label_to_idx.get(l, None) for l in labels]
    if any(l is None for l in label_indices):
        warning('Removing {} texts with unknown labels'.format(
            len([l for l in label_indices if l is None])))
        filtered_t, filtered_l = [], []
        for t, l in zip(text_indices, label_indices):
            if l is not None:
                filtered_t.append(t)
                filtered_l.append(l)
        text_indices, label_indices = filtered_t, filtered_l
    idx_to_label = { v: k for k, v in label_to_idx.items() }
    print('Loaded {} texts, {} labels'.format(
        len(text_indices), len(set(label_indices))), file=log)

    test_X = pad_sequences(text_indices, maxlen=max_length,
                           padding='post', truncating='post')
    num_classes = len(label_to_idx)
    test_Y = to_categorical(label_indices, num_classes)

    pred_Y = model.predict(test_X, verbose=0)

    # TODO cut this
    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    print('accuracy:\t{:.4f}'.format(acc))

    acc = (np.argmax(pred_Y, axis=1) == np.argmax(test_Y, axis=1)).mean()
    print('accuracy:\t{:.4f}'.format(acc))

    aucs = []
    for c in range(num_classes):
        try:
            auc = roc_auc_score(test_Y[:, c], pred_Y[:, c])
            print('{} AUC:\t{:.4f}'.format(idx_to_label[c], auc))
            aucs.append(auc)
        except:
            print('{} AUC:\tN/A'.format(idx_to_label[c]))
    print('average AUC:\t{:.4f}'.format(np.mean(aucs)))

    return 0


if __name__ == '__main__':
    try:
        retval = main(sys.argv)
    except ValueError as e:
        error(e)
        retval = -1
    sys.exit(retval)
