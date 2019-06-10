#!/usr/bin/env python3

# Summarize labeled values in TSV data.

import sys
import os

import numpy as np

from collections import defaultdict


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('file', nargs='+', help='two-column TSV (label, value)')
    return ap


def process(path):
    grouped = defaultdict(list)
    with open(path) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            if len(fields) != 2:
                raise ValueError(
                    'require 2 TSV fields, got {} on line {} in {}: {}'.\
                    format(len(fields), ln, path, l))
            label, value = fields
            try:
                float(value)
            except:
                raise ValueError(
                    'non-numeric value on line {} in {}: {}'.\
                    format(ln, path, l))
            grouped[fields[0]].append(value)
    name = os.path.splitext(os.path.basename(path))[0]
    # print('file\tavg\tstdev\tvalue\tcount\tvalue ...')
    for label, values in sorted(grouped.items()):
        floats = [float(v) for v in values]
        print('{}\t{}\t{:.4f}\t{:.4f}\t{}\t{}'.format(
            name, label, np.mean(floats), np.std(floats), len(floats),
            '\t'.join(values)))
    return grouped
    

def main(argv):
    args = argparser().parse_args(argv[1:])
    for path in args.file:
        process(path)
    return 0


if __name__ == '__main__':
    main(sys.argv)
