#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_searchtree.py
Created on Aug 22 2019 12:13
Creat a search model for nearest neighbor search
@author: Tu Bui tb0035@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors as NN
from utils.database import get_database_reader

IN = '/vol/vssp/cvpnobackup/scratch_4weeks/tb0035/projects/archangel/tmp/hash_database.npz'
OUT = '/vol/vssp/cvpnobackup/scratch_4weeks/tb0035/projects/archangel/tmp/search_index.pkl'
num = 1  # number of nearest neighbors to be returned

parser = argparse.ArgumentParser(description='Input arguments.')
parser.add_argument('-i', '--input', default=IN, help='hash database')
parser.add_argument('-o', '--output', default=OUT, help='output search index')
parser.add_argument('-n', '--num', default=num, help='number of nearest neighbors', type=int)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Loading hash database from %s.' % args.input)
    db = get_database_reader(args.input)
    feats = db.get_hashes()
    nbrs = NN(args.num, algorithm='ball_tree').fit(feats)
    with open(args.output, 'wb') as f:
        pickle.dump(nbrs, f)
    print('Done. Search index saved at %s.' % args.output)
