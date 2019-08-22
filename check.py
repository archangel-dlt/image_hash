#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check.py
Created on Aug 22 2019 13:58
check if an input image is already in the database
also report the most similar image in the database
@author: Tu Bui tb0035@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

IN = 'cat.jpg'
DATABASE = '/vol/vssp/cvpnobackup/scratch_4weeks/tb0035/projects/archangel/tmp/hash_database.npz'
SEARCH_INDEX = '/vol/vssp/cvpnobackup/scratch_4weeks/tb0035/projects/archangel/tmp/search_index.pkl'

import argparse
parser = argparse.ArgumentParser(description='Input arguments.')
parser.add_argument('-i', '--input', default=IN, help='image to check')
parser.add_argument('-d', '--database', default=DATABASE, help='hash database file')
parser.add_argument('-s', '--search-index', default=SEARCH_INDEX, help='search index file')

import pickle
from utils.extractor import Extractor
from utils.database import NumpyData as DatabaseReader
# from utils.database import H5pyData as DatabaseReader

if __name__ == '__main__':
    args = parser.parse_args()
    print('Loading database ...')
    db = DatabaseReader(args.database)
    ths = db.get_thresholds()

    print('Loading search index ...')
    with open(args.search_index, 'rb') as f:
        search = pickle.load(f)

    print('Hashing query image ...')
    hasher = Extractor()
    feat, th = hasher.extract2(args.input)
    feat = feat[None, ...]

    print('Checking ...')
    dist, ids = search.kneighbors(feat)
    ids = ids.squeeze()
    dist = dist.squeeze()
    if dist > ths[ids]:
        print('No near-duplicated image found.')
    else:
        print('Found near-duplicated image with id %d in the database' % ids)
        print('Threshold %f. Distance to this image: %f' % (ths[ids], dist))
