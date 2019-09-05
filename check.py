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

import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from utils.extractor import Extractor
from utils.database import get_database_reader
from utils.geometry import geometry_matching
import argparse

MIN_INLINERS = 10

parser = argparse.ArgumentParser(description='Query an image against a database for near-duplicated detection.')
parser.add_argument('-i', '--input', help='image to check')
parser.add_argument('-d', '--hash-database', help='hash database file')
parser.add_argument('-s', '--search-index', help='search index file')
parser.add_argument('-l', '--image-list', help='a list (txt, csv) containing full path to images')
parser.add_argument('-v', '--verbose', action='store_true', default=False)


def neardup_detect(query, hasher, image_list, thresholds, search_index, verbose=True):
    """
    near duplication wrapper
    :param query: [string] path to query image
    :param hasher: [object] an object of the Extractor class defined in utils/extractor.py
    :param image_list: [list] list of paths to database images
    :param thresholds: [list] semantic thresholds of the database images
    :param search_index: [object] search index object
    :param verbose: [bool] print out work steps if True
    :return: [tuple] (neardup decision, closest_image_id, path_to_the_closest_image)
    """
    if verbose:
        print('Hashing query image ...')
    feat, th = hasher.extract2(query)
    feat = feat[None, ...]

    if verbose:
        print('Nearest neighbor search ...')
    dist, ids = search_index.kneighbors(feat)
    candidate_id = ids[0][0]
    dist = dist[0][0]
    candidate_path = image_list[candidate_id]

    if verbose:
        print('Near-duplication checking ...')

    if dist > thresholds[candidate_id]:  # semantic check
        msg = 'Semantic match: False. No near-duplicated image found.'
        duplicate_decision = False
    else:  # geometry check
        msg = 'Found a semantic match.'
        # we need to read the query and candidate images so will need physical paths to them
        query = np.array(Image.open(args.input).convert('L'))
        img_candidate = np.array(Image.open(candidate_path).convert('L'))
        inliners1, total1 = geometry_matching(query, img_candidate)
        inliners2, total2 = geometry_matching(query[:, ::-1], img_candidate)  # check flip version

        if inliners1 > MIN_INLINERS or inliners2 > MIN_INLINERS:
            msg += '\nGeometry matched. Duplicated found.'
            duplicate_decision = True
        else:
            msg += '\nGeometry not matched. No duplication.'
            duplicate_decision = False
        inliners, total = (inliners1, total1) if inliners1 >= inliners2 else (inliners2, total2)
        msg += '\n%d out of %d keypoints matched.' % (inliners, total)
    if verbose:
        print(msg)
    return duplicate_decision, candidate_id, candidate_path


if __name__ == '__main__':
    args = parser.parse_args()
    print('Loading hash database ...')
    db = get_database_reader(args.hash_database)
    ths = db.get_thresholds()

    print('Loading search index ...')
    with open(args.search_index, 'rb') as f:
        search = pickle.load(f)

    extract = Extractor()  # this object can be reused
    img_lst = pd.read_csv(args.image_list, header=None)[0].tolist()  # list of paths of database images

    dup_decision, nearest_img_id, nearest_img_path = neardup_detect(args.input, extract, img_lst, ths,
                                                                    search, args.verbose)
    print('Closest image id: #%d, path: %s' % (nearest_img_id, nearest_img_path))
    print('Final decision: Duplication detect? {}'.format(dup_decision))
