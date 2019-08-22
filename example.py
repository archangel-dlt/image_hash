#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14 Jun 2019 11:27
hash an image file or list of image files
Tu Bui tb0035@surrey.ac.uk
"""

import os
import numpy as np
from utils.extractor import Extractor
from utils.database import H5pyData

if __name__ == '__main__':
    hasher = Extractor()

    # hash a single image
    img_path = 'cat_neardup.png'
    hash, th = hasher.extract2(img_path)
    print('hash: {}\nNear-duplication threshold: {}'.format(hash, th))

    # hash batch of images and create a hash database
    img_lst = ['samples/cat.jpg', 'samples/airplane1.jpg', 'samples/airplane2.jpg']
    hashes, ths = hasher.extract_batch(img_lst)
    np.savez('hash_database.npz', feats=hashes, ths=ths)

    # Build search model
    # Note: it's better to execute this command directly in bash terminal rather calling from this script
    cmd = 'python build_searchtree.py -i hash_database.npz -o search_index.pkl'
    os.system(cmd)

    # Check if an image has near-duplicated copy in the database
    # Note: it's better to execute this command directly in bash terminal rather calling from this script
    cmd = 'python check.py -i cat_neardup.png -d hash_database.npz -s search_index.pkl'
    os.system(cmd)
