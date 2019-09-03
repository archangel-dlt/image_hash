#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14 Jun 2019 11:27
hash an image file or list of image files
Tu Bui tb0035@surrey.ac.uk
"""

import os
import numpy as np
import pandas as pd
from utils.extractor import Extractor
from utils.database import H5pyData

if __name__ == '__main__':
    hasher = Extractor()

    """hash a single image"""
    img_path = 'cat_neardup.png'
    des, th = hasher.extract2(img_path)
    print('hash: {}\nNear-duplication threshold: {}'.format(des, th))

    """hash batch of images and create a hash database"""
    img_lst = ['samples/cat.jpg', 'samples/airplane1.jpg', 'samples/airplane2.jpg']
    hashes, ths = hasher.extract_batch(img_lst)
    # save as numpy database
    np.savez('hash_database.npz', feats=hashes, ths=ths)
    # alternatively, you can save as hdf5 database (efficient for large database and incremental saving)
    data = H5pyData('hash_database.h5', 'w')
    data.append(hashes, ths)
    # also save the image paths for geometry matching
    pd.DataFrame(img_lst).to_csv('image_list.txt', header=False, index=False)

    """Build search model"""
    # Note: it's better to execute this command directly in bash terminal rather calling from this script
    cmd = 'python build_searchtree.py -i hash_database.npz -o search_index.pkl'
    os.system(cmd)

    """Check if an image has near-duplicated copy in the database"""
    # Note: it's better to execute this command directly in bash terminal rather calling from this script
    cmd = 'python check.py -i cat_neardup.png -d hash_database.npz -s search_index.pkl -l image_list.txt -v'
    os.system(cmd)
