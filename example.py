#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14 Jun 2019 11:27
Tu Bui tb0035@surrey.ac.uk

"""

from utils.extractor import Extractor

if __name__ == '__main__':
    hasher = Extractor()

    img_path = 'cat.jpg'
    hash = hasher.extract(img_path)
    print(hash.shape)
    # the hasher object can be re-used to extract other images