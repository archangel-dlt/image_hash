#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry.py
Created on Sep 02 2019 14:58
Determine number of geometry matching points between two images.

ref1: homography
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
ref2: brute force search
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/

@author: Tu Bui tb0035@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from PIL import Image


def geometry_matching(im1, im2, debug=False):
    """
    match  two images using homography
    :param im1: image array 1
    :param im2: image array 2
    :param debug: if True, visualise the key points and homography
    :return: # inliners, # total match points
    """
    MIN_MATCH_COUNT = 10
    img1 = np.copy(im1)  # don't modify destructively
    img2 = np.copy(im2)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if len(kp1) == 0 or len(kp2) == 0:  # cannot verify
        return 0, 0

    # FLANN search
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=12,  # 6,  # 12
    #                     key_size=20,  # 12,  # 20
    #                     multi_probe_level=2)  # 1)  # 2
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)

    # Brute force
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m in matches:
        if len(m) == 2:
            if m[0].distance < 0.6 * m[1].distance:
                good.append(m[0])

    # RANSAC
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        matchesMask = mask.ravel().tolist()
        if debug:  # draw homography for the last run
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            out = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
            Image.fromarray(out).save('res_match.png')
        return sum(matchesMask), len(good)
    else:
        # visualise the features
        if debug:
            out1 = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
            out2 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
            images = [Image.fromarray(out1), Image.fromarray(out2)]
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]
            new_im.save('res_nomatch.png')
        return 0, len(good)


if __name__ == '__main__':
    path1 = '../cat_neardup.png'
    path2 = '../samples/cat.jpg'
    img1 = np.array(Image.open(path1).convert('L'))
    img2 = np.array(Image.open(path2).convert('L'))
    res1 = geometry_matching(img1, img2, True)
    print('inliners: {}, total: {}'.format(res1[0], res1[1]))
    # geometry doesn't work with flip so we must test the flip case manually
    res2 = geometry_matching(img1[:, ::-1], img2, True)
    print('flip detect: inliners: {}, total: {}'.format(res2[0], res2[1]))
    if res1[0] > 10 or res2[0] > 10:
        print('Near duplication detected.')
    else:
        print('Not duplicated.')
