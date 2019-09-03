#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
database.py
Created on Aug 22 2019 14:22
class for reading/writing hash database
The hash database can be of any format but must store the hashes of the images and the corresponding
near-duplication thresholds

Each class must have at least 3 following methods:
get_hashes() : return all hashes stored in the database (for reading)
get_thresholds(): return all threshold values associated with the hashes (for reading)
append(hashes, thresholds): append or create if database not exist data (for writing)

@author: Tu Bui tb0035@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import h5py as h5

supported_database_extensions = ['h5', 'hdf5', 'npz']


def get_database_reader(data_path):
    """
    automatically determine database type based on file extension
    Note: please update this method if you create a new database class
    :param data_path: path to the database
    :return: a database object
    """
    if data_path.endswith('.npz'):
        print('Numpy database detected.')
        return NumpyData(data_path, 'r')
    elif data_path.endswith('.h5') or data_path.endswith('.hdf5'):
        print('HDF5 database detected.')
        return H5pyData(data_path, 'r')
    else:
        raise TypeError("Error! Database must have extension %s" % supported_database_extensions)


class NumpyData(object):
    """
    database reader/writer using numpy
    Read usage:
    npdata = NumpyData('my_database.npz', 'r')
    feats = npdata.get_hashes()
    thresholds = npdata.get_thresholds()

    Write usage:
    npdata = NumpyData('my_database.npz', 'w')
    npdata.append(feats1, ths1)  # feats1: NxD, ths1: N,
    npdata.append(feats2, ths2)  # feats2: MxD, ths2: M,
    """
    def __init__(self, data_path, mode='r'):
        """
        initializer
        :param data_path: path to this database
        :param mode: 'r' for read mode, 'w' for write mode
        """
        assert mode in ['r', 'w'], "Error! Mode can only be 'r' or 'w'."
        self.mode = mode
        self.data_path = data_path
        if self.mode == 'r':
            self.data = np.load(self.data_path)

    def get_hashes(self):
        """
        :return: all hash descriptors
        """
        assert self.mode == 'r', "Error! Function get_hashes() can only be used in read mode."
        return self.data['feats']

    def get_thresholds(self):
        """
        :return: array of corresponding thresholds (each image has a threshold)
        """
        assert self.mode == 'r', "Error! Function get_thresholds() can only be used in read mode."
        return self.data['ths']

    def append(self, feats, ths):
        """
        append data to database; create a new database if not exist.
        :param feats: hashes NxD
        :param ths: threshold values N
        :return: 0
        """
        ths = ths.squeeze()
        feats = feats + np.zeros((1, 1))  # make sure feats have shape NxD
        assert self.mode == 'w', "Error! Function append() can only be used in write mode."
        if os.path.isfile(self.data_path):  # append to existing database
            data = np.load(self.data_path)
            feats_new = np.r_[data['feats'], feats]
            ths_new = np.r_[data['ths'], ths]
            np.savez(self.data_path, feats=feats_new, ths=ths_new)
        else:  # database not exist, create a new one
            np.savez(self.data_path, feats=feats, ths=ths)


class H5pyData(object):
    """
    database reader/writer using h5py

    Read usage:
    npdata = NumpyData('my_database.h5', 'r')
    feats = npdata.get_hashes()
    thresholds = npdata.get_thresholds()

    Write usage:
    npdata = NumpyData('my_database.h5', 'w')
    npdata.append(feats1, ths1)  # feats1: NxD, ths1: N,
    npdata.append(feats2, ths2)  # feats2: MxD, ths2: M,
    """
    def __init__(self, data_path, mode='r'):
        """
        initializer
        :param data_path: path to the database to be read/created.
        :param mode: 'r' for read mode, 'w' for write mode
        """
        assert mode in ['r', 'w'], "Error! Mode can only be 'r' or 'w'."
        self.mode = mode
        self.data_path = data_path
        if self.mode == 'r':
            self.data = h5.File(data_path, 'r')

    def __del__(self):
        if self.mode == 'r':
            try:
                self.data.close()
            except Exception as e:
                pass

    def get_hashes(self):
        """
        :return: all hash descriptors
        """
        assert self.mode == 'r', "Error! Function get_hashes() can only be used in read mode."
        return self.data['feats'][...]

    def get_thresholds(self):
        """
        :return: array of corresponding thresholds (each image has a threshold)
        """
        assert self.mode == 'r', "Error! Function get_thresholds() can only be used in read mode."
        return self.data['ths'][...]

    def append(self, feats, ths):
        """
        append data to database; create a new database if not exist.
        :param feats: hashes NxD
        :param ths: threshold values N
        :return: 0
        """
        ths = ths.squeeze()
        feats = feats + np.zeros((1, 1))  # make sure feats have shape NxD
        assert self.mode == 'w', "Error! Function append() can only be used in write mode."
        n, dim = feats.shape
        if os.path.isfile(self.data_path):  # append to existing database
            with h5.File(self.data_path, 'a') as f:
                feat_data = f['feats']
                n0 = feat_data.shape[0]
                feat_data.resize((n+n0, feat_data.shape[1]))
                feat_data[-n:] = feats
                th_data = f['ths']
                th_data.resize((n+n0, ))
                th_data[-n:] = ths
                f.flush()
        else:  # database not exist, create a new one
            with h5.File(self.data_path, 'w') as f:
                f.create_dataset('feats', data=feats,
                                 shape=(n, dim),
                                 maxshape=(None, dim),
                                 dtype=np.float32,
                                 compression='gzip')

                f.create_dataset('ths', data=ths,
                                 shape=(n,),
                                 maxshape=(None,),
                                 dtype=np.float32,
                                 compression='gzip')
