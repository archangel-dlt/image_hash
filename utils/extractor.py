#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14 Jun 2019 11:29
Tu Bui tb0035@surrey.ac.uk

"""
from tensorflow.keras import applications as apps
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np

# Architecture_name: [base_name, input_shape]
supported_architectures = {
    'VGG16': ['vgg16', (224, 224, 3)],
    'InceptionV3': ['inception_v3', (299, 299, 3)],
    'ResNet50': ['resnet50', (224, 224, 3)],
    'InceptionResNetV2': ['inception_resnet_v2', (299, 299, 3)],
    'NASNetMobile': ['nasnet', (224, 224, 3)],
    'DenseNet121': ['densenet', (224, 224, 3)],
    'Xception': ['xception', [299, 299, 3]]
}
augmentation = {
    'rotation': [-6, 6],  # rotation range
    'flip': True,
    'crop': 0.1  # fraction along each image dimension to be removed
}


class Extractor(object):
    def __init__(self, arch='ResNet50'):
        """
        image feature extraction class
        :param arch: deep architecture to be used e.g. ResNet
        :param verbose: print out some logging info
        """
        assert arch in supported_architectures, 'Error! %s not supported.' % arch
        tf_arch = getattr(apps, supported_architectures[arch][0])
        tf_base = getattr(tf_arch, arch)
        tf_model = tf_base(weights='imagenet', include_top=False, input_shape=supported_architectures[arch][1])
        top_layer_id = len(tf_model.layers) - 1
        while 'relu' in tf_model.layers[top_layer_id].name:
            top_layer_id -= 1
        output_layer = GlobalAveragePooling2D()(tf_model.layers[top_layer_id].output)
        self.model = Model(inputs=tf_model.input, outputs=output_layer)
        self.in_shape = self.model.input_shape[1:3]
        self.prefn = getattr(tf_arch, 'preprocess_input')
        # augmentation setting
        self.rot = np.linspace(augmentation['rotation'][0], augmentation['rotation'][1], 5)
        self.flip = [1, -1] if augmentation['flip'] else [1, ]
        self.img_shape = np.int64(np.array(self.in_shape, dtype=np.float32)*(1+augmentation['crop']))
        half_crop = np.int64((self.img_shape - np.array(self.in_shape))/2)
        self.crop_pos = [(0, 0),  # top left crop position
                         (0, half_crop[1]*2),
                         half_crop,
                         (half_crop[0]*2, 0),
                         half_crop*2]

    def __del__(self):
        del self.model
        K.clear_session()

    def extract(self, img_path):
        """
        legacy code to extract image descriptor
        use keras built-in image reader
        :param img_path: path to image
        :return: 1-D float32 descriptor
        """
        im = image.load_img(img_path, target_size=self.in_shape)
        im = image.img_to_array(im)[None, ...]  # 4-D numpy array
        im = self.prefn(im)
        feat = self.model.predict(im).squeeze()
        return feat

    def extract2(self, img_path, return_threshold=True):
        """
        extract image feat and near-duplication threshold
        :param img_path: path to image
        :param return_threshold: if True also return near-duplication threshold
        :return: feat (1-D float32), thres (scalar)
        """
        # TODO: augment then pass
        if return_threshold:
            im = Image.open(img_path).convert('RGB')
            ims = []
            for r in self.rot:
                im_r = im.rotate(r, Image.BILINEAR, expand=False).resize(self.img_shape[::-1], Image.BILINEAR)
                for c in self.crop_pos:
                    for f in self.flip:
                        im_ = np.array(im_r)[c[0]:c[0]+self.in_shape[0], c[1]:c[1]+self.in_shape[1], ::f]
                        ims.append(im_)
            ims = self.prefn(np.array(ims, dtype=np.float32))
            feats = self.model.predict(ims).squeeze()
            feat = feats.mean(axis=0, keepdims=True)
            th = np.sqrt(np.sum((feats - feat)**2, axis=1).max())
            return feat.squeeze(), th
        else:
            return self.extract(img_path)

    def extract_batch(self, img_lst):
        """
        extract features and threshold values for batch of images
        :param img_lst: list of paths to N images
        :return: feats (Nxdim float32), threshold values (N)
        """
        feats = []
        ths = []
        for path in img_lst:
            feat, th = self.extract2(path)
            feats.append(feat)
            ths.append(th)
        return np.array(feats), np.array(ths)
