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
        self.img_shape = self.model.input_shape[1:3]
        self.prefn = getattr(tf_arch, 'preprocess_input')

    def extract(self, img_path):
        im = image.load_img(img_path, target_size=self.img_shape)
        im = image.img_to_array(im)[None, ...]  # 4-D numpy array
        im = self.prefn(im)
        feat = self.model.predict(im).squeeze()
        return feat