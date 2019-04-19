import os
import logging
import tensorflow as tf
import time
import numpy as np
import cv2
import glob
from model.adanet_model import ImageClassificationAdaNet
from model.log_handler import QtHandler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BackendModel(object):
    Supported_Image_Extension = ["jpg", "gif", "png", "jpeg"]

    def __init__(self):
        logger = logging.getLogger('tensorflow')
        self.log_handler = QtHandler()
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.INFO)
        tf.logging.set_verbosity(3)
        self.default_testing_data = None
        self.dataset_loaded = False
        self.trained = False
        self._shape = None
        self._data = None
        self.config = {}
        self.num_classes = None
        self._adanet_model = None
        self.custom_test = None

    def train(self):
        if not self._data and self._shape:
            raise RuntimeError('Dataset not loaded')
        print(self.config)
        self._adanet_model = ImageClassificationAdaNet(self._data, num_classes=self.num_classes,
                                                       image_shape=self._shape, **self.config)
        results, _ = self._adanet_model.train_and_evaluate("uniform_average_ensemble_baseline")
        self.trained = True
        return results

    def _callback(self, result):
        self.result = result
        self.trained = True

    def predict(self):
        if not self.trained:
            raise RuntimeError
        prediction = self._adanet_model.predict()
        print('predict', self._adanet_model.x_test[:10].shape)
        return prediction, list(self._adanet_model.x_test[:10].copy())

    def load_dataset(self, dataset_name):
        if dataset_name == 'Fashion MNIST':
            self._data = tf.keras.datasets.fashion_mnist.load_data()
            self._shape = (28, 28, 1)
            self.num_classes = 10
        elif dataset_name == 'CIFAR-10':
            self._data = tf.keras.datasets.cifar10.load_data()
            self._shape = (32, 32, 3)
            self.num_classes = 10
        elif dataset_name == 'MNIST':
            self._data = tf.keras.datasets.mnist.load_data()
            self._shape = (28, 28, 1)
            self.num_classes = 10
        else:
            self.dataset_loaded = False
            return False

        self.default_testing_data = self._data[1][0]
        self.dataset_loaded = True
        self.trained = False
        return True

    def set_testing_data(self, dirname):
        image_list = [cv2.imread(item) for i in
                      [glob.glob(dirname + '/*.%s' % ext) for ext in self.Supported_Image_Extension] for item in i]
        if self._shape[-1] == 1:
            image_list = [cv2.threshold(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), 0, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] for img_rgb in image_list]
        image_list = [cv2.resize(image, self._shape[:2]) for image in image_list]
        self.custom_test = np.array(image_list, dtype=np.uint8)
        self._adanet_model.x_test = self.custom_test
        return True

    def use_custom_test(self):
        if not self.default_testing_data:
            raise RuntimeError('No Default Testing Data')
        self._adanet_model.x_test = self.custom_test

    def use_default_test(self):
        if not self.default_testing_data:
            raise RuntimeError('No Default Testing Data')
        self._adanet_model.x_test = self.default_testing_data
