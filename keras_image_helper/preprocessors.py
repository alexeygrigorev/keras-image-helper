
from keras_image_helper.base import BasePreprocessor


def tf_preprocessing(x):
    x /= 127.5
    x -= 1.0
    return x


def caffe_preprocessing(x):
    # 'RGB'->'BGR'
    x = x[..., ::-1]

    mean = [103.939, 116.779, 123.68]

    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return x


class ResnetPreprocessor(BasePreprocessor):
    # sources:
    # 
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py
    #   preprocess_input: 
    #      imagenet_utils.preprocess_input(x, mode='tf', **kwargs)
    # 
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    #   _preprocess_numpy_input, mode == 'tf'
    # 

    def preprocess(self, X):
        return caffe_preprocessing(X)


class XceptionPreprocessor(BasePreprocessor):
    # sources:
    # 
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py
    #   preprocess_input: 
    #      imagenet_utils.preprocess_input(x, mode='tf', **kwargs)
    # 
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    #   _preprocess_numpy_input, mode == 'tf'
    # 

    def preprocess(self, x):
        return tf_preprocessing(x)


class VGGPreprocessor(BasePreprocessor):
    # sources:
    # 
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
    #   preprocess_input = imagenet_utils.preprocess_input
    #
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    #   _preprocess_numpy_input, mode == 'caffe'
    # 

    def preprocess(self, x):
        return caffe_preprocessing(x)


class InceptionPreprocessor(BasePreprocessor):
    # sources:
    # 
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py
    #    imagenet_utils.preprocess_input(x, mode='tf', **kwargs)
    #
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    #   _preprocess_numpy_input, mode == 'tf'
    # 

    def preprocess(self, x):
        return tf_preprocessing(x)