
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


class FunctionPreprocessor(BasePreprocessor):
    """
    A preprocessor that uses a custom function for preprocessing.
    
    This class allows you to pass any function that takes an array/tensor
    and returns a processed array/tensor.
    """
    
    def __init__(self, target_size, func):
        """
        Initialize the FunctionPreprocessor.
        
        Args:
            target_size: Tuple of (width, height) for image resizing
            func: Function to apply during preprocessing. Should take an array/tensor
                  and return a processed array/tensor
        """
        super().__init__(target_size)
        self.func = func
    
    def preprocess(self, X):
        """
        Apply the custom function to the input data.
        
        Args:
            X: Input array/tensor
            
        Returns:
            Processed array/tensor
        """
        return self.func(X)

