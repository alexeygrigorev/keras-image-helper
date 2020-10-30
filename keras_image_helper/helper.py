from keras_image_helper.preprocessors import ResnetPreprocessor
from keras_image_helper.preprocessors import XceptionPreprocessor
from keras_image_helper.preprocessors import VGGPreprocessor
from keras_image_helper.preprocessors import InceptionPreprocessor


# reference: https://keras.io/api/applications/

preprocessors = {
    'xception': XceptionPreprocessor,
    'resnet50': ResnetPreprocessor,
    'vgg16': VGGPreprocessor,
    'inception_v3': InceptionPreprocessor,
}


def create_preprocessor(name, target_size, **params):
    name = name.lower()
    if name not in preprocessors:
        raise Exception('Unknown model %s' % name)
    Preprocessor = preprocessors[name]
    return Preprocessor(target_size=target_size, **params)