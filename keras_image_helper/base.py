
from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def default_prepare_image(img, target_size=(224, 224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


class BasePreprocessor:
    def __init__(self, target_size):
        self.target_size = target_size
    
    def resize_image(self, img):
        return default_prepare_image(img, self.target_size)

    def image_to_array(self, img):
        return np.array(img, dtype='float32')

    def preprocess(self, X):
        raise Exception('not implemented')

    def convert_to_tensor(self, img):
        small = self.resize_image(img)
        x = self.image_to_array(small)
        batch = np.expand_dims(x, axis=0)
        return self.preprocess(batch)

    def from_path(self, path):
        with Image.open(path) as img:
            return self.convert_to_tensor(img)

    def from_url(self, url):
        img = download_image(url)
        return self.convert_to_tensor(img)
    

