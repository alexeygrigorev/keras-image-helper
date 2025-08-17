# Keras Image Helper

A lightweight library for pre-processing images for pre-trained keras models

Imagine you have a Keras model. To use it, you need to apply a certain pre-processing
function to all the images. Something like that:

```python
from tensorflow.keras.applications.xception import preprocess_input
```

What if you want to now deploy this model to AWS Lambda? Or deploy your model with TF-Serving? 
You don't want to use the entire TensorFlow package just for that.

The solution is simple - use `keras_image_helper`

## Usage

For an xception model:

```python
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(299, 299))

url = 'http://bit.ly/mlbookcamp-pants'
X = preprocessor.from_url(url)
```

It's also possible to provide custom functions
instead of preprocessor name:

```python
def custom_function(x):
    x = x / 255.0
    x = x.round(1)
    x.transpose(0, 3, 1, 2)
    return x

preprocessor = create_preprocessor(custom_function, target_size=(3, 3))
```

Now you can use `X` for your model:

```python
preds = model.predict(X)
```

That's all :tada:

For more examples, check [test.ipynb](test.ipynb)

Currently you can use the following pre-processors:

* `xception`
* `resnet50`
* `vgg16`
* `inception_v3`


If something you need is missing, PRs are welcome


## Installation 

It's available on PyPI, so you can install it with pip:

```bash
pip install keras_image_helper
```

You can also install the latest version from this repo:

```bash
git clone git@github.com:alexeygrigorev/keras-image-helper.git
python setup.py install
```


## Publishing

1. Install development dependencies:
```bash
uv sync --dev
```

2. Build the package:
```bash
uv run hatch build
```

3. Publish to test PyPI:
```bash
uv run hatch publish --repo test
```

4. Publish to PyPI:
```bash
uv run hatch publish
```

5. Clean up:
```bash
rm -r dist/
```

Note: For Hatch publishing, you'll need to configure your PyPI credentials in `~/.pypirc` or use environment variables.

## PyPI Credentials Setup

Create a `.pypirc` file in your home directory with your PyPI credentials:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-your-main-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

Done!