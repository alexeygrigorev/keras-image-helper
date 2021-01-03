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

Or with Pipenv:

```bash
pipenv install keras_image_helper
```

You can also install the latest version from this repo:

```bash
git clone git@github.com:alexeygrigorev/keras-image-helper.git
python setup.py install
```


## Publishing

Use twine for that:

```bash
pip install twine
```

Generate a wheel:

```python
python setup.py sdist bdist_wheel
```

Check the packages:

```bash
twine check dist/*
```

Upload the library to test PyPI to verify everything is working:

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Upload to PyPI:

```bash
twine upload dist/*
```

Done!