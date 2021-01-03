import pathlib
from setuptools import setup

from keras_image_helper import __version__

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="keras_image_helper",
    version=__version__,
    description="A lightweight library for pre-processing images for pre-trained keras models",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/alexeygrigorev/keras-image-helper",
    author="Alexey Grigorev",
    author_email="alexey@datatalks.club",
    license="WTFPL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["keras_image_helper"],
    include_package_data=True,
    install_requires=["numpy", "pillow"],
)