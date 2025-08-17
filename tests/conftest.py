"""
Shared test fixtures and configuration for keras-image-helper tests.
"""

import pytest
import numpy as np
from PIL import Image


@pytest.fixture(scope="session")
def sample_rgb_image():
    """Create a sample RGB test image for testing."""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture(scope="session")
def sample_grayscale_image():
    """Create a sample grayscale test image for testing."""
    return Image.new('L', (100, 100), color=128)


@pytest.fixture(scope="session")
def sample_large_image():
    """Create a larger test image for performance testing."""
    return Image.new('RGB', (512, 512), color='blue')


@pytest.fixture(scope="session")
def sample_test_data():
    """Create sample numpy test data for preprocessing functions."""
    return np.array([[[[100, 150, 200]]]], dtype='float32')


@pytest.fixture(scope="session")
def expected_tf_result():
    """Expected result for TensorFlow preprocessing."""
    # For input [100, 150, 200], TF preprocessing gives:
    # (100/127.5 - 1, 150/127.5 - 1, 200/127.5 - 1)
    # = (0.784 - 1, 1.176 - 1, 1.569 - 1) = (-0.216, 0.176, 0.569)
    return np.array([[[[-0.216, 0.176, 0.569]]]], dtype='float32')


@pytest.fixture(scope="session")
def expected_caffe_result():
    """Expected result for Caffe preprocessing."""
    # For input [100, 150, 200], Caffe preprocessing gives:
    # RGB -> BGR: [100, 150, 200] -> [200, 150, 100]
    # Mean subtraction: [200, 150, 100] - [103.939, 116.779, 123.68]
    # = [96.061, 33.221, -23.68]
    return np.array([[[[96.061, 33.221, -23.68]]]], dtype='float32')
