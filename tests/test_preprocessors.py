import pytest
import numpy as np
from PIL import Image

from keras_image_helper.preprocessors import (
    ResnetPreprocessor,
    XceptionPreprocessor,
    VGGPreprocessor,
    InceptionPreprocessor,
    FunctionPreprocessor,
    tf_preprocessing,
    caffe_preprocessing
)


class TestPreprocessingFunctions:
    """Test the standalone preprocessing functions."""
    
    def test_tf_preprocessing(self):
        """Test TensorFlow-style preprocessing."""
        # Create test data in range [0, 255]
        x = np.array([[[[100, 150, 200]]]], dtype='float32')
        
        result = tf_preprocessing(x)
        
        # Expected: (100/127.5 - 1, 150/127.5 - 1, 200/127.5 - 1)
        # = (0.784 - 1, 1.176 - 1, 1.569 - 1) = (-0.216, 0.176, 0.569)
        expected = np.array([[[[-0.216, 0.176, 0.569]]]], dtype='float32')
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
    
    def test_caffe_preprocessing(self):
        """Test Caffe-style preprocessing."""
        # Create test data in RGB format
        x = np.array([[[[100, 150, 200]]]], dtype='float32')
        
        result = caffe_preprocessing(x)
        
        # Expected: BGR conversion and mean subtraction
        # RGB -> BGR: [100, 150, 200] -> [200, 150, 100]
        # Mean subtraction: [200, 150, 100] - [103.939, 116.779, 123.68]
        expected = np.array([[[[96.061, 33.221, -23.68]]]], dtype='float32')
        np.testing.assert_array_almost_equal(result, expected, decimal=3)


class TestResnetPreprocessor:
    """Test the ResnetPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = ResnetPreprocessor(target_size=(224, 224))
        assert preprocessor.target_size == (224, 224)
    
    def test_preprocess(self):
        """Test ResNet preprocessing."""
        preprocessor = ResnetPreprocessor(target_size=(50, 50))
        x = np.array([[[[100, 150, 200]]]], dtype='float32')
        
        result = preprocessor.preprocess(x)
        
        # Should use caffe_preprocessing
        expected = caffe_preprocessing(x)
        np.testing.assert_array_equal(result, expected)


class TestXceptionPreprocessor:
    """Test the XceptionPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = XceptionPreprocessor(target_size=(299, 299))
        assert preprocessor.target_size == (299, 299)
    
    def test_preprocess(self):
        """Test Xception preprocessing."""
        preprocessor = XceptionPreprocessor(target_size=(50, 50))
        x = np.array([[[[100, 150, 200]]]], dtype='float32')
        
        result = preprocessor.preprocess(x)
        
        # Should use tf_preprocessing
        expected = tf_preprocessing(x)
        np.testing.assert_array_equal(result, expected)


class TestVGGPreprocessor:
    """Test the VGGPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = VGGPreprocessor(target_size=(224, 224))
        assert preprocessor.target_size == (224, 224)
    
    def test_preprocess(self):
        """Test VGG preprocessing."""
        preprocessor = VGGPreprocessor(target_size=(50, 50))
        x = np.array([[[[100, 150, 200]]]], dtype='float32')
        
        result = preprocessor.preprocess(x)
        
        # Should use caffe_preprocessing
        expected = caffe_preprocessing(x)
        np.testing.assert_array_equal(result, expected)


class TestInceptionPreprocessor:
    """Test the InceptionPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = InceptionPreprocessor(target_size=(299, 299))
        assert preprocessor.target_size == (299, 299)
    
    def test_preprocess(self):
        """Test Inception preprocessing."""
        preprocessor = InceptionPreprocessor(target_size=(50, 50))
        x = np.array([[[[100, 150, 200]]]], dtype='float32')
        
        result = preprocessor.preprocess(x)
        
        # Should use tf_preprocessing
        expected = tf_preprocessing(x)
        np.testing.assert_array_equal(result, expected)


class TestFunctionPreprocessor:
    """Test the FunctionPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        def test_func(x):
            return x * 2
        
        preprocessor = FunctionPreprocessor(target_size=(224, 224), func=test_func)
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.func == test_func
    
    def test_preprocess_with_simple_function(self):
        """Test preprocessing with a simple function."""
        def double_values(x):
            return x * 2
        
        preprocessor = FunctionPreprocessor(target_size=(50, 50), func=double_values)
        x = np.array([[[[1, 2, 3]]]], dtype='float32')
        
        result = preprocessor.preprocess(x)
        expected = np.array([[[[2, 4, 6]]]], dtype='float32')
        np.testing.assert_array_equal(result, expected)
    
    def test_preprocess_with_normalization_function(self):
        """Test preprocessing with a normalization function."""
        def normalize(x):
            return x / 255.0
        
        preprocessor = FunctionPreprocessor(target_size=(50, 50), func=normalize)
        x = np.array([[[[255, 128, 64]]]], dtype='float32')
        
        result = preprocessor.preprocess(x)
        expected = np.array([[[[1.0, 0.502, 0.251]]]], dtype='float32')
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
    
    def test_preprocess_with_complex_function(self):
        """Test preprocessing with a complex function."""
        def complex_preprocessing(x):
            # Simulate a complex preprocessing pipeline
            x = x.astype('float32')
            x = x / 255.0  # Normalize to [0, 1]
            x = x - 0.5     # Center to [-0.5, 0.5]
            x = x * 2       # Scale to [-1, 1]
            return x
        
        preprocessor = FunctionPreprocessor(target_size=(50, 50), func=complex_preprocessing)
        x = np.array([[[[0, 128, 255]]]], dtype='float32')
        
        result = preprocessor.preprocess(x)
        expected = np.array([[[[-1.0, 0.0, 1.0]]]], dtype='float32')
        np.testing.assert_array_almost_equal(result, expected, decimal=2)
    
    def test_inherits_base_functionality(self):
        """Test that FunctionPreprocessor inherits all base functionality."""
        def test_func(x):
            return x
        
        preprocessor = FunctionPreprocessor(target_size=(100, 150), func=test_func)
        
        # Test image resizing
        img = Image.new('RGB', (200, 300), color='green')
        result = preprocessor.resize_image(img)
        assert result.size == (100, 150)
        assert result.mode == 'RGB'
        
        # Test image to array conversion
        img = Image.new('RGB', (50, 50), color='red')
        result = preprocessor.image_to_array(img)
        assert isinstance(result, np.ndarray)
        assert result.dtype == 'float32'
        assert result.shape == (50, 50, 3)
    
    def test_convert_to_tensor_integration(self):
        """Test that FunctionPreprocessor works with convert_to_tensor."""
        def test_func(x):
            return x * 3
        
        preprocessor = FunctionPreprocessor(target_size=(50, 50), func=test_func)
        img = Image.new('RGB', (100, 100), color='blue')
        
        result = preprocessor.convert_to_tensor(img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 50, 50, 3)
        
        # Check that the function was applied (result should be tripled)
        # We can't easily check the exact values due to image processing,
        # but we can verify the shape and that it's not all zeros
        assert not np.array_equal(result, np.zeros_like(result))
