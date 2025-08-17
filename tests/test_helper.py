import pytest
import numpy as np

from keras_image_helper.helper import create_preprocessor
from keras_image_helper.preprocessors import (
    ResnetPreprocessor,
    XceptionPreprocessor,
    VGGPreprocessor,
    InceptionPreprocessor,
    FunctionPreprocessor
)


class TestCreatePreprocessor:
    """Test the create_preprocessor function."""
    
    def test_create_resnet_preprocessor(self):
        """Test creating ResNet preprocessor by name."""
        preprocessor = create_preprocessor('resnet50', target_size=(224, 224))
        
        assert isinstance(preprocessor, ResnetPreprocessor)
        assert preprocessor.target_size == (224, 224)
    
    def test_create_xception_preprocessor(self):
        """Test creating Xception preprocessor by name."""
        preprocessor = create_preprocessor('xception', target_size=(299, 299))
        
        assert isinstance(preprocessor, XceptionPreprocessor)
        assert preprocessor.target_size == (299, 299)
    
    def test_create_vgg_preprocessor(self):
        """Test creating VGG preprocessor by name."""
        preprocessor = create_preprocessor('vgg16', target_size=(224, 224))
        
        assert isinstance(preprocessor, VGGPreprocessor)
        assert preprocessor.target_size == (224, 224)
    
    def test_create_inception_preprocessor(self):
        """Test creating Inception preprocessor by name."""
        preprocessor = create_preprocessor('inception_v3', target_size=(299, 299))
        
        assert isinstance(preprocessor, InceptionPreprocessor)
        assert preprocessor.target_size == (299, 299)
    
    def test_create_preprocessor_case_insensitive(self):
        """Test that preprocessor names are case insensitive."""
        preprocessor1 = create_preprocessor('RESNET50', target_size=(224, 224))
        preprocessor2 = create_preprocessor('resnet50', target_size=(224, 224))
        
        assert isinstance(preprocessor1, ResnetPreprocessor)
        assert isinstance(preprocessor2, ResnetPreprocessor)
        assert preprocessor1.target_size == preprocessor2.target_size
    
    def test_create_preprocessor_with_unknown_name(self):
        """Test that unknown preprocessor names raise an exception."""
        with pytest.raises(Exception, match='Unknown model unknown_model'):
            create_preprocessor('unknown_model', target_size=(224, 224))
    
    def test_create_function_preprocessor_with_function(self):
        """Test creating FunctionPreprocessor with a callable function."""
        def test_func(x):
            return x * 2
        
        preprocessor = create_preprocessor(test_func, target_size=(224, 224))
        
        assert isinstance(preprocessor, FunctionPreprocessor)
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.func == test_func
    
    def test_create_function_preprocessor_with_lambda(self):
        """Test creating FunctionPreprocessor with a lambda function."""
        lambda_func = lambda x: x / 255.0
        
        preprocessor = create_preprocessor(lambda_func, target_size=(100, 100))
        
        assert isinstance(preprocessor, FunctionPreprocessor)
        assert preprocessor.target_size == (100, 100)
        assert preprocessor.func == lambda_func
    
    def test_create_function_preprocessor_with_builtin_function(self):
        """Test creating FunctionPreprocessor with a builtin function."""
        # Note: This is a contrived example, but tests the callable check
        preprocessor = create_preprocessor(abs, target_size=(50, 50))
        
        assert isinstance(preprocessor, FunctionPreprocessor)
        assert preprocessor.target_size == (50, 50)
        assert preprocessor.func == abs
    
    def test_function_preprocessor_works_correctly(self):
        """Test that the created FunctionPreprocessor works correctly."""
        def double_values(x):
            return x * 2
        
        preprocessor = create_preprocessor(double_values, target_size=(50, 50))
        
        # Test the preprocessing
        x = np.array([[[[1, 2, 3]]]], dtype='float32')
        result = preprocessor.preprocess(x)
        expected = np.array([[[[2, 4, 6]]]], dtype='float32')
        np.testing.assert_array_equal(result, expected)
    
    def test_mixed_usage_patterns(self):
        """Test mixing function and string usage patterns."""
        # Create a function preprocessor
        def normalize(x):
            return x / 255.0
        
        func_preprocessor = create_preprocessor(normalize, target_size=(100, 100))
        assert isinstance(func_preprocessor, FunctionPreprocessor)
        
        # Create a named preprocessor
        named_preprocessor = create_preprocessor('resnet50', target_size=(100, 100))
        assert isinstance(named_preprocessor, ResnetPreprocessor)
        
                # Both should work independently
        x = np.array([[[[255, 128, 64]]]], dtype='float32')

        func_result = func_preprocessor.preprocess(x)
        assert func_result[0, 0, 0, 0] == 1.0

        named_result = named_preprocessor.preprocess(x)
        # ResNet uses caffe preprocessing, so result will be different
        assert not np.array_equal(func_result, named_result)
    
    def test_function_preprocessor_inheritance(self):
        """Test that FunctionPreprocessor inherits all base functionality."""
        def test_func(x):
            return x
        
        preprocessor = create_preprocessor(test_func, target_size=(75, 125))
        
        # Test that it has all the expected methods
        assert hasattr(preprocessor, 'resize_image')
        assert hasattr(preprocessor, 'image_to_array')
        assert hasattr(preprocessor, 'convert_to_tensor')
        assert hasattr(preprocessor, 'from_path')
        assert hasattr(preprocessor, 'from_url')
        
        # Test that target_size is set correctly
        assert preprocessor.target_size == (75, 125)
