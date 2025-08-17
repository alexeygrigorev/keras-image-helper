import pytest

import numpy as np
from PIL import Image

from keras_image_helper.helper import create_preprocessor
from keras_image_helper.preprocessors import FunctionPreprocessor


class TestIntegration:
    """Integration tests for actual image processing."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        return img
    
    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """Create a temporary image file for testing."""
        img = Image.new('RGB', (100, 100), color='blue')
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)
        return str(img_path)
    
    def test_function_preprocessor_with_real_image(self, sample_image):
        """Test FunctionPreprocessor with a real PIL image."""
        def normalize_and_center(x):
            x = x.astype('float32')
            x = x / 255.0  # Normalize to [0, 1]
            x = x - 0.5    # Center to [-0.5, 0.5]
            return x
        
        preprocessor = FunctionPreprocessor(target_size=(50, 50), func=normalize_and_center)
        
        # Process the image
        result = preprocessor.convert_to_tensor(sample_image)
        
        # Check the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 50, 50, 3)
        assert result.dtype == 'float32'
        
        # Check that normalization and centering worked
        # Red color should be normalized and centered
        # Red = [255, 0, 0] -> [1.0, 0.0, 0.0] -> [0.5, -0.5, -0.5]
        assert np.allclose(result[0, 0, 0], [0.5, -0.5, -0.5], atol=0.01)
    
    def test_function_preprocessor_from_path(self, sample_image_path):
        """Test FunctionPreprocessor loading image from file path."""
        def simple_normalize(x):
            return x.astype('float32') / 255.0
        
        preprocessor = FunctionPreprocessor(target_size=(75, 75), func=simple_normalize)
        
        # Load and process image from path
        result = preprocessor.from_path(sample_image_path)
        
        # Check the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 75, 75, 3)
        assert result.dtype == 'float32'
        
        # Check that normalization worked (values should be in [0, 1])
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_create_preprocessor_with_function_integration(self, sample_image):
        """Test create_preprocessor with function parameter."""
        def custom_preprocessing(x):
            # Complex preprocessing pipeline
            x = x.astype('float32')
            x = x / 255.0  # Normalize
            x = x - 0.5    # Center
            x = x * 2      # Scale to [-1, 1]
            return x
        
        # Create preprocessor using helper function
        preprocessor = create_preprocessor(custom_preprocessing, target_size=(60, 60))
        
        # Verify it's the right type
        assert isinstance(preprocessor, FunctionPreprocessor)
        assert preprocessor.func == custom_preprocessing
        
        # Test it works
        result = preprocessor.convert_to_tensor(sample_image)
        assert result.shape == (1, 60, 60, 3)
        assert result.dtype == 'float32'
        
        # Check that the custom preprocessing was applied
        # Red pixel should go: [255, 0, 0] -> [1.0, 0.0, 0.0] -> [0.5, -0.5, -0.5] -> [1.0, -1.0, -1.0]
        assert np.allclose(result[0, 0, 0], [1.0, -1.0, -1.0], atol=0.01)
    
    def test_mixed_preprocessor_types(self, sample_image):
        """Test mixing different types of preprocessors."""
        # Create a function preprocessor
        def func_preprocess(x):
            return x / 255.0
        
        func_preprocessor = create_preprocessor(func_preprocess, target_size=(50, 50))
        
        # Create a named preprocessor
        named_preprocessor = create_preprocessor('resnet50', target_size=(50, 50))
        
        # Process the same image with both
        func_result = func_preprocessor.convert_to_tensor(sample_image)
        named_result = named_preprocessor.convert_to_tensor(sample_image)
        
        # Both should produce valid results
        assert func_result.shape == (1, 50, 50, 3)
        assert named_result.shape == (1, 50, 50, 3)
        
        # Results should be different (different preprocessing)
        assert not np.array_equal(func_result, named_result)
    
    def test_function_preprocessor_edge_cases(self, sample_image):
        """Test FunctionPreprocessor with edge case functions."""
        # Identity function
        def identity(x):
            return x
        
        identity_preprocessor = FunctionPreprocessor(target_size=(50, 50), func=identity)
        result = identity_preprocessor.convert_to_tensor(sample_image)
        
        # Should preserve original values (after resizing and conversion)
        assert result.shape == (1, 50, 50, 3)
        assert result.dtype == 'float32'
        
        # Zero function
        def zero_func(x):
            return np.zeros_like(x)
        
        zero_preprocessor = FunctionPreprocessor(target_size=(50, 50), func=zero_func)
        result = zero_preprocessor.convert_to_tensor(sample_image)
        
        # Should return all zeros
        assert np.array_equal(result, np.zeros((1, 50, 50, 3)))

    def test_large_image_processing(self):
        """Test processing larger images (marked as slow)."""
        # Create a larger test image
        large_img = Image.new('RGB', (512, 512), color='green')
        
        def efficient_preprocessing(x):
            # Efficient preprocessing for large images
            return x.astype('float32') / 255.0
        
        preprocessor = FunctionPreprocessor(target_size=(256, 256), func=efficient_preprocessing)
        
        # Process the large image
        result = preprocessor.convert_to_tensor(large_img)
        
        assert result.shape == (1, 256, 256, 3)
        assert result.dtype == 'float32'
        assert np.all(result >= 0)
        assert np.all(result <= 1)
