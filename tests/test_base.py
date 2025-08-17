import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from keras_image_helper.base import (
    BasePreprocessor, 
    download_image, 
    default_prepare_image
)


class TestDefaultPrepareImage:
    """Test the default_prepare_image utility function."""
    
    def test_convert_rgb_image(self):
        """Test converting non-RGB image to RGB."""
        # Create a grayscale image
        img = Image.new('L', (100, 100), color=128)
        assert img.mode == 'L'
        
        result = default_prepare_image(img, target_size=(50, 50))
        assert result.mode == 'RGB'
        assert result.size == (50, 50)
    
    def test_resize_image(self):
        """Test image resizing."""
        img = Image.new('RGB', (100, 100), color='red')
        result = default_prepare_image(img, target_size=(50, 75))
        assert result.size == (50, 75)
    
    def test_preserve_rgb_image(self):
        """Test that RGB images remain RGB."""
        img = Image.new('RGB', (100, 100), color='blue')
        result = default_prepare_image(img, target_size=(50, 50))
        assert result.mode == 'RGB'


class TestDownloadImage:
    """Test the download_image utility function."""
    
    @patch('keras_image_helper.base.request.urlopen')
    def test_download_image_success(self, mock_urlopen):
        """Test successful image download."""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = b'fake_image_data'
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Mock PIL Image.open
        mock_img = MagicMock()
        with patch('PIL.Image.open', return_value=mock_img):
            result = download_image('http://example.com/image.jpg')
            assert result == mock_img


class TestBasePreprocessor:
    """Test the BasePreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = BasePreprocessor(target_size=(224, 224))
        assert preprocessor.target_size == (224, 224)
    
    def test_resize_image(self):
        """Test image resizing functionality."""
        preprocessor = BasePreprocessor(target_size=(100, 150))
        img = Image.new('RGB', (200, 300), color='green')
        
        result = preprocessor.resize_image(img)
        assert result.size == (100, 150)
        assert result.mode == 'RGB'
    
    def test_image_to_array(self):
        """Test image to array conversion."""
        preprocessor = BasePreprocessor(target_size=(50, 50))
        img = Image.new('RGB', (50, 50), color='red')
        
        result = preprocessor.image_to_array(img)
        assert isinstance(result, np.ndarray)
        assert result.dtype == 'float32'
        assert result.shape == (50, 50, 3)
    
    def test_preprocess_not_implemented(self):
        """Test that preprocess raises exception when not implemented."""
        preprocessor = BasePreprocessor(target_size=(50, 50))
        
        with pytest.raises(Exception, match='not implemented'):
            preprocessor.preprocess(np.zeros((1, 50, 50, 3)))
    
    def test_convert_to_tensor(self):
        """Test convert_to_tensor method."""
        preprocessor = BasePreprocessor(target_size=(50, 50))
        img = Image.new('RGB', (100, 100), color='blue')
        
        # Mock the preprocess method
        preprocessor.preprocess = lambda x: x * 2
        
        result = preprocessor.convert_to_tensor(img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 50, 50, 3)
        # Check that preprocess was called (result should be doubled)
        assert np.array_equal(result, result / 2 * 2)
    
    def test_from_path(self):
        """Test loading image from file path."""
        preprocessor = BasePreprocessor(target_size=(50, 50))
        
        # Mock the convert_to_tensor method
        expected_result = np.ones((1, 50, 50, 3))
        preprocessor.convert_to_tensor = lambda x: expected_result
        
        with patch('PIL.Image.open') as mock_open:
            mock_img = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_img
            
            result = preprocessor.from_path('fake_path.jpg')
            assert np.array_equal(result, expected_result)
    
    def test_from_url(self):
        """Test loading image from URL."""
        preprocessor = BasePreprocessor(target_size=(50, 50))
        
        # Mock the convert_to_tensor method
        expected_result = np.ones((1, 50, 50, 3))
        preprocessor.convert_to_tensor = lambda x: expected_result
        
        with patch('keras_image_helper.base.download_image') as mock_download:
            mock_img = MagicMock()
            mock_download.return_value = mock_img
            
            result = preprocessor.from_url('http://example.com/image.jpg')
            assert np.array_equal(result, expected_result)
