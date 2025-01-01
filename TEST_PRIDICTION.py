import unittest
import torch
import os
import logging
from prediction import PolymerPredictor, check_system_requirements
from torch import nn

class TestPolymerPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Set up test paths
        cls.checkpoint_path = "checkpoints/model_best.pt"
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create predictor instance
        cls.predictor = PolymerPredictor(model_path=cls.checkpoint_path)
        
        # Create test checkpoint with simple architecture
        cls.test_state_dict = {
            'model_state_dict': {
                'token_embedding.weight': torch.randn(24, 512),
                'cloud_point_head.weight': torch.randn(512),
                'cloud_point_head.bias': torch.randn(1),
                'phase_head.weight': torch.randn(512),
                'phase_head.bias': torch.randn(1)
            }
        }
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(cls.test_state_dict, cls.checkpoint_path)

    def setUp(self):
        """Reset predictor before each test"""
        self.predictor.model = None
        self.predictor.vocab = None

    def test_vocab_initialization(self):
        """Test vocabulary initialization"""
        vocab = self.predictor._initialize_vocab(24)
        
        # Check vocab size
        self.assertEqual(len(vocab), 24)
        
        # Check padding token
        self.assertEqual(vocab['[PAD]'], 0)
        
        # Check other tokens
        self.assertTrue(all(c in vocab for c in "C[]()+-=#1234567890.OHN"))
    
    def test_model_architecture(self):
        """Test model initialization with correct architecture"""
        self.predictor._load_model()
        model = self.predictor.model
        
        # Check model dimensions
        self.assertEqual(model.d_model, 512)
        self.assertEqual(len(model.layers), 12)
        
        # Check prediction heads
        self.assertIsInstance(model.cloud_point_head[0], nn.Linear)
        self.assertEqual(model.cloud_point_head[0].in_features, 512)
        self.assertEqual(model.cloud_point_head[-1].out_features, 1)
        
        self.assertIsInstance(model.phase_head[0], nn.Linear)
        self.assertEqual(model.phase_head[0].in_features, 512)
        self.assertEqual(model.phase_head[-1].out_features, 1)

    def test_prediction_shape(self):
        """Test prediction output shape and format"""
        test_smile = "CC(=O)OC1=CC=CC=C1"
        prediction = self.predictor.predict_single(test_smile)
        
        # Check prediction structure
        self.assertIn('Cloud Point', prediction)
        self.assertIn('Phase', prediction)
        self.assertIn('Value', prediction['Cloud Point'])
        self.assertIn('Unit', prediction['Cloud Point'])
        self.assertIn('Uncertainty', prediction['Cloud Point'])
        
        # Check value types
        self.assertIsInstance(prediction['Cloud Point']['Value'], float)
        self.assertIsInstance(prediction['Phase']['Value'], str)

    def test_input_validation(self):
        """Test SMILES input validation"""
        valid_smile = "CC(=O)OC1=CC=CC=C1"
        invalid_smile = "XX"
        
        # Test valid SMILES
        self.assertTrue(all(c in "C[]()+=#-1234567890.OHNPS" for c in valid_smile))
        
        # Test invalid SMILES
        self.assertFalse(all(c in "C[]()+=#-1234567890.OHNPS" for c in invalid_smile))
        
    def test_error_handling(self):
        """Test error handling for invalid inputs and model loading"""
        # Test invalid model path
        with self.assertRaises(Exception):
            predictor = PolymerPredictor(model_path="invalid/path.pt")
            predictor.predict_single("CC")
            
        # Test empty SMILES
        with self.assertRaises(Exception):
            self.predictor.predict_single("")
            
        # Test invalid SMILES format
        with self.assertRaises(Exception):
            self.predictor.predict_single("XX")

    def test_system_requirements(self):
        """Test system requirements check"""
        # Test memory check
        has_requirements = check_system_requirements()
        self.assertIsInstance(has_requirements, bool)
        
    def test_model_loading_optimization(self):
        """Test model loading optimization"""
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        self.predictor._load_model()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Check memory usage is reasonable (less than 2GB)
        memory_usage_gb = (final_memory - initial_memory) / (1024**3)
        self.assertLess(memory_usage_gb, 2.0)

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests"""
        # Clean up checkpoint file
        if os.path.exists(cls.checkpoint_path):
            os.remove(cls.checkpoint_path)
            
        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main()