"""
AI Toolkit - Comprehensive Artificial Intelligence Development Suite
Author: ereezyy
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "ereezyy"
__email__ = "ereezyy@github.com"

from .data import DataProcessor, load_dataset
from .models import ModelBuilder, PretrainedModels
from .training import Trainer
from .evaluation import Evaluator
from .deployment import ModelDeployer
from .automl import AutoMLPipeline

# Core functions for quick access
def create_project(name, description=""):
    """Create a new AI project with organized structure."""
    from .utils.project import ProjectManager
    return ProjectManager.create_project(name, description)

def load_data(path, **kwargs):
    """Load data from various formats."""
    return load_dataset(path, **kwargs)

def train(model, data, **kwargs):
    """Train a model with the given data."""
    trainer = Trainer(model)
    return trainer.fit(data, **kwargs)

def evaluate(model, data, **kwargs):
    """Evaluate model performance."""
    evaluator = Evaluator()
    return evaluator.evaluate(model, data, **kwargs)

def deploy(model, platform="local", **kwargs):
    """Deploy model to specified platform."""
    deployer = ModelDeployer()
    return deployer.deploy(model, platform, **kwargs)

def predict(model, input_data, **kwargs):
    """Make predictions with a trained model."""
    return model.predict(input_data, **kwargs)

# Quick model creation functions
def create_image_classifier(num_classes, architecture="resnet50", **kwargs):
    """Create an image classification model."""
    builder = ModelBuilder()
    return builder.create_image_classifier(num_classes, architecture, **kwargs)

def create_text_classifier(num_classes, model_name="bert-base-uncased", **kwargs):
    """Create a text classification model."""
    builder = ModelBuilder()
    return builder.create_text_classifier(num_classes, model_name, **kwargs)

def create_time_series_model(sequence_length, features, **kwargs):
    """Create a time series forecasting model."""
    builder = ModelBuilder()
    return builder.create_time_series_model(sequence_length, features, **kwargs)

# Utility functions
def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_device_info():
    """Get information about available compute devices."""
    import tensorflow as tf
    
    devices = {
        'cpu_count': tf.config.experimental.list_physical_devices('CPU'),
        'gpu_count': len(tf.config.experimental.list_physical_devices('GPU')),
        'gpu_available': tf.test.is_gpu_available(),
        'tensorflow_version': tf.__version__
    }
    
    return devices

# Configuration
class Config:
    """Global configuration for AI Toolkit."""
    
    # Default paths
    MODEL_STORAGE_PATH = "./models"
    DATA_STORAGE_PATH = "./data"
    LOG_PATH = "./logs"
    
    # Training defaults
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 0.001
    
    # Evaluation defaults
    DEFAULT_METRICS = ['accuracy', 'precision', 'recall', 'f1']
    
    # Deployment defaults
    DEFAULT_PLATFORM = "local"
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)

# Initialize logging
import logging
import os

def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration."""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(Config.LOG_PATH):
        os.makedirs(Config.LOG_PATH)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        log_file = os.path.join(Config.LOG_PATH, log_file)
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=level, format=log_format)
    
    # Set up logger for the package
    logger = logging.getLogger('ai_toolkit')
    logger.setLevel(level)
    
    return logger

# Initialize default logging
logger = setup_logging()

# Welcome message
def print_welcome():
    """Print welcome message with system information."""
    print(f"""
    🤖 AI Toolkit v{__version__}
    ================================
    
    Welcome to the comprehensive AI development suite!
    
    Quick Start:
    - Create project: ai.create_project('my_project')
    - Load data: data = ai.load_data('path/to/data.csv')
    - Build model: model = ai.create_image_classifier(10)
    - Train: ai.train(model, data)
    
    System Information:
    - TensorFlow: Available
    - GPU Support: {get_device_info()['gpu_available']}
    - GPUs: {get_device_info()['gpu_count']}
    
    Documentation: https://github.com/ereezyy/ai
    Support: ereezyy@github.com
    """)

# Auto-print welcome message on import
if __name__ != "__main__":
    import os
    if os.getenv('AI_TOOLKIT_QUIET') != '1':
        print_welcome()

