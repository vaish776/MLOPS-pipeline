import os
import sys
import pytest

# Make sure 'src' is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validate import validate_model

def test_model_file_exists():
    assert os.path.exists("src/model.pkl"), "Model file not found!"

def test_accuracy_threshold():
    validate_model()  # Will raise AssertionError if accuracy < BASELINE
