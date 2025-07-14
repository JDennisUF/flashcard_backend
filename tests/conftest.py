import pytest
import os
from unittest.mock import patch
from app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_env():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-api-key',
        'FLASK_ENV': 'testing',
        'PORT': '5000'
    }):
        yield
