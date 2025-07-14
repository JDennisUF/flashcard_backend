import pytest
import requests
import json
import os
import time
from unittest.mock import patch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TestIntegration:
    """Integration tests that test the actual server running."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.base_url = "http://localhost:5000"
        self.timeout = 5
        
    def test_server_health_check(self):
        """Test that the server health check works."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            assert response.status_code == 200
            
            data = response.json()
            assert data['status'] == 'healthy'
            assert data['service'] == 'flashcard-backend'
            assert 'timestamp' in data
            
        except requests.ConnectionError:
            pytest.skip("Server not running - start with 'python app.py' to run integration tests")
    
    def test_generate_flashcards_with_real_api(self):
        """Test flashcard generation with real OpenAI API (if key is valid)."""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OpenAI API key found - skipping real API test")
            
        test_data = {
            "prompt": "Create 2 simple flashcards about Python variables",
            "model": "gpt-3.5-turbo",
            "max_tokens": 300,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=30  # OpenAI can take a while
            )
            
            # Check if we got a successful response
            if response.status_code == 200:
                data = response.json()
                assert data['success'] == True
                assert 'content' in data
                assert 'usage' in data
                assert data['model'] == 'gpt-3.5-turbo'
                
                # Verify the content is not empty
                assert len(data['content']) > 0
                
                # Verify usage stats are present
                usage = data['usage']
                assert 'prompt_tokens' in usage
                assert 'completion_tokens' in usage
                assert 'total_tokens' in usage
                assert usage['total_tokens'] > 0
                
            elif response.status_code == 401:
                pytest.skip("Invalid OpenAI API key - skipping real API test")
            else:
                pytest.fail(f"Unexpected status code: {response.status_code}, response: {response.text}")
                
        except requests.ConnectionError:
            pytest.skip("Server not running - start with 'python app.py' to run integration tests")
    
    def test_invalid_prompt_request(self):
        """Test error handling for invalid requests."""
        try:
            # Test missing prompt
            response = requests.post(
                f"{self.base_url}/generate",
                json={"invalid": "data"},
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            assert response.status_code == 400
            data = response.json()
            assert 'error' in data
            assert data['error'] == 'Prompt is required'
            
        except requests.ConnectionError:
            pytest.skip("Server not running - start with 'python app.py' to run integration tests")
    
    def test_prompt_too_long(self):
        """Test error handling for prompts that are too long."""
        try:
            test_data = {
                "prompt": "x" * 4001  # Exceeds 4000 character limit
            }
            
            response = requests.post(
                f"{self.base_url}/generate",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            assert response.status_code == 400
            data = response.json()
            assert 'error' in data
            assert 'too long' in data['error']
            
        except requests.ConnectionError:
            pytest.skip("Server not running - start with 'python app.py' to run integration tests")
    
    def test_non_json_request(self):
        """Test error handling for non-JSON requests."""
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                data="not json",
                headers={"Content-Type": "text/plain"},
                timeout=self.timeout
            )
            
            assert response.status_code == 400
            data = response.json()
            assert 'error' in data
            assert data['error'] == 'Request must be JSON'
            
        except requests.ConnectionError:
            pytest.skip("Server not running - start with 'python app.py' to run integration tests")
    
    def test_endpoint_not_found(self):
        """Test 404 error handling."""
        try:
            response = requests.get(f"{self.base_url}/nonexistent", timeout=self.timeout)
            assert response.status_code == 404
            
            data = response.json()
            assert data['error'] == 'Endpoint not found'
            
        except requests.ConnectionError:
            pytest.skip("Server not running - start with 'python app.py' to run integration tests")
    
    def test_method_not_allowed(self):
        """Test 405 error handling."""
        try:
            response = requests.post(f"{self.base_url}/health", timeout=self.timeout)
            assert response.status_code == 405
            
            data = response.json()
            assert data['error'] == 'Method not allowed'
            
        except requests.ConnectionError:
            pytest.skip("Server not running - start with 'python app.py' to run integration tests")

class TestWithMockedOpenAI:
    """Integration tests with mocked OpenAI responses."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.base_url = "http://localhost:5000"
        self.timeout = 5
    
    @patch('openai.ChatCompletion.create')
    def test_generate_with_mocked_response(self, mock_openai):
        """Test generation with mocked OpenAI response."""
        # Skip if server not running
        try:
            requests.get(f"{self.base_url}/health", timeout=1)
        except requests.ConnectionError:
            pytest.skip("Server not running - start with 'python app.py' to run integration tests")
        
        # Mock OpenAI response
        from unittest.mock import MagicMock
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Mock flashcard content"
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 150
        mock_response.usage.total_tokens = 175
        mock_openai.return_value = mock_response
        
        test_data = {
            "prompt": "Create flashcards about Python"
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert data['content'] == "Mock flashcard content"
