import pytest
import json
import os
from unittest.mock import patch, MagicMock
from app import app

class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_endpoint_returns_200(self, client):
        """Test that health endpoint returns 200 status."""
        response = client.get('/health')
        assert response.status_code == 200
        
    def test_health_endpoint_returns_json(self, client):
        """Test that health endpoint returns proper JSON structure."""
        response = client.get('/health')
        data = response.get_json()
        
        assert 'status' in data
        assert 'timestamp' in data
        assert 'service' in data
        assert data['status'] == 'healthy'
        assert data['service'] == 'flashcard-backend'

class TestGenerateEndpoint:
    """Test the /generate endpoint."""
    
    @patch('app.openai_client.chat.completions.create')
    def test_generate_endpoint_success(self, mock_openai, client):
        """Test successful flashcard generation."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test flashcard content"
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 150
        mock_response.usage.total_tokens = 175
        mock_openai.return_value = mock_response
        
        # Test request
        test_data = {
            "prompt": "Create flashcards about Python"
        }
        
        response = client.post('/generate', 
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] == True
        assert data['content'] == "Test flashcard content"
        assert data['model'] == 'gpt-3.5-turbo'
        assert 'usage' in data
        
    def test_generate_endpoint_missing_prompt(self, client):
        """Test error when prompt is missing."""
        test_data = {}
        
        response = client.post('/generate',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert data['success'] == False
        assert data['error'] == 'Prompt is required'
        
    def test_generate_endpoint_non_json_request(self, client):
        """Test error when request is not JSON."""
        response = client.post('/generate',
                             data="not json",
                             content_type='text/plain')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert data['error'] == 'Request must be JSON'
        
    def test_generate_endpoint_prompt_too_long(self, client):
        """Test error when prompt is too long."""
        test_data = {
            "prompt": "x" * 4001  # Exceeds 4000 character limit
        }
        
        response = client.post('/generate',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert data['error'] == 'Prompt too long (max 4000 characters)'
        
    @patch('app.openai_client.chat.completions.create')
    def test_generate_endpoint_with_custom_params(self, mock_openai, client):
        """Test generation with custom parameters."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Custom flashcard content"
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 200
        mock_response.usage.total_tokens = 230
        mock_openai.return_value = mock_response
        
        # Test request with custom parameters
        test_data = {
            "prompt": "Create flashcards about Python",
            "model": "gpt-4",
            "max_tokens": 1500,
            "temperature": 0.5
        }
        
        response = client.post('/generate',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] == True
        assert data['model'] == 'gpt-4'
        
        # Verify OpenAI was called with correct parameters
        mock_openai.assert_called_once()
        call_args = mock_openai.call_args
        assert call_args[1]['model'] == 'gpt-4'
        assert call_args[1]['max_tokens'] == 1500
        assert call_args[1]['temperature'] == 0.5

class TestOpenAIErrorHandling:
    """Test OpenAI error handling."""
    
    @patch('app.openai_client.chat.completions.create')
    def test_openai_rate_limit_error(self, mock_openai, client):
        """Test handling of OpenAI rate limit error."""
        # Create a mock response object
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        
        from openai import RateLimitError
        mock_openai.side_effect = RateLimitError("Rate limit exceeded", response=mock_response, body=None)
        
        test_data = {"prompt": "Test prompt"}
        
        response = client.post('/generate',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        assert response.status_code == 429
        data = response.get_json()
        assert 'error' in data
        assert 'rate limit' in data['error'].lower()
        
    @patch('app.openai_client.chat.completions.create')
    def test_openai_authentication_error(self, mock_openai, client):
        """Test handling of OpenAI authentication error."""
        # Create a mock response object
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        
        from openai import AuthenticationError
        mock_openai.side_effect = AuthenticationError("Invalid API key", response=mock_response, body=None)
        
        test_data = {"prompt": "Test prompt"}
        
        response = client.post('/generate',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        assert response.status_code == 401
        data = response.get_json()
        assert 'error' in data
        assert data['error'] == 'Authentication failed'
        
    @patch('app.openai_client.chat.completions.create')
    def test_openai_invalid_request_error(self, mock_openai, client):
        """Test handling of OpenAI invalid request error."""
        # Create a mock response object
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        
        from openai import BadRequestError
        mock_openai.side_effect = BadRequestError("Invalid request", response=mock_response, body=None)
        
        test_data = {"prompt": "Test prompt"}
        
        response = client.post('/generate',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert data['error'] == 'Invalid request to OpenAI API'
        
    @patch('app.openai_client.chat.completions.create')
    def test_generic_openai_error(self, mock_openai, client):
        """Test handling of generic OpenAI error."""
        from openai import OpenAIError
        mock_openai.side_effect = OpenAIError("Generic error")
        
        test_data = {"prompt": "Test prompt"}
        
        response = client.post('/generate',
                             data=json.dumps(test_data),
                             content_type='application/json')
        
        assert response.status_code == 500
        data = response.get_json()
        assert 'error' in data
        assert data['error'] == 'OpenAI API error'

class TestErrorHandlers:
    """Test error handlers."""
    
    def test_404_handler(self, client):
        """Test 404 error handler."""
        response = client.get('/nonexistent')
        assert response.status_code == 404
        data = response.get_json()
        assert data['error'] == 'Endpoint not found'
        
    def test_405_handler(self, client):
        """Test 405 error handler."""
        response = client.post('/health')  # Health endpoint only accepts GET
        assert response.status_code == 405
        data = response.get_json()
        assert data['error'] == 'Method not allowed'
