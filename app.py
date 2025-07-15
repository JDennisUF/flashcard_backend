from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI, RateLimitError, AuthenticationError, BadRequestError, OpenAIError
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    logger.error("OpenAI API key not found in environment variables")
    raise ValueError("OpenAI API key is required")

# Create OpenAI client - simplified initialization
try:
    openai_client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise RuntimeError(f"OpenAI client initialization failed: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'flashcard-backend'
    })

@app.route('/generate', methods=['POST'])
def generate_flashcards():
    """Generate flashcards using OpenAI API"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Only accept 'prompt' and 'count'
        prompt = data.get('prompt', '').strip()
        count = data.get('count', None)
        
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required and cannot be empty'}), 400
        
        if count is None or not isinstance(count, int) or count < 1 or count > 1000:
            count = 20
        
        # Add count to the prompt
        prompt_with_count = f"Create {count} flashcards. {prompt}"
        
        # Extract optional parameters with validation
        model = data.get('model', 'gpt-3.5-turbo')
        valid_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-16k']
        if model not in valid_models:
            logger.warning(f"Invalid model requested: {model}, using gpt-3.5-turbo")
            model = 'gpt-3.5-turbo'
        max_tokens = data.get('max_tokens', 1000)
        temperature = data.get('temperature', 0.7)
        if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 4000:
            max_tokens = 1000
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            temperature = 0.7
        if len(prompt_with_count) > 4000:
            return jsonify({'success': False, 'error': 'Prompt too long (max 4000 characters)'}), 400
        logger.info(f"Generating {count} flashcards with model: {model}, prompt length: {len(prompt_with_count)}")
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates educational flashcards. Generate well-structured flashcards based on the given topic or content."
                },
                {
                    "role": "user",
                    "content": prompt_with_count
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        generated_content = response.choices[0].message.content
        logger.info(f"Successfully generated {len(generated_content)} characters of content")
        return jsonify({
            'success': True,
            'content': generated_content,
            'model': response.model,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        })
    except RateLimitError:
        logger.error("OpenAI rate limit exceeded")
        return jsonify({'success': False, 'error': 'Rate limit exceeded. Please try again later.'}), 429
    except AuthenticationError:
        logger.error("OpenAI authentication failed")
        return jsonify({'success': False, 'error': 'Authentication failed'}), 401
    except BadRequestError as e:
        logger.error(f"Invalid OpenAI request: {str(e)}")
        return jsonify({'success': False, 'error': 'Invalid request to OpenAI API'}), 400
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({'success': False, 'error': 'OpenAI API error'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
