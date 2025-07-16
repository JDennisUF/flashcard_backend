from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import requests

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
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

if not openai_api_key:
    logger.error("OpenAI API key not found in environment variables")
    raise ValueError("OpenAI API key is required")

if not openrouter_api_key:
    logger.error("OpenRouter API key not found in environment variables")
    raise ValueError("OpenRouter API key is required")

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
    """Generate flashcards using OpenRouter API"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        count = data.get('count', None)
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required and cannot be empty'}), 400
        if count is None or not isinstance(count, int) or count < 1 or count > 100:
            return jsonify({'success': False, 'error': 'Count must be an integer between 1 and 100'}), 400
        # Update prompt to demand strict format
        prompt_with_count = (
            f"Create {count} flashcards. "
            f"Return ONLY in this format, no extra text: "
            f"Flashcard N:\n**Question:** ...\n**Answer:** ...\n" 
            f"where N is the number, and each question/answer is on its own line. Do not include anything else. {prompt}"
        )
        model = data.get('model', 'mistralai/mistral-7b-instruct')
        max_tokens = data.get('max_tokens', 1000)
        temperature = data.get('temperature', 0.7)
        if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 4000:
            max_tokens = 1000
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            temperature = 0.7
        if len(prompt_with_count) > 4000:
            return jsonify({'success': False, 'error': 'Prompt too long (max 4000 characters)'}), 400
        # logger.info(f"Prompt: {prompt_with_count}")
        # Call OpenRouter API
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that creates educational flashcards. Generate well-structured flashcards based on the given topic or content."},
                {"role": "user", "content": prompt_with_count}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            logger.error(f"OpenRouter API error: {response.text}")
            return jsonify({'success': False, 'error': f'OpenRouter API error: {response.text}'}), response.status_code
        result = response.json()
        generated_content = result['choices'][0]['message']['content']
        # logger.info(f"Successfully generated {len(generated_content)} characters of content")
        usage = result.get('usage', {})

        # Parse flashcards from the generated content
        import re
        flashcards = []
        # Match blocks like: Flashcard N:\n**Question:** ...\n**Answer:** ... or N:\n**Question:** ...\n**Answer:** ...
        pattern = re.compile(r"(?:Flashcard\s*)?(\d+):\s*\*\*Question:\*\*\s*(.*?)\s*\*\*Answer:\*\*\s*(.*?)(?=\n(?:Flashcard\s*)?\d+:|$)", re.DOTALL)
        for match in pattern.finditer(generated_content):
            question = match.group(2).strip()
            answer = match.group(3).strip()
            flashcards.append({"question": question, "answer": answer})

        return jsonify({
            'success': True,
            'flashcards': flashcards,
            'model': model,
            'usage': usage
        })
    except requests.exceptions.Timeout:
        logger.error("OpenRouter API request timed out")
        return jsonify({'success': False, 'error': 'OpenRouter API request timed out'}), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API request error: {str(e)}")
        return jsonify({'success': False, 'error': f'OpenRouter API request error: {str(e)}'}), 502
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
