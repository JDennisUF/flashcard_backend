# Flashcard Backend

A secure Flask backend server for the flashcard application that handles OpenAI API requests without exposing the API key to the frontend.

## Features

- Secure Flask server with CORS support
- OpenAI API integration for flashcard generation
- Environment variable configuration
- Comprehensive error handling
- Rate limiting protection
- Request validation
- Health check endpoint

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. **Run the server:**
   ```bash
   python app.py
   ```

   The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
```

Returns server status and timestamp.

### Generate Flashcards
```
POST /generate
Content-Type: application/json
```

**Request Body:**
```json
{
  "prompt": "Create flashcards about Python programming",
  "model": "gpt-3.5-turbo",
  "max_tokens": 1000,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "content": "Generated flashcard content...",
  "model": "gpt-3.5-turbo",
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  }
}
```

## Security Features

- API key stored in environment variables
- CORS configured for frontend communication
- Request validation and sanitization
- Comprehensive error handling
- Rate limiting error handling

## Production Deployment

For production deployment, consider:

1. Use `gunicorn` as the WSGI server:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. Set `FLASK_ENV=production` in your environment

3. Use a reverse proxy (nginx) for additional security

4. Implement additional security measures like API rate limiting

## Error Handling

The server handles various OpenAI API errors:
- Rate limit exceeded (429)
- Invalid requests (400)
- Authentication errors (401)
- General OpenAI API errors (500)

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `FLASK_ENV`: Flask environment (development/production)
- `PORT`: Server port (default: 5000)
- `SECRET_KEY`: Flask secret key (for sessions, if needed)
