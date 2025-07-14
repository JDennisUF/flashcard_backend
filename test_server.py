import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test the server
def test_server():
    base_url = "http://localhost:5000"
    
    print("Testing Flashcard Backend Server...")
    print("=" * 40)
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test flashcard generation
    print("\n2. Testing flashcard generation...")
    test_prompt = {
        "prompt": "Create 3 flashcards about Python variables and data types",
        "model": "gpt-3.5-turbo",
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            json=test_prompt,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Model: {data.get('model')}")
            print(f"Usage: {data.get('usage')}")
            print(f"Content preview: {data.get('content', '')[:200]}...")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Generation test failed: {e}")
    
    # Test invalid request
    print("\n3. Testing invalid request...")
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={"invalid": "request"},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Error response: {response.json()}")
    except Exception as e:
        print(f"Invalid request test failed: {e}")

if __name__ == "__main__":
    test_server()
