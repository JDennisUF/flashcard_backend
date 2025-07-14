#!/usr/bin/env python3
"""
Test script to verify your Flask backend works with your OpenAI API key.
This tests both unit tests (mocked) and real API calls.
"""

import subprocess
import sys
import time
import requests
import json
import os
from dotenv import load_dotenv

def run_unit_tests():
    """Run the unit tests with mocked OpenAI responses."""
    print("🧪 Running Unit Tests...")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_app.py", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            print("✅ All unit tests passed!")
            return True
        else:
            print("❌ Some unit tests failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def test_import():
    """Test that the app can be imported without errors."""
    print("\n📦 Testing App Import...")
    print("=" * 50)
    
    try:
        from app import app
        print("✅ App imports successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to import app: {e}")
        return False

def test_openai_key():
    """Test that OpenAI API key is configured."""
    print("\n🔑 Checking OpenAI API Key...")
    print("=" * 50)
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No OpenAI API key found in .env file!")
        return False
    
    if not api_key.startswith('sk-'):
        print("⚠️  OpenAI API key doesn't look valid (should start with 'sk-')")
        return False
    
    print("✅ OpenAI API key is configured!")
    return True

def test_real_api_call():
    """Test a real API call to OpenAI (if possible)."""
    print("\n🌐 Testing Real OpenAI API Call...")
    print("=" * 50)
    
    try:
        from openai import OpenAI
        load_dotenv()
        
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Make a small, cheap API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API test successful' in exactly those words."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        content = response.choices[0].message.content
        print(f"OpenAI Response: {content}")
        
        if "API test successful" in content:
            print("✅ OpenAI API call successful!")
            return True
        else:
            print("⚠️  OpenAI API call worked but response was unexpected")
            return True
            
    except Exception as e:
        print(f"❌ OpenAI API call failed: {e}")
        return False

def test_flask_routes():
    """Test Flask routes without starting a server."""
    print("\n🧭 Testing Flask Routes...")
    print("=" * 50)
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/health')
            if response.status_code == 200:
                print("✅ Health endpoint works!")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
            
            # Test generate endpoint with missing prompt
            response = client.post('/generate', 
                                 json={}, 
                                 content_type='application/json')
            if response.status_code == 400:
                print("✅ Generate endpoint properly validates missing prompt!")
            else:
                print(f"❌ Generate endpoint validation failed: {response.status_code}")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Flask route test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Flask Backend Test Suite")
    print("=" * 50)
    
    results = []
    
    # Run all tests
    results.append(("Import Test", test_import()))
    results.append(("Unit Tests", run_unit_tests()))
    results.append(("OpenAI Key Check", test_openai_key()))
    results.append(("Flask Routes", test_flask_routes()))
    results.append(("Real API Call", test_real_api_call()))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your backend is ready for deployment!")
        print("\nTo deploy to Render.com:")
        print("1. Push your code to GitHub")
        print("2. Connect your GitHub repo to Render")
        print("3. Set environment variable: OPENAI_API_KEY")
        print("4. Use build command: pip install -r requirements.txt")
        print("5. Use start command: gunicorn app:app")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please fix before deploying.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
