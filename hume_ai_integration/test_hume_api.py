#!/usr/bin/env python3
"""
Test Hume AI API Calls

This script tests the Hume AI API to show you exactly what calls are being made
and what responses we get from the API.
"""

import os
import sys
import requests
import json
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hume_voice_client import HumeVoiceClient


def test_api_calls():
    """Test Hume AI API calls and show detailed request/response information"""
    print("🔍 Testing Hume AI API Calls")
    print("=" * 60)
    
    # API credentials from environment
    api_key = os.getenv("HUME_API_KEY", "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh")
    secret = os.getenv("HUME_SECRET", "3MO0RR86DZckIi2sofhGYf1JQL1qhdGLvACqPAe49qBZGM5N9zIGejosFj3ezmhg")
    
    print(f"🔑 API Key: {api_key[:20]}...")
    print(f"🔐 Secret: {secret[:20]}...")
    print()
    
    # Test different API endpoints and methods
    test_endpoints = [
        {
            'name': 'Voice Generation Endpoint',
            'url': 'https://api.hume.ai/v1/voice/generate',
            'method': 'POST',
            'payload': {
                "text": "Hello, this is a test of Hume AI voice generation.",
                "emotion": "neutral",
                "intensity": 0.5,
                "voice_style": "narrator",
                "output_format": "mp3"
            }
        },
        {
            'name': 'Alternative Voice Endpoint',
            'url': 'https://api.hume.ai/v1/evi/generate',
            'method': 'POST',
            'payload': {
                "text": "This is an alternative endpoint test.",
                "emotion": "happy",
                "intensity": 0.7
            }
        },
        {
            'name': 'Emotion Analysis Endpoint',
            'url': 'https://api.hume.ai/v1/emotion/analyze',
            'method': 'POST',
            'payload': {
                "text": "I am feeling very happy today!"
            }
        }
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Hume-Secret": secret
    }
    
    for i, endpoint in enumerate(test_endpoints, 1):
        print(f"\n🧪 Test {i}: {endpoint['name']}")
        print(f"📡 URL: {endpoint['url']}")
        print(f"🔧 Method: {endpoint['method']}")
        print(f"📦 Payload: {json.dumps(endpoint['payload'], indent=2)}")
        print(f"📋 Headers: {json.dumps(headers, indent=2)}")
        print("-" * 50)
        
        try:
            if endpoint['method'] == 'POST':
                response = requests.post(
                    endpoint['url'],
                    json=endpoint['payload'],
                    headers=headers,
                    timeout=30
                )
            else:
                response = requests.get(
                    endpoint['url'],
                    headers=headers,
                    timeout=30
                )
            
            print(f"📊 Status Code: {response.status_code}")
            print(f"📋 Response Headers: {dict(response.headers)}")
            
            if response.text:
                try:
                    response_json = response.json()
                    print(f"📄 Response JSON: {json.dumps(response_json, indent=2)}")
                except json.JSONDecodeError:
                    print(f"📄 Response Text: {response.text[:500]}...")
            else:
                print("📄 Response: (empty)")
            
            if response.status_code == 200:
                print("✅ Request successful!")
            elif response.status_code == 401:
                print("❌ Authentication failed - check API credentials")
            elif response.status_code == 403:
                print("❌ Access forbidden - check API permissions")
            elif response.status_code == 404:
                print("❌ Endpoint not found - check API URL")
            elif response.status_code == 429:
                print("❌ Rate limit exceeded - too many requests")
            else:
                print(f"❌ Request failed with status {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - check internet connection")
        except Exception as e:
            print(f"❌ Request failed with exception: {e}")
        
        print()
    
    # Test with HumeVoiceClient
    print("\n🎙️ Testing with HumeVoiceClient")
    print("-" * 50)
    
    try:
        client = HumeVoiceClient()
        print("✅ HumeVoiceClient initialized")
        
        # Test connection
        connection_result = client.test_connection()
        print(f"🔍 Connection test: {connection_result}")
        
        # Test voice generation
        voice_result = client.generate_voice(
            text="This is a test of the Hume AI voice generation system.",
            emotion="neutral",
            intensity=0.5,
            voice_style="narrator"
        )
        print(f"🎙️ Voice generation result: {json.dumps(voice_result, indent=2)}")
        
    except Exception as e:
        print(f"❌ HumeVoiceClient test failed: {e}")


def test_alternative_endpoints():
    """Test alternative API endpoints that might work"""
    print("\n🔄 Testing Alternative Endpoints")
    print("=" * 60)
    
    api_key = os.getenv("HUME_API_KEY", "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh")
    
    # Alternative endpoint configurations
    alternative_configs = [
        {
            'name': 'Hume AI EVI Endpoint',
            'url': 'https://api.hume.ai/v1/evi',
            'headers': {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        },
        {
            'name': 'Hume AI Voice Endpoint',
            'url': 'https://api.hume.ai/v1/voice',
            'headers': {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        },
        {
            'name': 'Hume AI Text-to-Speech',
            'url': 'https://api.hume.ai/v1/tts',
            'headers': {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        }
    ]
    
    for config in alternative_configs:
        print(f"\n🧪 Testing {config['name']}")
        print(f"📡 URL: {config['url']}")
        
        try:
            # Try GET request first
            response = requests.get(config['url'], headers=config['headers'], timeout=10)
            print(f"📊 GET Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ GET request successful!")
                try:
                    print(f"📄 Response: {json.dumps(response.json(), indent=2)}")
                except:
                    print(f"📄 Response: {response.text[:200]}...")
            else:
                print(f"❌ GET failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ GET request failed: {e}")
        
        # Try POST request with simple payload
        try:
            post_payload = {
                "text": "Hello, this is a test.",
                "emotion": "neutral"
            }
            
            response = requests.post(
                config['url'], 
                json=post_payload, 
                headers=config['headers'], 
                timeout=10
            )
            print(f"📊 POST Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ POST request successful!")
                try:
                    print(f"📄 Response: {json.dumps(response.json(), indent=2)}")
                except:
                    print(f"📄 Response: {response.text[:200]}...")
            else:
                print(f"❌ POST failed: {response.status_code}")
                print(f"📄 Error: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ POST request failed: {e}")


if __name__ == "__main__":
    print("🧪 Hume AI API Testing")
    print("=" * 60)
    
    # Test main API calls
    test_api_calls()
    
    # Test alternative endpoints
    test_alternative_endpoints()
    
    print("\n📋 Summary:")
    print("This test shows you exactly what API calls are being made to Hume AI")
    print("and what responses we get. This will help us understand:")
    print("1. Which endpoints are available")
    print("2. What authentication is required")
    print("3. What the API expects and returns")
    print("4. Any rate limits or restrictions")
