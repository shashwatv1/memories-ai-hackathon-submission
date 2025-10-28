#!/usr/bin/env python3
"""
Simple Hume AI API Endpoint Discovery

This script tests a few key endpoints to discover the correct Hume AI API structure.
"""

import requests
import json


def test_key_endpoints():
    """Test the most likely Hume AI API endpoints"""
    print("🔍 Testing Key Hume AI Endpoints")
    print("=" * 50)
    
    # API credentials
    api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
    secret = "3MO0RR86DZckIi2sofhGYf1JQL1qhdGLvACqPAe49qBZGM5N9zIGejosFj3ezmhg"
    
    # Test a few key endpoints
    endpoints = [
        {
            'name': 'Hume AI Base API',
            'url': 'https://api.hume.ai',
            'method': 'GET',
            'headers': {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        },
        {
            'name': 'Hume AI V1 API',
            'url': 'https://api.hume.ai/v1',
            'method': 'GET',
            'headers': {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        },
        {
            'name': 'Hume AI Models',
            'url': 'https://api.hume.ai/v1/models',
            'method': 'GET',
            'headers': {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        },
        {
            'name': 'Hume AI EVI',
            'url': 'https://api.hume.ai/v1/evi',
            'method': 'GET',
            'headers': {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        }
    ]
    
    for endpoint in endpoints:
        print(f"\n🧪 Testing: {endpoint['name']}")
        print(f"📡 URL: {endpoint['url']}")
        
        try:
            if endpoint['method'] == 'GET':
                response = requests.get(endpoint['url'], headers=endpoint['headers'], timeout=10)
            else:
                response = requests.post(endpoint['url'], headers=endpoint['headers'], timeout=10)
            
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ SUCCESS!")
                try:
                    data = response.json()
                    print(f"📄 Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"📄 Response: {response.text[:300]}...")
            elif response.status_code == 401:
                print("❌ Authentication failed")
            elif response.status_code == 403:
                print("❌ Access forbidden")
            elif response.status_code == 404:
                print("❌ Not found")
            else:
                print(f"⚠️  Status {response.status_code}: {response.text[:100]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")


def test_alternative_auth():
    """Test alternative authentication methods"""
    print(f"\n🔐 Testing Alternative Authentication")
    print("=" * 50)
    
    api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
    secret = "3MO0RR86DZckIi2sofhGYf1JQL1qhdGLvACqPAe49qBZGM5N9zIGejosFj3ezmhg"
    
    # Test different auth methods
    auth_methods = [
        {
            'name': 'Bearer Token Only',
            'headers': {"Authorization": f"Bearer {api_key}"}
        },
        {
            'name': 'API Key Header',
            'headers': {"X-API-Key": api_key}
        },
        {
            'name': 'Hume Secret Header',
            'headers': {"X-Hume-Secret": secret}
        },
        {
            'name': 'Both Keys',
            'headers': {
                "Authorization": f"Bearer {api_key}",
                "X-Hume-Secret": secret
            }
        }
    ]
    
    test_url = "https://api.hume.ai/v1/models"
    
    for auth in auth_methods:
        print(f"\n🔑 Testing: {auth['name']}")
        
        try:
            response = requests.get(test_url, headers=auth['headers'], timeout=10)
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ SUCCESS!")
                try:
                    data = response.json()
                    print(f"📄 Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"📄 Response: {response.text[:200]}...")
            else:
                print(f"❌ Failed: {response.text[:100]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")


def test_voice_generation_endpoints():
    """Test voice generation specific endpoints"""
    print(f"\n🎙️ Testing Voice Generation Endpoints")
    print("=" * 50)
    
    api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
    secret = "3MO0RR86DZckIi2sofhGYf1JQL1qhdGLvACqPAe49qBZGM5N9zIGejosFj3ezmhg"
    
    # Test voice generation endpoints
    voice_endpoints = [
        {
            'name': 'Voice Generate',
            'url': 'https://api.hume.ai/v1/voice/generate',
            'payload': {
                "text": "Hello, this is a test.",
                "emotion": "neutral"
            }
        },
        {
            'name': 'EVI Generate',
            'url': 'https://api.hume.ai/v1/evi/generate',
            'payload': {
                "text": "Hello, this is a test.",
                "emotion": "happy"
            }
        },
        {
            'name': 'Text to Speech',
            'url': 'https://api.hume.ai/v1/tts',
            'payload': {
                "text": "Hello, this is a test."
            }
        }
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    for endpoint in voice_endpoints:
        print(f"\n🎬 Testing: {endpoint['name']}")
        print(f"📡 URL: {endpoint['url']}")
        
        try:
            response = requests.post(
                endpoint['url'],
                json=endpoint['payload'],
                headers=headers,
                timeout=15
            )
            
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ SUCCESS!")
                try:
                    data = response.json()
                    print(f"📄 Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"📄 Response: {response.text[:300]}...")
            else:
                print(f"❌ Failed: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🧪 Simple Hume AI API Discovery")
    print("=" * 60)
    
    # Test key endpoints
    test_key_endpoints()
    
    # Test alternative authentication
    test_alternative_auth()
    
    # Test voice generation endpoints
    test_voice_generation_endpoints()
    
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print("This test shows us:")
    print("1. Which endpoints are accessible")
    print("2. What authentication method works")
    print("3. What the API structure looks like")
    print("4. Whether voice generation is available")
