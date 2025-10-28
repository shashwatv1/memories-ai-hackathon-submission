#!/usr/bin/env python3
"""
Comprehensive Hume AI API Testing

This script tests various possible API endpoints and authentication methods
to discover the correct way to use Hume AI's voice generation API.
"""

import os
import requests
import json
from typing import Dict, Any


def test_api_endpoints():
    """Test various possible API endpoints"""
    print("🔍 Comprehensive Hume AI API Testing")
    print("=" * 60)
    
    # API credentials
    api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
    secret = "3MO0RR86DZckIi2sofhGYf1JQL1qhdGLvACqPAe49qBZGM5N9zIGejosFj3ezmhg"
    
    # Test different base URLs
    base_urls = [
        "https://api.hume.ai",
        "https://hume.ai/api",
        "https://api.hume.ai/v1",
        "https://api.hume.ai/v2",
        "https://api.hume.ai/evi",
        "https://evi.hume.ai",
        "https://voice.hume.ai"
    ]
    
    # Test different authentication methods
    auth_methods = [
        {
            'name': 'Bearer Token',
            'headers': {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        },
        {
            'name': 'API Key Header',
            'headers': {
                "X-API-Key": api_key,
                "Content-Type": "application/json"
            }
        },
        {
            'name': 'Hume Secret Header',
            'headers': {
                "X-Hume-Secret": secret,
                "Content-Type": "application/json"
            }
        },
        {
            'name': 'Both Keys',
            'headers': {
                "Authorization": f"Bearer {api_key}",
                "X-Hume-Secret": secret,
                "Content-Type": "application/json"
            }
        },
        {
            'name': 'API Key + Secret',
            'headers': {
                "X-API-Key": api_key,
                "X-Hume-Secret": secret,
                "Content-Type": "application/json"
            }
        }
    ]
    
    # Test different endpoints
    endpoints = [
        "/",
        "/v1",
        "/v1/voice",
        "/v1/voice/generate",
        "/v1/evi",
        "/v1/evi/generate",
        "/v1/tts",
        "/v1/tts/generate",
        "/v1/speech",
        "/v1/speech/generate",
        "/v1/audio",
        "/v1/audio/generate",
        "/v1/models",
        "/v1/emotions",
        "/v1/emotion",
        "/v1/emotion/analyze",
        "/v1/emotion/generate",
        "/v1/text-to-speech",
        "/v1/synthesize",
        "/v1/speak",
        "/v1/voice/synthesize",
        "/v1/voice/speak",
        "/v1/voice/emotion",
        "/v1/voice/emotional",
        "/v1/empathic",
        "/v1/empathic/voice",
        "/v1/empathic/generate"
    ]
    
    # Test payloads
    test_payloads = [
        {
            'name': 'Simple Text',
            'payload': {
                "text": "Hello, this is a test."
            }
        },
        {
            'name': 'Text with Emotion',
            'payload': {
                "text": "Hello, this is a test.",
                "emotion": "happy"
            }
        },
        {
            'name': 'Full Voice Request',
            'payload': {
                "text": "Hello, this is a test of voice generation.",
                "emotion": "neutral",
                "intensity": 0.5,
                "voice_style": "narrator",
                "output_format": "mp3"
            }
        },
        {
            'name': 'EVI Request',
            'payload': {
                "text": "Hello, this is a test of EVI.",
                "emotion": "calm",
                "intensity": 0.3
            }
        }
    ]
    
    successful_endpoints = []
    
    for base_url in base_urls:
        print(f"\n🌐 Testing Base URL: {base_url}")
        print("-" * 50)
        
        for auth_method in auth_methods:
            print(f"\n🔐 Testing Auth: {auth_method['name']}")
            
            for endpoint in endpoints:
                full_url = base_url + endpoint
                
                # Test GET request first
                try:
                    response = requests.get(full_url, headers=auth_method['headers'], timeout=10)
                    
                    if response.status_code == 200:
                        print(f"✅ GET {full_url} - Status: {response.status_code}")
                        try:
                            data = response.json()
                            print(f"📄 Response: {json.dumps(data, indent=2)[:200]}...")
                            successful_endpoints.append({
                                'url': full_url,
                                'method': 'GET',
                                'auth': auth_method['name'],
                                'response': data
                            })
                        except:
                            print(f"📄 Response: {response.text[:200]}...")
                    elif response.status_code != 404:
                        print(f"⚠️  GET {full_url} - Status: {response.status_code}")
                        print(f"📄 Response: {response.text[:100]}...")
                        
                except Exception as e:
                    pass  # Skip failed requests
                
                # Test POST request with each payload
                for payload_info in test_payloads:
                    try:
                        response = requests.post(
                            full_url,
                            json=payload_info['payload'],
                            headers=auth_method['headers'],
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            print(f"✅ POST {full_url} ({payload_info['name']}) - Status: {response.status_code}")
                            try:
                                data = response.json()
                                print(f"📄 Response: {json.dumps(data, indent=2)[:200]}...")
                                successful_endpoints.append({
                                    'url': full_url,
                                    'method': 'POST',
                                    'auth': auth_method['name'],
                                    'payload': payload_info['name'],
                                    'response': data
                                })
                            except:
                                print(f"📄 Response: {response.text[:200]}...")
                        elif response.status_code not in [404, 405]:
                            print(f"⚠️  POST {full_url} ({payload_info['name']}) - Status: {response.status_code}")
                            print(f"📄 Response: {response.text[:100]}...")
                            
                    except Exception as e:
                        pass  # Skip failed requests
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total successful endpoints found: {len(successful_endpoints)}")
    
    if successful_endpoints:
        print(f"\n✅ Working Endpoints:")
        for endpoint in successful_endpoints:
            print(f"  - {endpoint['method']} {endpoint['url']}")
            print(f"    Auth: {endpoint['auth']}")
            if 'payload' in endpoint:
                print(f"    Payload: {endpoint['payload']}")
            print()
    else:
        print(f"\n❌ No working endpoints found")
        print(f"This suggests either:")
        print(f"1. The API endpoints are different from what we tested")
        print(f"2. The API requires different authentication")
        print(f"3. The API is not publicly available")
        print(f"4. The credentials are incorrect or expired")
    
    return successful_endpoints


def test_alternative_approaches():
    """Test alternative approaches to find the API"""
    print(f"\n🔄 Alternative Approaches")
    print("=" * 60)
    
    # Test if it's a different service entirely
    alternative_services = [
        {
            'name': 'OpenAI TTS',
            'url': 'https://api.openai.com/v1/audio/speech',
            'headers': {
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json"
            },
            'payload': {
                "model": "tts-1",
                "input": "Hello, this is a test.",
                "voice": "alloy"
            }
        },
        {
            'name': 'ElevenLabs',
            'url': 'https://api.elevenlabs.io/v1/text-to-speech/voice-id',
            'headers': {
                "xi-api-key": "test-key",
                "Content-Type": "application/json"
            },
            'payload': {
                "text": "Hello, this is a test.",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
        }
    ]
    
    for service in alternative_services:
        print(f"\n🧪 Testing {service['name']}")
        try:
            response = requests.post(
                service['url'],
                json=service['payload'],
                headers=service['headers'],
                timeout=10
            )
            print(f"📊 Status: {response.status_code}")
            if response.status_code != 404:
                print(f"📄 Response: {response.text[:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Test comprehensive API endpoints
    successful_endpoints = test_api_endpoints()
    
    # Test alternative approaches
    test_alternative_approaches()
    
    print(f"\n📋 CONCLUSION")
    print("=" * 60)
    print("This comprehensive test shows us:")
    print("1. What API endpoints exist (if any)")
    print("2. What authentication methods work")
    print("3. What request formats are expected")
    print("4. Whether the API is accessible with our credentials")
    print("\nBased on the results, we can determine the correct way to use Hume AI's API.")
