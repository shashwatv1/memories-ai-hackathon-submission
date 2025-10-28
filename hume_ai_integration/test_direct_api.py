#!/usr/bin/env python3
"""
Direct Hume AI API Testing

This script tests direct HTTP requests to discover the correct Hume AI API structure.
"""

import requests
import json
import os


def test_hume_api_structure():
    """Test different possible Hume AI API structures"""
    print("🔍 Testing Hume AI API Structure")
    print("=" * 50)
    
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
        "https://voice.hume.ai",
        "https://api.hume.ai/voice",
        "https://api.hume.ai/empathic",
        "https://empathic.hume.ai"
    ]
    
    # Test different authentication methods
    auth_configs = [
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
            'name': 'Hume Secret',
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
        }
    ]
    
    # Test different endpoints
    endpoints = [
        "/",
        "/v1",
        "/v1/models",
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
        "/v1/empathic",
        "/v1/empathic/voice",
        "/v1/empathic/generate",
        "/v1/text-to-speech",
        "/v1/synthesize",
        "/v1/speak",
        "/v1/voice/synthesize",
        "/v1/voice/speak",
        "/v1/voice/emotion",
        "/v1/voice/emotional",
        "/v1/empathic/voice",
        "/v1/empathic/generate"
    ]
    
    successful_combinations = []
    
    for base_url in base_urls:
        print(f"\n🌐 Testing Base URL: {base_url}")
        
        for auth in auth_configs:
            print(f"\n🔐 Testing Auth: {auth['name']}")
            
            for endpoint in endpoints:
                full_url = base_url + endpoint
                
                # Test GET request
                try:
                    response = requests.get(full_url, headers=auth['headers'], timeout=5)
                    
                    if response.status_code == 200:
                        print(f"✅ GET {full_url} - Status: {response.status_code}")
                        try:
                            data = response.json()
                            print(f"📄 Response: {json.dumps(data, indent=2)[:200]}...")
                            successful_combinations.append({
                                'url': full_url,
                                'method': 'GET',
                                'auth': auth['name'],
                                'response': data
                            })
                        except:
                            print(f"📄 Response: {response.text[:200]}...")
                    elif response.status_code not in [404, 405]:
                        print(f"⚠️  GET {full_url} - Status: {response.status_code}")
                        print(f"📄 Response: {response.text[:100]}...")
                        
                except Exception as e:
                    pass  # Skip failed requests
                
                # Test POST request with simple payload
                try:
                    payload = {
                        "text": "Hello, this is a test.",
                        "emotion": "neutral"
                    }
                    
                    response = requests.post(
                        full_url,
                        json=payload,
                        headers=auth['headers'],
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        print(f"✅ POST {full_url} - Status: {response.status_code}")
                        try:
                            data = response.json()
                            print(f"📄 Response: {json.dumps(data, indent=2)[:200]}...")
                            successful_combinations.append({
                                'url': full_url,
                                'method': 'POST',
                                'auth': auth['name'],
                                'response': data
                            })
                        except:
                            print(f"📄 Response: {response.text[:200]}...")
                    elif response.status_code not in [404, 405]:
                        print(f"⚠️  POST {full_url} - Status: {response.status_code}")
                        print(f"📄 Response: {response.text[:100]}...")
                        
                except Exception as e:
                    pass  # Skip failed requests
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total successful combinations: {len(successful_combinations)}")
    
    if successful_combinations:
        print(f"\n✅ Working Endpoints:")
        for combo in successful_combinations:
            print(f"  - {combo['method']} {combo['url']}")
            print(f"    Auth: {combo['auth']}")
            print()
    else:
        print(f"\n❌ No working endpoints found")
        print(f"This suggests:")
        print(f"1. The API endpoints are different from what we tested")
        print(f"2. The API requires different authentication")
        print(f"3. The API is not publicly available")
        print(f"4. The credentials are incorrect or expired")
        print(f"5. The API might be in beta or require special access")
    
    return successful_combinations


def test_alternative_services():
    """Test if Hume AI uses alternative service endpoints"""
    print(f"\n🔄 Testing Alternative Service Endpoints")
    print("=" * 50)
    
    # Test if Hume AI uses different service providers
    alternative_services = [
        {
            'name': 'Hume AI via OpenAI',
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
            'name': 'Hume AI via ElevenLabs',
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
    print("🧪 Direct Hume AI API Testing")
    print("=" * 60)
    
    # Test API structure
    successful_combinations = test_hume_api_structure()
    
    # Test alternative services
    test_alternative_services()
    
    print(f"\n📋 CONCLUSION")
    print("=" * 60)
    print("This comprehensive test shows us:")
    print("1. What API endpoints exist (if any)")
    print("2. What authentication methods work")
    print("3. What request formats are expected")
    print("4. Whether the API is accessible with our credentials")
    print("5. Whether Hume AI uses alternative service providers")
    print("\nBased on the results, we can determine the correct way to use Hume AI's API.")
