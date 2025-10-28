#!/usr/bin/env python3
"""
Simple Runware API Test

This script tests the basic Runware API connection and response format.
"""

import sys
import os
import requests
import uuid
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_simple_api():
    """Test simple API call to understand response format"""
    print("🔌 Testing Runware API Response Format")
    print("=" * 50)
    
    # API configuration
    api_token = "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
    api_base_url = "https://api.runware.ai/v1"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Simple test request
    task_uuid = str(uuid.uuid4())
    request_data = [{
        "taskType": "videoInference",
        "taskUUID": task_uuid,
        "positivePrompt": "A beautiful sunset over mountains",
        "model": "klingai:5@3",
        "duration": 5,
        "width": 1920,
        "height": 1080,
        "deliveryMethod": "async"
    }]
    
    print(f"🎯 Making API request...")
    print(f"   Task UUID: {task_uuid}")
    print(f"   Model: klingai:5@3")
    print(f"   Prompt: A beautiful sunset over mountains")
    
    try:
        response = requests.post(
            api_base_url,
            json=request_data,
            headers=headers,
            timeout=60
        )
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"✅ API request successful!")
            print(f"📄 Response data:")
            print(json.dumps(response_data, indent=2))
            
            # Check if there's a video URL in the response
            if isinstance(response_data, list) and len(response_data) > 0:
                task_result = response_data[0]
                print(f"\n🔍 Analyzing response structure:")
                for key, value in task_result.items():
                    print(f"   {key}: {value}")
            
            return True
        else:
            print(f"❌ API request failed!")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Simple Runware API Test")
    print("=" * 60)
    
    success = test_simple_api()
    
    print(f"\n{'='*60}")
    if success:
        print("✅ API test completed successfully!")
        print("💡 Check the response format above to understand the API structure")
    else:
        print("❌ API test failed!")
        print("💡 Check the error messages above")
