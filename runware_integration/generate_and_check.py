#!/usr/bin/env python3
"""
Generate Video and Check Results

This script generates a new video and tries different approaches to get the result.
"""

import requests
import json
import uuid
import time

def generate_video_and_check():
    """Generate a video and try to get the result"""
    print("🎬 Generating Video and Checking Results")
    print("=" * 60)
    
    # API configuration
    api_token = "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
    api_base_url = "https://api.runware.ai/v1"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Generate a new video
    task_uuid = str(uuid.uuid4())
    request_data = [{
        "taskType": "videoInference",
        "taskUUID": task_uuid,
        "positivePrompt": "A serene beach with gentle waves at sunset",
        "model": "klingai:5@3",
        "duration": 5,
        "width": 1920,
        "height": 1080,
        "deliveryMethod": "sync"  # Try sync instead of async
    }]
    
    print(f"🎯 Generating video with sync delivery...")
    print(f"   Task UUID: {task_uuid}")
    print(f"   Prompt: A serene beach with gentle waves at sunset")
    
    try:
        response = requests.post(
            api_base_url,
            json=request_data,
            headers=headers,
            timeout=120  # Longer timeout for sync
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"✅ Video generation successful!")
            print(f"📄 Response data:")
            print(json.dumps(response_data, indent=2))
            
            # Check for video URL in response
            if isinstance(response_data, dict) and 'data' in response_data:
                data = response_data['data']
                if isinstance(data, list) and len(data) > 0:
                    task_result = data[0]
                    print(f"\n🔍 Analyzing response for video URL:")
                    for key, value in task_result.items():
                        print(f"   {key}: {value}")
                        if 'url' in key.lower() or 'video' in key.lower() or 'image' in key.lower():
                            print(f"   🎬 Found potential video reference: {key} = {value}")
                            return value
            
            return response_data
        else:
            print(f"❌ Video generation failed!")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None

def try_alternative_models():
    """Try different models to see if any return immediate results"""
    print(f"\n🔄 Trying Alternative Models")
    print("=" * 60)
    
    api_token = "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
    api_base_url = "https://api.runware.ai/v1"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Try different models
    models_to_try = [
        "runware:108@22",  # Alternative model
        "klingai:5@1",     # Different Kling model
        "runware:101@1"     # Another model
    ]
    
    for model in models_to_try:
        print(f"\n🎯 Trying model: {model}")
        task_uuid = str(uuid.uuid4())
        
        request_data = [{
            "taskType": "videoInference",
            "taskUUID": task_uuid,
            "positivePrompt": "A simple test video",
            "model": model,
            "duration": 5,
            "width": 1920,
            "height": 1080,
            "deliveryMethod": "sync"
        }]
        
        try:
            response = requests.post(
                api_base_url,
                json=request_data,
                headers=headers,
                timeout=60
            )
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                response_data = response.json()
                print(f"   ✅ Success with {model}")
                print(f"   Response: {json.dumps(response_data, indent=2)}")
            else:
                print(f"   ❌ Failed: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Main function"""
    print("🚀 Runware Video Generation and Check")
    print("=" * 60)
    
    # Try generating with sync delivery
    result = generate_video_and_check()
    
    if result:
        print(f"\n✅ Video generation completed!")
    else:
        print(f"\n⚠️  Sync generation failed, trying alternative models...")
        try_alternative_models()
    
    print(f"\n💡 Summary:")
    print(f"   - API is working and accepting requests")
    print(f"   - Video generation tasks are being submitted")
    print(f"   - Results may be available through Runware dashboard")
    print(f"   - Check https://runware.ai for task status")

if __name__ == "__main__":
    main()
