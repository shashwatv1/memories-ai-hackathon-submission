#!/usr/bin/env python3
"""
Single Test Script for Runware Video Generation

This script runs one test case at a time to avoid long waits.
"""

import os
import sys
import time
import requests
import json
import uuid
from pathlib import Path

def run_single_test(test_name, prompt, negative_prompt, duration, width, height):
    """Run a single test case"""
    print(f"🧪 {test_name}")
    print(f"📝 Prompt: {prompt}")
    print(f"🚫 Negative: {negative_prompt}")
    print(f"⏱️  Duration: {duration}s")
    print(f"📐 Resolution: {width}x{height}")
    print("-" * 50)
    
    # API configuration
    api_token = "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
    api_base_url = "https://api.runware.ai/v1"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    try:
        # Submit generation request
        task_uuid = str(uuid.uuid4())
        request_data = [{
            "taskType": "videoInference",
            "taskUUID": task_uuid,
            "positivePrompt": prompt,
            "negativePrompt": negative_prompt,
            "model": "klingai:5@3",
            "duration": duration,
            "width": width,
            "height": height,
            "deliveryMethod": "async"
        }]
        
        print(f"   🔄 Submitting video generation request...")
        response = requests.post(
            api_base_url,
            json=request_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"   ✅ Request submitted successfully")
            print(f"   📋 Task UUID: {task_uuid}")
            
            # Poll for completion
            print(f"   🔍 Polling for completion...")
            poll_result = poll_task_status(task_uuid, headers, api_base_url)
            
            if poll_result['success']:
                print(f"   🎬 Video URL: {poll_result['video_url']}")
                
                # Download video
                print(f"   📥 Downloading video...")
                download_result = download_video(poll_result['video_url'], test_name)
                
                if download_result['success']:
                    print(f"   ✅ Video downloaded: {download_result['file_path']}")
                    print(f"   📊 File size: {download_result['file_size']:,} bytes ({download_result['file_size']/1024/1024:.2f} MB)")
                    return {
                        'success': True,
                        'file_path': download_result['file_path'],
                        'file_size': download_result['file_size'],
                        'task_uuid': task_uuid,
                        'video_url': poll_result['video_url']
                    }
                else:
                    print(f"   ❌ Download failed: {download_result['error']}")
                    return {'success': False, 'error': f"Download failed: {download_result['error']}"}
            else:
                print(f"   ❌ Polling failed: {poll_result['error']}")
                return {'success': False, 'error': f"Polling failed: {poll_result['error']}"}
        else:
            print(f"   ❌ API Error: {response.status_code} - {response.text}")
            return {'success': False, 'error': f"API Error: {response.status_code}"}
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'success': False, 'error': str(e)}

def poll_task_status(task_uuid, headers, api_base_url, max_attempts=15, delay=10):
    """Poll task status until completion"""
    for attempt in range(1, max_attempts + 1):
        try:
            request_data = [{
                "taskType": "getResponse",
                "taskUUID": task_uuid
            }]
            
            response = requests.post(
                api_base_url,
                json=request_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                if isinstance(response_data, dict) and 'data' in response_data:
                    data = response_data['data']
                    if isinstance(data, list) and len(data) > 0:
                        task_result = data[0]
                        status = task_result.get('status', 'unknown')
                        
                        print(f"   📊 Attempt {attempt}: Status = {status}")
                        
                        if status == 'success':
                            # Look for video URL
                            video_url = None
                            for key, value in task_result.items():
                                if 'url' in key.lower() and ('video' in key.lower() or 'image' in key.lower()):
                                    video_url = value
                                    break
                            
                            if video_url:
                                return {'success': True, 'video_url': video_url}
                            else:
                                return {'success': False, 'error': 'No video URL found in response'}
                        
                        elif status in ['failed', 'error']:
                            return {'success': False, 'error': f"Task failed with status: {status}"}
                        
                        else:
                            if attempt < max_attempts:
                                print(f"   ⏰ Waiting {delay} seconds...")
                                time.sleep(delay)
                            continue
            
            if attempt < max_attempts:
                time.sleep(delay)
                
        except Exception as e:
            print(f"   ❌ Attempt {attempt}: Error - {e}")
            if attempt < max_attempts:
                time.sleep(delay)
    
    return {'success': False, 'error': f"Max attempts ({max_attempts}) reached"}

def download_video(video_url, name):
    """Download video from URL"""
    try:
        # Create output directory
        output_dir = Path("output/generated_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace(' ', '_').replace(':', '').replace('/', '_')
        filename = f"{safe_name}_{timestamp}.mp4"
        file_path = output_dir / filename
        
        # Download video
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = file_path.stat().st_size
        return {
            'success': True,
            'file_path': str(file_path),
            'file_size': file_size
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # Test 1: Basic 5s Landscape Video
    print("🚀 Starting Test 1: Basic 5s Landscape Video")
    result1 = run_single_test(
        "Test_1_Basic_5s_Landscape",
        "A beautiful sunset over mountains",
        "blurry, dark, low quality",
        5,
        1920,
        1080
    )
    
    print(f"\n📊 Test 1 Result: {'✅ SUCCESS' if result1.get('success') else '❌ FAILED'}")
    if result1.get('success'):
        print(f"   📁 File: {result1['file_path']}")
        print(f"   📊 Size: {result1['file_size']:,} bytes")
    else:
        print(f"   ❌ Error: {result1.get('error')}")
