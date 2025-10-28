#!/usr/bin/env python3
"""
Simple Test Script for Runware Video Generation

This script tests the 4-parameter implementation with actual video generation.
"""

import os
import sys
import time
import requests
import json
import uuid
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_runware_api():
    """Test Runware API with different parameter combinations"""
    print("🧪 Testing Runware 4-Parameter Implementation")
    print("=" * 60)
    
    # API configuration
    api_token = "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
    api_base_url = "https://api.runware.ai/v1"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Test cases with different parameter combinations
    test_cases = [
        {
            'name': 'Test 1: Basic 5s Landscape',
            'params': {
                'prompt': 'A beautiful sunset over mountains',
                'negative_prompt': 'blurry, dark, low quality',
                'duration': 5,
                'width': 1920,
                'height': 1080
            }
        },
        {
            'name': 'Test 2: Portrait 10s Video',
            'params': {
                'prompt': 'A bird flying through mountain peaks',
                'negative_prompt': 'ugly, distorted, bad quality',
                'duration': 10,
                'width': 1080,
                'height': 1920
            }
        },
        {
            'name': 'Test 3: Square 5s Video',
            'params': {
                'prompt': 'A thunderstorm with lightning flashing',
                'negative_prompt': 'calm, peaceful, sunny',
                'duration': 5,
                'width': 1080,
                'height': 1080
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🎬 {test_case['name']}")
        print(f"📝 Prompt: {test_case['params']['prompt']}")
        print(f"🚫 Negative: {test_case['params']['negative_prompt']}")
        print(f"⏱️  Duration: {test_case['params']['duration']}s")
        print(f"📐 Resolution: {test_case['params']['width']}x{test_case['params']['height']}")
        
        try:
            # Submit generation request
            task_uuid = str(uuid.uuid4())
            request_data = [{
                "taskType": "videoInference",
                "taskUUID": task_uuid,
                "positivePrompt": test_case['params']['prompt'],
                "negativePrompt": test_case['params']['negative_prompt'],
                "model": "klingai:5@3",
                "duration": test_case['params']['duration'],
                "width": test_case['params']['width'],
                "height": test_case['params']['height'],
                "deliveryMethod": "async"
            }]
            
            print(f"   🔄 Submitting request...")
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
                    download_result = download_video(poll_result['video_url'], test_case['name'])
                    
                    if download_result['success']:
                        print(f"   ✅ Video downloaded: {download_result['file_path']}")
                        print(f"   📊 File size: {download_result['file_size']:,} bytes")
                        results.append({
                            'name': test_case['name'],
                            'success': True,
                            'file_path': download_result['file_path'],
                            'file_size': download_result['file_size'],
                            'params': test_case['params']
                        })
                    else:
                        print(f"   ❌ Download failed: {download_result['error']}")
                        results.append({
                            'name': test_case['name'],
                            'success': False,
                            'error': f"Download failed: {download_result['error']}"
                        })
                else:
                    print(f"   ❌ Polling failed: {poll_result['error']}")
                    results.append({
                        'name': test_case['name'],
                        'success': False,
                        'error': f"Polling failed: {poll_result['error']}"
                    })
            else:
                print(f"   ❌ API Error: {response.status_code} - {response.text}")
                results.append({
                    'name': test_case['name'],
                    'success': False,
                    'error': f"API Error: {response.status_code}"
                })
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'name': test_case['name'],
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {total - successful}")
    print(f"Success Rate: {(successful/total)*100:.1f}%")
    
    if successful > 0:
        print(f"\n🎬 Generated Videos:")
        for result in results:
            if result.get('success', False):
                print(f"  - {result['name']}")
                print(f"    📁 Path: {result['file_path']}")
                print(f"    📊 Size: {result['file_size']:,} bytes")
                print(f"    📝 Prompt: {result['params']['prompt']}")
                print(f"    🚫 Negative: {result['params']['negative_prompt']}")
                print(f"    ⏱️  Duration: {result['params']['duration']}s")
                print(f"    📐 Resolution: {result['params']['width']}x{result['params']['height']}")
                print()
    
    return results

def poll_task_status(task_uuid, headers, api_base_url, max_attempts=20, delay=15):
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
                                return {
                                    'success': True,
                                    'video_url': video_url
                                }
                            else:
                                return {
                                    'success': False,
                                    'error': 'No video URL found in response'
                                }
                        
                        elif status in ['failed', 'error']:
                            return {
                                'success': False,
                                'error': f"Task failed with status: {status}"
                            }
                        
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
    
    return {
        'success': False,
        'error': f"Max attempts ({max_attempts}) reached"
    }

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
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    test_runware_api()
