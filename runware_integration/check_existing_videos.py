#!/usr/bin/env python3
"""
Check Existing Videos and Generate New Ones

This script checks existing videos and generates new ones with different parameters.
"""

import os
import sys
import time
import requests
import json
import uuid
from pathlib import Path

def check_existing_videos():
    """Check existing videos in the output directory"""
    print("🔍 Checking Existing Videos")
    print("=" * 50)
    
    output_dir = Path("output/generated_videos")
    if output_dir.exists():
        video_files = list(output_dir.glob("*.mp4"))
        print(f"Found {len(video_files)} existing videos:")
        
        for video_file in video_files:
            file_size = video_file.stat().st_size
            print(f"  - {video_file.name}")
            print(f"    📊 Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            print(f"    📁 Path: {video_file}")
            print()
    else:
        print("No output directory found")
    
    return video_files

def generate_new_video(test_name, prompt, negative_prompt, duration, width, height):
    """Generate a new video with specific parameters"""
    print(f"🎬 Generating: {test_name}")
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
            print(f"   💡 Note: Video generation is in progress. Check Runware dashboard for status.")
            print(f"   🔗 Dashboard: https://runware.ai/dashboard")
            return {
                'success': True,
                'task_uuid': task_uuid,
                'message': 'Request submitted successfully. Video generation is in progress.'
            }
        else:
            print(f"   ❌ API Error: {response.status_code} - {response.text}")
            return {'success': False, 'error': f"API Error: {response.status_code}"}
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Main function"""
    print("🚀 Runware Video Generation Status Check")
    print("=" * 60)
    
    # Check existing videos
    existing_videos = check_existing_videos()
    
    # Generate new videos with different parameters
    print("\n🎬 Generating New Videos with Different Parameters")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Test_2_Portrait_10s',
            'prompt': 'A bird flying through mountain peaks with dramatic clouds',
            'negative_prompt': 'ugly, distorted, bad quality, pixelated',
            'duration': 10,
            'width': 1080,
            'height': 1920
        },
        {
            'name': 'Test_3_Square_5s',
            'prompt': 'A thunderstorm with lightning flashing across the sky',
            'negative_prompt': 'calm, peaceful, sunny',
            'duration': 5,
            'width': 1080,
            'height': 1080
        },
        {
            'name': 'Test_4_Landscape_10s',
            'prompt': 'A serene beach at sunset with gentle waves and seagulls flying',
            'negative_prompt': 'stormy, rough, dark, night time',
            'duration': 10,
            'width': 1920,
            'height': 1080
        }
    ]
    
    results = []
    for test_case in test_cases:
        result = generate_new_video(
            test_case['name'],
            test_case['prompt'],
            test_case['negative_prompt'],
            test_case['duration'],
            test_case['width'],
            test_case['height']
        )
        results.append({
            'name': test_case['name'],
            'result': result
        })
        print()
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Existing videos: {len(existing_videos)}")
    print(f"New requests submitted: {len(results)}")
    
    successful_requests = sum(1 for r in results if r['result'].get('success', False))
    print(f"Successful requests: {successful_requests}")
    
    if successful_requests > 0:
        print(f"\n💡 Next steps:")
        print(f"1. Check Runware dashboard: https://runware.ai/dashboard")
        print(f"2. Wait for videos to complete (usually 1-3 minutes)")
        print(f"3. Use poll_and_download.py to check specific task UUIDs")
        print(f"4. Videos will appear in output/generated_videos/ when ready")
    
    return results

if __name__ == "__main__":
    main()
