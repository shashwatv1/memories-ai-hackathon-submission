#!/usr/bin/env python3
"""
Poll Runware Task and Download Video

This script polls the Runware API for task completion and downloads the video when ready.
"""

import requests
import json
import uuid
import time
from pathlib import Path

def poll_task_status(task_uuid, max_attempts=30, delay=10):
    """
    Poll the task status until completion
    
    Args:
        task_uuid: The task UUID to check
        max_attempts: Maximum number of polling attempts
        delay: Delay between attempts in seconds
    
    Returns:
        Dictionary with task result or None if failed
    """
    print(f"🔍 Polling task status for UUID: {task_uuid}")
    print(f"⏳ Max attempts: {max_attempts}, Delay: {delay}s")
    print("=" * 60)
    
    # API configuration
    api_token = "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
    api_base_url = "https://api.runware.ai/v1"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n🎯 Attempt {attempt}/{max_attempts}")
        
        try:
            # Use getResponse task type as found in documentation
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
            
            print(f"📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                print(f"📄 Response:")
                print(json.dumps(response_data, indent=2))
                
                # Check if task is complete
                if isinstance(response_data, dict) and 'data' in response_data:
                    data = response_data['data']
                    if isinstance(data, list) and len(data) > 0:
                        task_result = data[0]
                        
                        # Check for completion status
                        status = task_result.get('status', 'unknown')
                        print(f"📊 Task Status: {status}")
                        
                        if status == 'success':
                            print(f"✅ Task completed successfully!")
                            
                            # Look for video URL
                            video_url = None
                            for key, value in task_result.items():
                                if 'url' in key.lower() and ('video' in key.lower() or 'image' in key.lower()):
                                    video_url = value
                                    print(f"🎬 Found video URL: {key} = {value}")
                                    break
                            
                            if video_url:
                                return {
                                    'success': True,
                                    'video_url': video_url,
                                    'task_result': task_result
                                }
                            else:
                                print(f"⚠️  Task completed but no video URL found")
                                print(f"Available fields: {list(task_result.keys())}")
                                return {
                                    'success': True,
                                    'video_url': None,
                                    'task_result': task_result
                                }
                        
                        elif status == 'failed' or status == 'error':
                            print(f"❌ Task failed with status: {status}")
                            return {
                                'success': False,
                                'error': f"Task failed with status: {status}",
                                'task_result': task_result
                            }
                        
                        else:
                            print(f"⏳ Task still processing (status: {status})")
                            if attempt < max_attempts:
                                print(f"⏰ Waiting {delay} seconds before next attempt...")
                                time.sleep(delay)
                            continue
                
                print(f"⚠️  Unexpected response format")
                if attempt < max_attempts:
                    print(f"⏰ Waiting {delay} seconds before next attempt...")
                    time.sleep(delay)
                    continue
            
            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                if attempt < max_attempts:
                    print(f"⏰ Waiting {delay} seconds before next attempt...")
                    time.sleep(delay)
                    continue
        
        except Exception as e:
            print(f"❌ Error during polling: {e}")
            if attempt < max_attempts:
                print(f"⏰ Waiting {delay} seconds before next attempt...")
                time.sleep(delay)
                continue
    
    print(f"⏰ Max attempts reached. Task may still be processing.")
    return None

def download_video(video_url, output_path):
    """Download video from URL to local file"""
    print(f"📥 Downloading video from: {video_url}")
    print(f"📁 Saving to: {output_path}")
    
    try:
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = output_path.stat().st_size
        print(f"✅ Video downloaded successfully!")
        print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    """Main function to poll and download video"""
    print("🚀 Runware Video Polling and Download")
    print("=" * 60)
    
    # Use the task UUID from our previous test
    task_uuid = "953dda5b-9b5a-45a0-b92f-7ddfe08bbe0b"
    
    # Poll for task completion
    result = poll_task_status(task_uuid, max_attempts=20, delay=15)
    
    if result and result.get('success') and result.get('video_url'):
        print(f"\n🎉 Video generation completed!")
        print(f"🔗 Video URL: {result['video_url']}")
        
        # Download the video
        output_dir = Path("output/generated_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"runware_video_{timestamp}.mp4"
        output_path = output_dir / output_filename
        
        if download_video(result['video_url'], output_path):
            print(f"\n✅ Video saved successfully to: {output_path}")
        else:
            print(f"\n❌ Failed to download video")
    
    elif result and result.get('success') and not result.get('video_url'):
        print(f"\n⚠️  Task completed but no video URL found")
        print(f"📄 Task result: {result.get('task_result', {})}")
    
    else:
        print(f"\n⚠️  Task polling failed or task not completed yet")
        print(f"💡 You can try again later or check the Runware dashboard")

if __name__ == "__main__":
    main()
