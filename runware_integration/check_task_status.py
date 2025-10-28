#!/usr/bin/env python3
"""
Check Runware Task Status

This script checks the status of a video generation task using the task UUID.
"""

import requests
import json
import uuid

def check_task_status(task_uuid):
    """Check the status of a video generation task"""
    print(f"🔍 Checking task status for UUID: {task_uuid}")
    print("=" * 60)
    
    # API configuration
    api_token = "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
    api_base_url = "https://api.runware.ai/v1"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Try different approaches to check task status
    approaches = [
        {
            "name": "Task Status Check",
            "data": [{
                "taskType": "getTaskStatus",
                "taskUUID": task_uuid
            }]
        },
        {
            "name": "Task Result Check", 
            "data": [{
                "taskType": "getTaskResult",
                "taskUUID": task_uuid
            }]
        },
        {
            "name": "Video Status Check",
            "data": [{
                "taskType": "videoStatus",
                "taskUUID": task_uuid
            }]
        }
    ]
    
    for approach in approaches:
        print(f"\n🎯 Trying: {approach['name']}")
        try:
            response = requests.post(
                api_base_url,
                json=approach['data'],
                headers=headers,
                timeout=30
            )
            
            print(f"📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                print(f"✅ Success! Response:")
                print(json.dumps(response_data, indent=2))
                
                # Check if there's a video URL in the response
                if isinstance(response_data, dict) and 'data' in response_data:
                    data = response_data['data']
                    if isinstance(data, list) and len(data) > 0:
                        task_result = data[0]
                        for key, value in task_result.items():
                            if 'url' in key.lower() or 'video' in key.lower():
                                print(f"🎬 Found video reference: {key} = {value}")
                                return value
                
                return response_data
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n⚠️  Could not check task status using standard methods")
    print(f"💡 Task UUID: {task_uuid}")
    print(f"💡 Check Runware dashboard manually for task status")
    return None

def main():
    """Check the status of our generated task"""
    print("🚀 Runware Task Status Checker")
    print("=" * 60)
    
    # Use the task UUID from our last test
    task_uuid = "953dda5b-9b5a-45a0-b92f-7ddfe08bbe0b"
    
    result = check_task_status(task_uuid)
    
    if result:
        print(f"\n✅ Task status retrieved successfully!")
    else:
        print(f"\n⚠️  Could not retrieve task status automatically")
        print(f"💡 The task may still be processing")
        print(f"💡 Check the Runware dashboard at https://runware.ai for status")

if __name__ == "__main__":
    main()
