#!/usr/bin/env python3
"""
Generate Test Videos with Different Parameter Combinations

This script generates actual videos with different parameter combinations
to test the Runware 4-parameter implementation.
"""

import os
import sys
import time
import requests
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from runware_client import RunwareVideoClient
from models.video_result import VideoGenerationParams


class VideoGenerator:
    """Generate test videos with different parameter combinations"""
    
    def __init__(self):
        """Initialize the generator"""
        self.client = RunwareVideoClient()
        self.api_token = "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
        self.api_base_url = "https://api.runware.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        self.test_image_path = self._create_test_image()
        self.generated_videos = []
        
    def _create_test_image(self) -> Path:
        """Create a simple test image for testing"""
        try:
            from PIL import Image
            
            # Create a simple test image
            img = Image.new('RGB', (512, 512), color='lightblue')
            test_path = Path("test_image_generator.jpg")
            img.save(test_path)
            print(f"✅ Created test image: {test_path}")
            return test_path
            
        except ImportError:
            print("⚠️  PIL not available, using existing image if available")
            # Look for any existing image file
            for ext in ['.jpg', '.jpeg', '.png']:
                for img_file in Path('.').glob(f'*{ext}'):
                    if img_file.exists():
                        print(f"✅ Using existing image: {img_file}")
                        return img_file
            
            # Create a dummy path for testing (will fail gracefully)
            return Path("dummy_test_image.jpg")
    
    def generate_test_videos(self) -> Dict[str, Any]:
        """Generate test videos with different parameter combinations"""
        print("🎬 Generating Test Videos with Different Parameter Combinations")
        print("=" * 70)
        
        test_cases = [
            {
                'name': 'Test 1: Basic 5s Landscape Video',
                'description': 'Simple prompt, default negative prompt, 5s duration, landscape resolution',
                'params': {
                    'prompt': 'A beautiful sunset over mountains',
                    'negative_prompt': None,  # Use default
                    'duration': 5,
                    'width': 1920,
                    'height': 1080
                }
            },
            {
                'name': 'Test 2: Portrait 10s Video with Custom Negative Prompt',
                'description': 'Detailed prompt, custom negative prompt, 10s duration, portrait resolution',
                'params': {
                    'prompt': 'A bird flying through mountain peaks with dramatic clouds',
                    'negative_prompt': 'blurry, dark, low quality, pixelated',
                    'duration': 10,
                    'width': 1080,
                    'height': 1920
                }
            },
            {
                'name': 'Test 3: Square 5s Video with Action Prompt',
                'description': 'Action prompt, custom negative prompt, 5s duration, square resolution',
                'params': {
                    'prompt': 'A thunderstorm with lightning flashing across the sky',
                    'negative_prompt': 'calm, peaceful, sunny',
                    'duration': 5,
                    'width': 1080,
                    'height': 1080
                }
            },
            {
                'name': 'Test 4: Landscape 10s Video with Cinematic Prompt',
                'description': 'Cinematic prompt, detailed negative prompt, 10s duration, landscape resolution',
                'params': {
                    'prompt': 'A serene beach at sunset with gentle waves and seagulls flying',
                    'negative_prompt': 'stormy, rough, dark, night time',
                    'duration': 10,
                    'width': 1920,
                    'height': 1080
                }
            }
        ]
        
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"\n🎬 {test_case['name']}")
            print(f"📝 Description: {test_case['description']}")
            print("-" * 50)
            
            try:
                result = self._generate_single_video(test_case)
                results.append(result)
                
                if result['success']:
                    print(f"✅ {test_case['name']}: SUCCESS")
                    print(f"   📁 File: {result['file_path']}")
                    print(f"   📊 Size: {result['file_size']:,} bytes ({result['file_size']/1024/1024:.2f} MB)")
                else:
                    print(f"❌ {test_case['name']}: FAILED - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ {test_case['name']}: ERROR - {e}")
                results.append({
                    'name': test_case['name'],
                    'success': False,
                    'error': str(e)
                })
        
        return self._generate_summary(results)
    
    def _generate_single_video(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single video with the given parameters"""
        name = test_case['name']
        params = test_case['params']
        
        print(f"   📝 Prompt: {params['prompt']}")
        print(f"   🚫 Negative: {params['negative_prompt'] or 'default'}")
        print(f"   ⏱️  Duration: {params['duration']}s")
        print(f"   📐 Resolution: {params['width']}x{params['height']}")
        
        # Step 1: Submit generation request
        print("   🔄 Submitting video generation request...")
        task_uuid = self._submit_generation_request(params)
        
        if not task_uuid:
            return {
                'name': name,
                'success': False,
                'error': 'Failed to get task UUID'
            }
        
        print(f"   📋 Task UUID: {task_uuid}")
        
        # Step 2: Poll for completion
        print("   🔍 Polling for completion...")
        poll_result = self._poll_task_completion(task_uuid)
        
        if not poll_result['success']:
            return {
                'name': name,
                'success': False,
                'error': f"Polling failed: {poll_result.get('error', 'Unknown error')}"
            }
        
        video_url = poll_result['video_url']
        print(f"   🎬 Video URL: {video_url}")
        
        # Step 3: Download video
        print("   📥 Downloading video...")
        download_result = self._download_video(video_url, name, params)
        
        if not download_result['success']:
            return {
                'name': name,
                'success': False,
                'error': f"Download failed: {download_result.get('error', 'Unknown error')}"
            }
        
        return {
            'name': name,
            'success': True,
            'task_uuid': task_uuid,
            'video_url': video_url,
            'file_path': download_result['file_path'],
            'file_size': download_result['file_size'],
            'params': params
        }
    
    def _submit_generation_request(self, params: Dict[str, Any]) -> Optional[str]:
        """Submit video generation request and return task UUID"""
        try:
            task_uuid = str(uuid.uuid4())
            
            request_data = [{
                "taskType": "videoInference",
                "taskUUID": task_uuid,
                "positivePrompt": params['prompt'],
                "negativePrompt": params['negative_prompt'] or "blurry, low quality, distorted",
                "model": "klingai:5@3",
                "duration": params['duration'],
                "width": params['width'],
                "height": params['height'],
                "deliveryMethod": "async"
            }]
            
            response = requests.post(
                self.api_base_url,
                json=request_data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"   📊 API Response: {response.status_code}")
                return task_uuid
            else:
                print(f"   ❌ API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"   ❌ Request Error: {e}")
            return None
    
    def _poll_task_completion(self, task_uuid: str, max_attempts: int = 20, delay: int = 15) -> Dict[str, Any]:
        """Poll task status until completion"""
        print(f"   ⏳ Polling task {task_uuid} (max {max_attempts} attempts, {delay}s delay)")
        
        for attempt in range(1, max_attempts + 1):
            try:
                request_data = [{
                    "taskType": "getResponse",
                    "taskUUID": task_uuid
                }]
                
                response = requests.post(
                    self.api_base_url,
                    json=request_data,
                    headers=self.headers,
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
                                        'video_url': video_url,
                                        'task_result': task_result
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
                
                print(f"   ⚠️  Attempt {attempt}: Unexpected response format")
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
    
    def _download_video(self, video_url: str, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download video and save to file"""
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
            print(f"   📥 Downloading to: {file_path}")
            response = requests.get(video_url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = file_path.stat().st_size
            print(f"   📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            
            self.generated_videos.append({
                'name': name,
                'file_path': str(file_path),
                'file_size': file_size,
                'params': params
            })
            
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
    
    def _generate_summary(self, results: list) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get('success', False))
        failed_tests = total_tests - successful_tests
        
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if successful_tests > 0:
            print(f"\n🎬 Generated Videos:")
            for video in self.generated_videos:
                print(f"  - {video['name']}: {video['file_size']:,} bytes")
                print(f"    📁 Path: {video['file_path']}")
                print(f"    📝 Prompt: {video['params']['prompt']}")
                print(f"    🚫 Negative: {video['params']['negative_prompt'] or 'default'}")
                print(f"    ⏱️  Duration: {video['params']['duration']}s")
                print(f"    📐 Resolution: {video['params']['width']}x{video['params']['height']}")
                print()
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in results:
                if not result.get('success', False):
                    print(f"  - {result['name']}: {result.get('error', 'Unknown error')}")
        
        return {
            'total_tests': total_tests,
            'successful': successful_tests,
            'failed': failed_tests,
            'success_rate': (successful_tests/total_tests)*100,
            'generated_videos': self.generated_videos,
            'results': results
        }


def main():
    """Main test execution"""
    print("🚀 Runware 4-Parameter Video Generation Test")
    print("Generating videos with different parameter combinations")
    print("=" * 70)
    
    generator = VideoGenerator()
    summary = generator.generate_test_videos()
    
    # Clean up test image
    if generator.test_image_path.exists() and generator.test_image_path.name == "test_image_generator.jpg":
        generator.test_image_path.unlink()
        print(f"\n🧹 Cleaned up test image: {generator.test_image_path}")
    
    return summary


if __name__ == "__main__":
    main()
