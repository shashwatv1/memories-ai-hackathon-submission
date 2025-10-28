#!/usr/bin/env python3
"""
Basic Runware Video Generation Test

This script tests the basic functionality of the Runware video generation client.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from runware_client import RunwareVideoClient


def test_connection():
    """Test API connection"""
    print("🔌 Testing Runware API Connection")
    print("=" * 40)
    
    try:
        client = RunwareVideoClient()
        result = client.test_connection()
        
        if result['success']:
            print("✅ API connection successful!")
            print(f"   Status: {result.get('status_code', 'Unknown')}")
        else:
            print("❌ API connection failed!")
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def test_basic_generation():
    """Test basic video generation"""
    print("\n🎬 Testing Basic Video Generation")
    print("=" * 40)
    
    try:
        client = RunwareVideoClient()
        
        # Test with a simple prompt
        test_prompt = "A beautiful sunset over mountains with flowing clouds"
        
        print(f"🎯 Generating video with prompt: '{test_prompt}'")
        
        result = client.generate_video(
            image_path='test_image.jpg',  # We'll create this
            prompt=test_prompt,
            num_frames=14,
            fps=6,
            motion=0.5,
            return_video=True,
            output_filename='test_basic_video.mp4'
        )
        
        if result['success']:
            print("✅ Video generation successful!")
            print(f"📁 Video saved to: {result['video_path']}")
            print(f"📊 File size: {result['file_size']:,} bytes ({result['file_size']/1024/1024:.2f} MB)")
            print(f"⏱️  Duration: {result['duration']} seconds")
            print(f"🔗 Video URL: {result['video_url']}")
            return True
        else:
            print("❌ Video generation failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_variations():
    """Test different parameter combinations"""
    print("\n🎛️  Testing Parameter Variations")
    print("=" * 40)
    
    try:
        client = RunwareVideoClient()
        
        test_cases = [
            {
                "name": "Low Motion",
                "prompt": "A calm lake with gentle ripples",
                "motion": 0.3,
                "fps": 6,
                "filename": "test_low_motion.mp4"
            },
            {
                "name": "High Motion", 
                "prompt": "A fast-moving river with rapids",
                "motion": 1.5,
                "fps": 12,
                "filename": "test_high_motion.mp4"
            },
            {
                "name": "High FPS",
                "prompt": "A bird flying through the sky",
                "motion": 1.0,
                "fps": 24,
                "filename": "test_high_fps.mp4"
            }
        ]
        
        success_count = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🎬 Test {i}: {test_case['name']}")
            print(f"   Prompt: {test_case['prompt']}")
            print(f"   Motion: {test_case['motion']}, FPS: {test_case['fps']}")
            
            result = client.generate_video(
                image_path='test_image.jpg',
                prompt=test_case['prompt'],
                motion=test_case['motion'],
                fps=test_case['fps'],
                return_video=True,
                output_filename=test_case['filename']
            )
            
            if result['success']:
                print(f"✅ {test_case['name']} successful!")
                print(f"📁 Saved to: {result['video_path']}")
                success_count += 1
            else:
                print(f"❌ {test_case['name']} failed!")
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        print(f"\n📊 Parameter Test Results: {success_count}/{len(test_cases)} successful")
        return success_count == len(test_cases)
        
    except Exception as e:
        print(f"❌ Parameter test failed: {e}")
        return False


def test_usage_tracking():
    """Test usage tracking and statistics"""
    print("\n📈 Testing Usage Tracking")
    print("=" * 40)
    
    try:
        client = RunwareVideoClient()
        
        # Get initial stats
        initial_stats = client.get_usage_stats()
        print(f"📊 Initial stats: {initial_stats['generation_count']} generations, ${initial_stats['estimated_total_cost']:.4f} cost")
        
        # List generated videos
        videos = client.list_generated_videos()
        print(f"📚 Generated videos: {len(videos)} total")
        
        for video in videos[:3]:  # Show first 3
            print(f"   - {video['filename']} ({video['size_mb']} MB)")
        
        if len(videos) > 3:
            print(f"   ... and {len(videos) - 3} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Usage tracking test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Runware Video Generation Test Suite")
    print("=" * 50)
    
    # Create test image if it doesn't exist
    test_image_path = Path('test_image.jpg')
    if not test_image_path.exists():
        print("⚠️  Creating placeholder test image...")
        from PIL import Image
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color='lightblue')
        img.save(test_image_path)
        print(f"✅ Created test image: {test_image_path}")
    
    tests = [
        ("API Connection", test_connection),
        ("Basic Generation", test_basic_generation),
        ("Parameter Variations", test_parameter_variations),
        ("Usage Tracking", test_usage_tracking)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Runware integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()
