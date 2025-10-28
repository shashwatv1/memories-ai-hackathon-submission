#!/usr/bin/env python3
"""
Runware Video Generation Test V2

This script tests the V2 client with proper async handling.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from runware_client_v2 import RunwareVideoClientV2


def test_v2_client():
    """Test the V2 client with async handling"""
    print("🎬 Runware Video Generation Test V2")
    print("=" * 50)
    
    try:
        # Initialize client
        client = RunwareVideoClientV2()
        print("✅ Client initialized successfully")
        
        # Test connection first
        print("\n🔌 Testing API connection...")
        connection_result = client.test_connection()
        
        if not connection_result['success']:
            print("❌ API connection failed!")
            print(f"Error: {connection_result.get('error', 'Unknown error')}")
            return False
        
        print("✅ API connection successful!")
        
        # Test video generation
        print("\n🎬 Testing video generation...")
        result = client.generate_video_sync(
            image_path='test_image.jpg',
            prompt="A beautiful sunset over mountains with flowing clouds",
            return_video=False  # Don't try to download since it's async
        )
        
        if result['success']:
            print("✅ Video generation task submitted successfully!")
            print(f"📋 Task UUID: {result['task_uuid']}")
            print(f"⏳ Status: {result['status']}")
            print(f"💡 Message: {result['message']}")
            
            # Show usage stats
            stats = client.get_usage_stats()
            print(f"\n📈 Usage Statistics:")
            print(f"   Total generations: {stats['generation_count']}")
            print(f"   Estimated cost: ${stats['estimated_total_cost']:.4f}")
            print(f"   Output directory: {stats['output_directory']}")
            
            return True
        else:
            print("❌ Video generation failed!")
            error = result.get('error', 'Unknown error')
            print(f"Error: {error}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the V2 test"""
    print("🚀 Runware Video Generation Test V2")
    print("=" * 60)
    
    # Create test image if it doesn't exist
    test_image_path = Path('test_image.jpg')
    if not test_image_path.exists():
        print("⚠️  Creating placeholder test image...")
        try:
            from PIL import Image
            # Create a simple test image
            img = Image.new('RGB', (512, 512), color='lightblue')
            img.save(test_image_path)
            print(f"✅ Created test image: {test_image_path}")
        except ImportError:
            print("❌ PIL not available. Please install pillow: pip install pillow")
            return
    
    success = test_v2_client()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 V2 test completed successfully!")
        print("💡 The video generation task has been submitted to Runware")
        print("💡 Check the Runware dashboard to monitor progress")
        print("💡 The task UUID can be used to check status later")
    else:
        print("❌ V2 test failed!")
        print("💡 Check the error messages above")


if __name__ == "__main__":
    main()
