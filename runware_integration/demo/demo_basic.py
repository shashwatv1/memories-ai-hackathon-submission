#!/usr/bin/env python3
"""
Basic Runware Video Generation Demo

This script demonstrates basic video generation using Runware's API.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from runware_client import RunwareVideoClient


def demo_basic_generation():
    """Demonstrate basic video generation"""
    print("🎬 Runware Video Generation Demo")
    print("=" * 50)
    
    try:
        # Initialize client
        client = RunwareVideoClient()
        print("✅ Client initialized successfully")
        
        # Test connection first
        print("\n🔌 Testing API connection...")
        connection_result = client.test_connection()
        
        if not connection_result['success']:
            print("❌ API connection failed!")
            print(f"Error: {connection_result.get('error', 'Unknown error')}")
            return False
        
        print("✅ API connection successful!")
        
        # Generate a video with a good prompt
        excellent_prompt = """A serene beach at sunset, with gentle waves lapping at the shore. 
        A young woman in a flowing white dress walks barefoot along the water's edge, 
        leaving footprints in the wet sand. The sky is painted in hues of orange and pink, 
        and seagulls fly overhead. The camera follows her from behind, capturing the peaceful 
        ambiance and the rhythmic sound of the waves."""
        
        print(f"\n🎯 Generating video with prompt:")
        print(f"   '{excellent_prompt}'")
        print()
        
        # Generate video
        result = client.generate_video(
            image_path='test_image.jpg',  # We'll create this
            prompt=excellent_prompt,
            num_frames=14,
            fps=6,
            motion=1.0,
            return_video=True,
            output_filename='demo_runware_video.mp4'
        )
        
        if result['success']:
            print("🎉 Video generated successfully!")
            print(f"📁 Saved to: {result['video_path']}")
            print(f"📊 File size: {result['file_size']:,} bytes ({result['file_size']/1024/1024:.2f} MB)")
            print(f"⏱️  Duration: {result['duration']} seconds")
            print(f"🔗 Video URL: {result['video_url']}")
            
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
        print(f"❌ Demo failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_parameter_variations():
    """Demonstrate different parameter combinations"""
    print("\n🎛️  Parameter Variations Demo")
    print("=" * 50)
    
    try:
        client = RunwareVideoClient()
        
        variations = [
            {
                "name": "Calm Scene",
                "prompt": "A peaceful garden with butterflies gently floating around colorful flowers",
                "motion": 0.3,
                "fps": 6,
                "filename": "demo_calm_scene.mp4"
            },
            {
                "name": "Dynamic Scene",
                "prompt": "A thunderstorm with lightning flashing across dark clouds and rain falling heavily",
                "motion": 1.5,
                "fps": 12,
                "filename": "demo_dynamic_scene.mp4"
            },
            {
                "name": "Smooth Motion",
                "prompt": "A bird soaring gracefully through mountain peaks with clouds drifting below",
                "motion": 1.0,
                "fps": 24,
                "filename": "demo_smooth_motion.mp4"
            }
        ]
        
        for i, variation in enumerate(variations, 1):
            print(f"\n🎬 Variation {i}: {variation['name']}")
            print(f"   Prompt: {variation['prompt']}")
            print(f"   Motion: {variation['motion']}, FPS: {variation['fps']}")
            
            result = client.generate_video(
                image_path='test_image.jpg',
                prompt=variation['prompt'],
                motion=variation['motion'],
                fps=variation['fps'],
                return_video=True,
                output_filename=variation['filename']
            )
            
            if result['success']:
                print(f"✅ {variation['name']} generated successfully!")
                print(f"📁 Saved to: {result['video_path']}")
                print(f"📊 Size: {result['file_size']/1024/1024:.2f} MB")
            else:
                print(f"❌ {variation['name']} failed!")
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter demo failed: {e}")
        return False


def main():
    """Run the demo"""
    print("🚀 Runware Video Generation Demo")
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
    
    # Run demos
    demos = [
        ("Basic Generation", demo_basic_generation),
        ("Parameter Variations", demo_parameter_variations)
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*60}")
        print(f"🎬 {demo_name}")
        print("=" * 60)
        
        try:
            success = demo_func()
            if success:
                print(f"✅ {demo_name} completed successfully!")
            else:
                print(f"❌ {demo_name} failed!")
        except Exception as e:
            print(f"❌ {demo_name} error: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 Demo completed!")
    print("💡 Check the 'output/generated_videos' directory for your videos")


if __name__ == "__main__":
    main()
