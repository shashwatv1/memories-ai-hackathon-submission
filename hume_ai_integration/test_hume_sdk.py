#!/usr/bin/env python3
"""
Test Hume AI Python SDK

This script tests the official Hume AI Python SDK to see what API calls are made.
"""

import os
import sys

def test_hume_sdk():
    """Test Hume AI Python SDK"""
    print("🧪 Testing Hume AI Python SDK")
    print("=" * 50)
    
    try:
        # Try to import Hume SDK
        print("📦 Attempting to import Hume SDK...")
        from hume import Hume
        from hume.models import TextToSpeech
        print("✅ Hume SDK imported successfully!")
        
        # Set API key
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        print(f"🔑 Using API key: {api_key[:20]}...")
        
        # Initialize client
        print("🔧 Initializing Hume client...")
        client = Hume(api_key=api_key)
        print("✅ Hume client initialized!")
        
        # Test Text-to-Speech
        print("🎙️ Testing Text-to-Speech...")
        tts = TextToSpeech(client)
        
        # Generate speech
        text = "Hello, this is a test of Hume AI voice generation."
        print(f"📝 Text: {text}")
        
        print("🔄 Generating speech...")
        response = tts.synthesize(text=text)
        print("✅ Speech generated successfully!")
        
        # Save audio file
        output_file = "test_output.wav"
        print(f"💾 Saving audio to {output_file}...")
        with open(output_file, "wb") as audio_file:
            audio_file.write(response.content)
        print(f"✅ Audio saved to {output_file}")
        
        # Show response details
        print(f"📊 Response type: {type(response)}")
        print(f"📊 Response content length: {len(response.content)} bytes")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Hume SDK: {e}")
        print("💡 Try installing with: pip install hume")
        return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False


def test_alternative_imports():
    """Test alternative import methods"""
    print(f"\n🔄 Testing Alternative Import Methods")
    print("=" * 50)
    
    # Test different import methods
    import_methods = [
        "from hume import Hume",
        "import hume",
        "from hume.ai import Hume",
        "from humeai import Hume",
        "import humeai"
    ]
    
    for method in import_methods:
        print(f"\n🧪 Testing: {method}")
        try:
            exec(method)
            print("✅ Import successful!")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")


def test_pip_install():
    """Test if we can install the Hume SDK"""
    print(f"\n📦 Testing Hume SDK Installation")
    print("=" * 50)
    
    import subprocess
    
    try:
        print("🔄 Attempting to install Hume SDK...")
        result = subprocess.run(
            ["pip", "install", "hume"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"📊 Return code: {result.returncode}")
        print(f"📄 Output: {result.stdout}")
        if result.stderr:
            print(f"❌ Error: {result.stderr}")
            
        if result.returncode == 0:
            print("✅ Hume SDK installed successfully!")
            return True
        else:
            print("❌ Failed to install Hume SDK")
            return False
            
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Hume AI SDK Testing")
    print("=" * 60)
    
    # Test SDK installation
    install_success = test_pip_install()
    
    if install_success:
        # Test SDK usage
        test_hume_sdk()
    else:
        # Test alternative imports
        test_alternative_imports()
    
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print("This test shows us:")
    print("1. Whether the Hume SDK can be installed")
    print("2. How to import and use the SDK")
    print("3. What API calls are made internally")
    print("4. Whether voice generation works")
