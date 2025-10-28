#!/usr/bin/env python3
"""
Test Hume AI SDK Properly

This script tests the official Hume AI SDK to see what API calls are made.
"""

import os
import sys
from hume import HumeClient
from hume.models import TextToSpeech


def test_hume_sdk():
    """Test Hume AI SDK with proper imports"""
    print("🧪 Testing Hume AI SDK")
    print("=" * 50)
    
    try:
        # Set API key
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        print(f"🔑 Using API key: {api_key[:20]}...")
        
        # Initialize client
        print("🔧 Initializing Hume client...")
        client = HumeClient(api_key=api_key)
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
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empathic_voice():
    """Test Hume AI Empathic Voice Interface"""
    print(f"\n🎭 Testing Empathic Voice Interface")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        
        # Test empathic voice
        from hume.models import EmpathicVoiceInterface
        
        print("🔧 Initializing Empathic Voice Interface...")
        evi = EmpathicVoiceInterface(client)
        print("✅ EVI initialized!")
        
        # Test voice generation with emotion
        text = "I am feeling very happy today!"
        print(f"📝 Text: {text}")
        
        print("🔄 Generating empathic voice...")
        response = evi.synthesize(text=text)
        print("✅ Empathic voice generated successfully!")
        
        # Save audio file
        output_file = "test_empathic_output.wav"
        print(f"💾 Saving audio to {output_file}...")
        with open(output_file, "wb") as audio_file:
            audio_file.write(response.content)
        print(f"✅ Audio saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during empathic voice testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice_parameters():
    """Test different voice parameters"""
    print(f"\n🎛️ Testing Voice Parameters")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        tts = TextToSpeech(client)
        
        # Test different parameters
        test_cases = [
            {
                'name': 'Basic TTS',
                'text': 'Hello, this is a basic test.',
                'params': {}
            },
            {
                'name': 'Emotional TTS',
                'text': 'I am feeling very happy and excited!',
                'params': {'emotion': 'happy'}
            },
            {
                'name': 'Calm TTS',
                'text': 'Take a deep breath and relax.',
                'params': {'emotion': 'calm'}
            },
            {
                'name': 'Dramatic TTS',
                'text': 'The storm approaches with thunderous force!',
                'params': {'emotion': 'dramatic'}
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🧪 Testing: {test_case['name']}")
            print(f"📝 Text: {test_case['text']}")
            print(f"🎛️ Params: {test_case['params']}")
            
            try:
                response = tts.synthesize(
                    text=test_case['text'],
                    **test_case['params']
                )
                
                output_file = f"test_{test_case['name'].lower().replace(' ', '_')}.wav"
                with open(output_file, "wb") as audio_file:
                    audio_file.write(response.content)
                
                print(f"✅ Generated: {output_file}")
                print(f"📊 Size: {len(response.content)} bytes")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during parameter testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Hume AI SDK Testing")
    print("=" * 60)
    
    # Test basic SDK
    success1 = test_hume_sdk()
    
    # Test empathic voice
    success2 = test_empathic_voice()
    
    # Test voice parameters
    success3 = test_voice_parameters()
    
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print(f"Basic TTS: {'✅ Success' if success1 else '❌ Failed'}")
    print(f"Empathic Voice: {'✅ Success' if success2 else '❌ Failed'}")
    print(f"Voice Parameters: {'✅ Success' if success3 else '❌ Failed'}")
    
    if success1 or success2 or success3:
        print(f"\n🎉 Hume AI SDK is working!")
        print(f"📁 Generated files:")
        for file in os.listdir("."):
            if file.endswith(".wav"):
                print(f"  - {file}")
    else:
        print(f"\n❌ All tests failed. Check the errors above.")
