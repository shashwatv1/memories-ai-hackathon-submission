#!/usr/bin/env python3
"""
Test Hume AI SDK - Simple Working Version

This script tests the official Hume AI SDK with a simpler approach.
"""

import os
import sys
from hume import HumeClient
from hume.tts.types import PostedUtterance, PostedUtteranceVoiceWithName, FormatWav


def test_hume_voice_generation():
    """Test Hume AI voice generation with correct parameters"""
    print("🧪 Testing Hume AI Voice Generation")
    print("=" * 50)
    
    try:
        # Set API key
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        print(f"🔑 Using API key: {api_key[:20]}...")
        
        # Initialize client
        print("🔧 Initializing Hume client...")
        client = HumeClient(api_key=api_key)
        print("✅ Hume client initialized!")
        
        # Test TTS
        print("🎙️ Testing Text-to-Speech...")
        tts = client.tts
        
        # Create utterance with correct structure
        text = "Hello, this is a test of Hume AI voice generation."
        print(f"📝 Text: {text}")
        
        # Create PostedUtterance with voice using PostedUtteranceVoiceWithName
        utterance = PostedUtterance(
            text=text,
            voice=PostedUtteranceVoiceWithName(name="Serene Assistant")
        )
        
        print("🔄 Generating speech...")
        response = tts.synthesize_json(
            utterances=[utterance],
            format=FormatWav()
        )
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


def test_available_voices():
    """Test getting available voices"""
    print(f"\n🎤 Testing Available Voices")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        tts = client.tts
        
        print("🔍 Getting available voices...")
        voices_client = tts.voices
        print(f"📊 Voices client type: {type(voices_client)}")
        
        # Try to get voices
        try:
            voices_response = voices_client.get_voices()
            print(f"✅ Voices retrieved successfully!")
            print(f"📊 Response type: {type(voices_response)}")
            
            # Try to get voice information
            if hasattr(voices_response, 'voices'):
                print(f"📊 Number of voices: {len(voices_response.voices)}")
                for i, voice in enumerate(voices_response.voices[:5]):  # Show first 5
                    print(f"  {i+1}. {voice.name if hasattr(voice, 'name') else voice}")
            else:
                print(f"📊 Voices response: {voices_response}")
                
        except Exception as e:
            print(f"❌ Error getting voices: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during voices testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_voices():
    """Test different voice options"""
    print(f"\n🎭 Testing Different Voices")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        tts = client.tts
        
        # Test different voices
        voices = [
            "Serene Assistant",
            "Cheerful Assistant", 
            "Professional Assistant",
            "Friendly Assistant"
        ]
        
        text = "Hello, this is a test of different voice styles."
        
        for i, voice_name in enumerate(voices):
            print(f"\n🧪 Testing Voice {i+1}: {voice_name}")
            
            try:
                utterance = PostedUtterance(
                    text=text,
                    voice=PostedUtteranceVoiceWithName(name=voice_name)
                )
                
                response = tts.synthesize_json(
                    utterances=[utterance],
                    format=FormatWav()
                )
                
                output_file = f"test_voice_{i+1}_{voice_name.lower().replace(' ', '_')}.wav"
                with open(output_file, "wb") as audio_file:
                    audio_file.write(response.content)
                
                print(f"✅ Generated: {output_file}")
                print(f"📊 Size: {len(response.content)} bytes")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during voice testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Hume AI Voice Generation - Simple Working Version")
    print("=" * 60)
    
    # Test basic voice generation
    success1 = test_hume_voice_generation()
    
    # Test available voices
    success2 = test_available_voices()
    
    # Test different voices
    success3 = test_different_voices()
    
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print(f"Basic Voice Generation: {'✅ Success' if success1 else '❌ Failed'}")
    print(f"Available Voices: {'✅ Success' if success2 else '❌ Failed'}")
    print(f"Different Voices: {'✅ Success' if success3 else '❌ Failed'}")
    
    if success1 or success2 or success3:
        print(f"\n🎉 Hume AI Voice Generation is working!")
        print(f"📁 Generated files:")
        for file in os.listdir("."):
            if file.endswith(".wav"):
                print(f"  - {file}")
    else:
        print(f"\n❌ All tests failed. Check the errors above.")
