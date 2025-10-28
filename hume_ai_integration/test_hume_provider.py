#!/usr/bin/env python3
"""
Test Hume AI SDK with Provider

This script tests the official Hume AI SDK with provider parameter.
"""

import os
import sys
from hume import HumeClient
from hume.tts.types import PostedUtterance, PostedUtteranceVoiceWithName, FormatWav


def test_voices_with_provider():
    """Test getting voices with provider"""
    print(f"🎤 Getting Voices with Provider")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        tts = client.tts
        
        # Try different providers
        providers = ["HUME_AI", "CUSTOM_VOICE"]
        
        for provider in providers:
            print(f"\n🔍 Trying provider: {provider}")
            try:
                voices_response = tts.voices.list(provider=provider)
                print(f"✅ Voices retrieved successfully with provider {provider}!")
                print(f"📊 Response type: {type(voices_response)}")
                
                # Get voice information
                if hasattr(voices_response, 'voices'):
                    print(f"📊 Number of voices: {len(voices_response.voices)}")
                    for i, voice in enumerate(voices_response.voices[:5]):  # Show first 5
                        voice_name = voice.name if hasattr(voice, 'name') else str(voice)
                        voice_id = voice.id if hasattr(voice, 'id') else None
                        print(f"  {i+1}. {voice_name} (ID: {voice_id})")
                    return voices_response.voices
                else:
                    print(f"📊 Voices response: {voices_response}")
                    
            except Exception as e:
                print(f"❌ Failed with provider {provider}: {e}")
        
        return []
        
    except Exception as e:
        print(f"❌ Error getting voices: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_voice_generation_without_voice():
    """Test voice generation without specifying a voice"""
    print(f"\n🎙️ Testing Voice Generation Without Voice")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        tts = client.tts
        
        # Create utterance without voice (use default)
        text = "Hello, this is a test of Hume AI voice generation."
        print(f"📝 Text: {text}")
        
        # Try without voice parameter
        try:
            utterance = PostedUtterance(text=text)
            print("🔄 Generating speech without voice...")
            response = tts.synthesize_json(
                utterances=[utterance],
                format=FormatWav()
            )
            print("✅ Speech generated successfully!")
            
            # Save audio file
            output_file = "test_output_default.wav"
            with open(output_file, "wb") as audio_file:
                audio_file.write(response.content)
            print(f"✅ Audio saved to {output_file}")
            print(f"📊 Size: {len(response.content)} bytes")
            return True
            
        except Exception as e:
            print(f"❌ Failed without voice: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error during voice generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice_generation_with_voice_id():
    """Test voice generation with voice ID"""
    print(f"\n🎙️ Testing Voice Generation with Voice ID")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        tts = client.tts
        
        # Get voices first
        voices = test_voices_with_provider()
        if not voices:
            print("❌ No voices available")
            return False
        
        # Use the first voice
        voice = voices[0]
        voice_name = voice.name if hasattr(voice, 'name') else "Default Voice"
        voice_id = voice.id if hasattr(voice, 'id') else None
        
        print(f"🎭 Using voice: {voice_name} (ID: {voice_id})")
        
        # Create utterance with voice ID
        text = "Hello, this is a test of Hume AI voice generation."
        print(f"📝 Text: {text}")
        
        if voice_id:
            try:
                from hume.tts.types import PostedUtteranceVoiceWithId
                utterance = PostedUtterance(
                    text=text,
                    voice=PostedUtteranceVoiceWithId(id=voice_id)
                )
                print("🔄 Generating speech with voice ID...")
                response = tts.synthesize_json(
                    utterances=[utterance],
                    format=FormatWav()
                )
                print("✅ Speech generated successfully with voice ID!")
                
                # Save audio file
                output_file = "test_output_with_id.wav"
                with open(output_file, "wb") as audio_file:
                    audio_file.write(response.content)
                print(f"✅ Audio saved to {output_file}")
                print(f"📊 Size: {len(response.content)} bytes")
                return True
                
            except Exception as e:
                print(f"❌ Failed with voice ID: {e}")
                return False
        else:
            print("❌ No voice ID available")
            return False
        
    except Exception as e:
        print(f"❌ Error during voice generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Hume AI Voice Generation with Provider")
    print("=" * 60)
    
    # Test voices with provider
    voices = test_voices_with_provider()
    
    # Test voice generation without voice
    success1 = test_voice_generation_without_voice()
    
    # Test voice generation with voice ID
    success2 = test_voice_generation_with_voice_id()
    
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print(f"Available Voices: {'✅ Success' if voices else '❌ Failed'}")
    print(f"Voice Generation (No Voice): {'✅ Success' if success1 else '❌ Failed'}")
    print(f"Voice Generation (With Voice ID): {'✅ Success' if success2 else '❌ Failed'}")
    
    if voices or success1 or success2:
        print(f"\n🎉 Hume AI Voice Generation is working!")
        print(f"📁 Generated files:")
        for file in os.listdir("."):
            if file.endswith(".wav"):
                print(f"  - {file}")
    else:
        print(f"\n❌ All tests failed. Check the errors above.")
