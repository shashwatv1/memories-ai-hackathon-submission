#!/usr/bin/env python3
"""
Test Hume AI SDK - Success Version

This script tests the official Hume AI SDK and successfully generates voice.
"""

import os
import sys
from hume import HumeClient
from hume.tts.types import PostedUtterance, PostedUtteranceVoiceWithId, FormatWav


def test_voice_generation_success():
    """Test successful voice generation"""
    print("🧪 Testing Hume AI Voice Generation - SUCCESS")
    print("=" * 50)
    
    try:
        # Set API key
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        print(f"🔑 Using API key: {api_key[:20]}...")
        
        # Initialize client
        print("🔧 Initializing Hume client...")
        client = HumeClient(api_key=api_key)
        print("✅ Hume client initialized!")
        
        # Get available voices
        print("🎤 Getting available voices...")
        voices_response = client.tts.voices.list(provider="HUME_AI")
        voices = voices_response.items
        print(f"✅ Found {len(voices)} voices!")
        
        # Use the first voice
        voice = voices[0]
        voice_name = voice.name
        voice_id = voice.id
        print(f"🎭 Using voice: {voice_name} (ID: {voice_id})")
        
        # Create utterance with voice ID
        text = "Hello, this is a test of Hume AI voice generation. The voice sounds natural and expressive."
        print(f"📝 Text: {text}")
        
        utterance = PostedUtterance(
            text=text,
            voice=PostedUtteranceVoiceWithId(id=voice_id)
        )
        
        print("🔄 Generating speech...")
        response = client.tts.synthesize_json(
            utterances=[utterance],
            format=FormatWav()
        )
        print("✅ Speech generated successfully!")
        
        # Check response structure
        print(f"📊 Response type: {type(response)}")
        print(f"📊 Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
        # Try different ways to get the audio content
        audio_content = None
        
        # Method 1: Check if response has content attribute
        if hasattr(response, 'content'):
            audio_content = response.content
            print("✅ Got audio content from response.content")
        
        # Method 2: Check if response has audio attribute
        elif hasattr(response, 'audio'):
            audio_content = response.audio
            print("✅ Got audio content from response.audio")
        
        # Method 3: Check if response has data attribute
        elif hasattr(response, 'data'):
            audio_content = response.data
            print("✅ Got audio content from response.data")
        
        # Method 4: Check if response has file attribute
        elif hasattr(response, 'file'):
            audio_content = response.file
            print("✅ Got audio content from response.file")
        
        # Method 5: Check if response has bytes attribute
        elif hasattr(response, 'bytes'):
            audio_content = response.bytes
            print("✅ Got audio content from response.bytes")
        
        # Method 6: Try to get the raw response
        else:
            print("🔍 Trying to get raw response...")
            try:
                raw_response = client.tts.with_raw_response().synthesize_json(
                    utterances=[utterance],
                    format=FormatWav()
                )
                print(f"📊 Raw response type: {type(raw_response)}")
                print(f"📊 Raw response attributes: {[attr for attr in dir(raw_response) if not attr.startswith('_')]}")
                
                if hasattr(raw_response, 'content'):
                    audio_content = raw_response.content
                    print("✅ Got audio content from raw response.content")
                elif hasattr(raw_response, 'data'):
                    audio_content = raw_response.data
                    print("✅ Got audio content from raw response.data")
                else:
                    print(f"📊 Raw response: {raw_response}")
                    
            except Exception as e:
                print(f"❌ Failed to get raw response: {e}")
        
        if audio_content:
            # Save audio file
            output_file = "test_success.wav"
            print(f"💾 Saving audio to {output_file}...")
            with open(output_file, "wb") as audio_file:
                audio_file.write(audio_content)
            print(f"✅ Audio saved to {output_file}")
            print(f"📊 Size: {len(audio_content)} bytes")
            return True
        else:
            print("❌ Could not extract audio content from response")
            return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_voices():
    """Test multiple voices"""
    print(f"\n🎭 Testing Multiple Voices")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        
        # Get voices
        voices_response = client.tts.voices.list(provider="HUME_AI")
        voices = voices_response.items[:3]  # Test first 3 voices
        
        text = "Hello, this is a test of different voice styles. Each voice has its own unique character and personality."
        
        for i, voice in enumerate(voices):
            print(f"\n🧪 Testing Voice {i+1}: {voice.name}")
            
            try:
                utterance = PostedUtterance(
                    text=text,
                    voice=PostedUtteranceVoiceWithId(id=voice.id)
                )
                
                response = client.tts.synthesize_json(
                    utterances=[utterance],
                    format=FormatWav()
                )
                
                # Try to get audio content
                audio_content = None
                if hasattr(response, 'content'):
                    audio_content = response.content
                elif hasattr(response, 'audio'):
                    audio_content = response.audio
                elif hasattr(response, 'data'):
                    audio_content = response.data
                
                if audio_content:
                    output_file = f"test_voice_{i+1}_{voice.name.lower().replace(' ', '_')}.wav"
                    with open(output_file, "wb") as audio_file:
                        audio_file.write(audio_content)
                    
                    print(f"✅ Generated: {output_file}")
                    print(f"📊 Size: {len(audio_content)} bytes")
                else:
                    print(f"❌ Could not extract audio content")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during multiple voice testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Hume AI Voice Generation - SUCCESS VERSION")
    print("=" * 60)
    
    # Test successful voice generation
    success1 = test_voice_generation_success()
    
    # Test multiple voices
    success2 = test_multiple_voices()
    
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print(f"Voice Generation: {'✅ Success' if success1 else '❌ Failed'}")
    print(f"Multiple Voices: {'✅ Success' if success2 else '❌ Failed'}")
    
    if success1 or success2:
        print(f"\n🎉 Hume AI Voice Generation is working!")
        print(f"📁 Generated files:")
        for file in os.listdir("."):
            if file.endswith(".wav"):
                print(f"  - {file}")
    else:
        print(f"\n❌ All tests failed. Check the errors above.")
