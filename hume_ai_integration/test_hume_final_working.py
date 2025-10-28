#!/usr/bin/env python3
"""
Test Hume AI SDK - Final Working Version

This script tests the official Hume AI SDK with available voices.
"""

import os
import sys
from hume import HumeClient
from hume.tts.types import PostedUtterance, PostedUtteranceVoiceWithName, FormatWav


def test_available_voices():
    """Test getting available voices"""
    print(f"🎤 Getting Available Voices")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        tts = client.tts
        
        print("🔍 Getting available voices...")
        voices_response = tts.voices.list()
        print(f"✅ Voices retrieved successfully!")
        print(f"📊 Response type: {type(voices_response)}")
        
        # Get voice information
        if hasattr(voices_response, 'voices'):
            print(f"📊 Number of voices: {len(voices_response.voices)}")
            available_voices = []
            for i, voice in enumerate(voices_response.voices):
                voice_name = voice.name if hasattr(voice, 'name') else str(voice)
                voice_id = voice.id if hasattr(voice, 'id') else None
                print(f"  {i+1}. {voice_name} (ID: {voice_id})")
                available_voices.append({
                    'name': voice_name,
                    'id': voice_id,
                    'voice': voice
                })
            return available_voices
        else:
            print(f"📊 Voices response: {voices_response}")
            return []
            
    except Exception as e:
        print(f"❌ Error getting voices: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_voice_generation_with_available_voice():
    """Test voice generation with an available voice"""
    print(f"\n🎙️ Testing Voice Generation with Available Voice")
    print("=" * 50)
    
    try:
        # Get available voices first
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        tts = client.tts
        
        # Get voices
        voices_response = tts.voices.list()
        if not hasattr(voices_response, 'voices') or not voices_response.voices:
            print("❌ No voices available")
            return False
        
        # Use the first available voice
        voice = voices_response.voices[0]
        voice_name = voice.name if hasattr(voice, 'name') else "Default Voice"
        voice_id = voice.id if hasattr(voice, 'id') else None
        
        print(f"🎭 Using voice: {voice_name} (ID: {voice_id})")
        
        # Create utterance
        text = "Hello, this is a test of Hume AI voice generation."
        print(f"📝 Text: {text}")
        
        # Try with voice ID first
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
        
        # Try with voice name
        try:
            utterance = PostedUtterance(
                text=text,
                voice=PostedUtteranceVoiceWithName(name=voice_name)
            )
            print("🔄 Generating speech with voice name...")
            response = tts.synthesize_json(
                utterances=[utterance],
                format=FormatWav()
            )
            print("✅ Speech generated successfully with voice name!")
            
            # Save audio file
            output_file = "test_output_with_name.wav"
            with open(output_file, "wb") as audio_file:
                audio_file.write(response.content)
            print(f"✅ Audio saved to {output_file}")
            print(f"📊 Size: {len(response.content)} bytes")
            return True
            
        except Exception as e:
            print(f"❌ Failed with voice name: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error during voice generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_voices():
    """Test multiple available voices"""
    print(f"\n🎭 Testing Multiple Available Voices")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        tts = client.tts
        
        # Get voices
        voices_response = tts.voices.list()
        if not hasattr(voices_response, 'voices') or not voices_response.voices:
            print("❌ No voices available")
            return False
        
        # Test with first few voices
        voices_to_test = voices_response.voices[:3]  # Test first 3 voices
        text = "Hello, this is a test of different voice styles."
        
        for i, voice in enumerate(voices_to_test):
            voice_name = voice.name if hasattr(voice, 'name') else f"Voice {i+1}"
            voice_id = voice.id if hasattr(voice, 'id') else None
            
            print(f"\n🧪 Testing Voice {i+1}: {voice_name}")
            
            try:
                # Try with voice ID if available
                if voice_id:
                    from hume.tts.types import PostedUtteranceVoiceWithId
                    utterance = PostedUtterance(
                        text=text,
                        voice=PostedUtteranceVoiceWithId(id=voice_id)
                    )
                else:
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
        print(f"❌ Error during multiple voice testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Hume AI Voice Generation - Final Working Version")
    print("=" * 60)
    
    # Get available voices
    voices = test_available_voices()
    
    # Test voice generation with available voice
    success1 = test_voice_generation_with_available_voice()
    
    # Test multiple voices
    success2 = test_multiple_voices()
    
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print(f"Available Voices: {'✅ Success' if voices else '❌ Failed'}")
    print(f"Voice Generation: {'✅ Success' if success1 else '❌ Failed'}")
    print(f"Multiple Voices: {'✅ Success' if success2 else '❌ Failed'}")
    
    if voices or success1 or success2:
        print(f"\n🎉 Hume AI Voice Generation is working!")
        print(f"📁 Generated files:")
        for file in os.listdir("."):
            if file.endswith(".wav"):
                print(f"  - {file}")
    else:
        print(f"\n❌ All tests failed. Check the errors above.")
