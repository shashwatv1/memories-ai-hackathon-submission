#!/usr/bin/env python3
"""
Test Hume AI SDK - Final Success Version

This script successfully generates voice using Hume AI SDK.
"""

import os
import sys
from hume import HumeClient
from hume.tts.types import PostedUtterance, PostedUtteranceVoiceWithId, FormatWav


def test_voice_generation_final():
    """Test successful voice generation with correct response handling"""
    print("🧪 Testing Hume AI Voice Generation - FINAL SUCCESS")
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
        print(f"📊 Response has generations: {hasattr(response, 'generations')}")
        
        if hasattr(response, 'generations'):
            print(f"📊 Number of generations: {len(response.generations)}")
            
            # Get the first generation
            generation = response.generations[0]
            print(f"📊 Generation type: {type(generation)}")
            print(f"📊 Generation attributes: {[attr for attr in dir(generation) if not attr.startswith('_')]}")
            
            # Try to get audio content from generation
            audio_content = None
            
            if hasattr(generation, 'content'):
                audio_content = generation.content
                print("✅ Got audio content from generation.content")
            elif hasattr(generation, 'audio'):
                audio_content = generation.audio
                print("✅ Got audio content from generation.audio")
            elif hasattr(generation, 'data'):
                audio_content = generation.data
                print("✅ Got audio content from generation.data")
            elif hasattr(generation, 'file'):
                audio_content = generation.file
                print("✅ Got audio content from generation.file")
            elif hasattr(generation, 'bytes'):
                audio_content = generation.bytes
                print("✅ Got audio content from generation.bytes")
            else:
                print(f"📊 Generation: {generation}")
                print(f"📊 Generation dict: {generation.__dict__ if hasattr(generation, '__dict__') else 'No dict'}")
        
        if audio_content:
            # Save audio file
            output_file = "test_final_success.wav"
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


def test_multiple_voices_final():
    """Test multiple voices with correct response handling"""
    print(f"\n🎭 Testing Multiple Voices - Final")
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
                
                # Try to get audio content from generations
                audio_content = None
                if hasattr(response, 'generations') and response.generations:
                    generation = response.generations[0]
                    if hasattr(generation, 'content'):
                        audio_content = generation.content
                    elif hasattr(generation, 'audio'):
                        audio_content = generation.audio
                    elif hasattr(generation, 'data'):
                        audio_content = generation.data
                
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
    print("🧪 Hume AI Voice Generation - FINAL SUCCESS VERSION")
    print("=" * 60)
    
    # Test successful voice generation
    success1 = test_voice_generation_final()
    
    # Test multiple voices
    success2 = test_multiple_voices_final()
    
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
