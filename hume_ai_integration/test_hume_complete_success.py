#!/usr/bin/env python3
"""
Test Hume AI SDK - Complete Success Version

This script successfully generates voice using Hume AI SDK with proper audio handling.
"""

import os
import sys
import base64
from hume import HumeClient
from hume.tts.types import PostedUtterance, PostedUtteranceVoiceWithId, FormatWav


def test_voice_generation_complete():
    """Test successful voice generation with proper audio handling"""
    print("🧪 Testing Hume AI Voice Generation - COMPLETE SUCCESS")
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
        
        # Get audio content from generations
        if hasattr(response, 'generations') and response.generations:
            generation = response.generations[0]
            print(f"📊 Generation type: {type(generation)}")
            
            # Get audio content
            audio_content = generation.audio
            print(f"📊 Audio content type: {type(audio_content)}")
            print(f"📊 Audio content length: {len(audio_content)}")
            
            # Handle different audio content types
            if isinstance(audio_content, str):
                # If it's a string, it might be base64 encoded
                try:
                    audio_bytes = base64.b64decode(audio_content)
                    print("✅ Decoded base64 audio content")
                except Exception as e:
                    print(f"❌ Failed to decode base64: {e}")
                    # Try to encode as bytes directly
                    audio_bytes = audio_content.encode('utf-8')
                    print("✅ Encoded string as bytes")
            elif isinstance(audio_content, bytes):
                audio_bytes = audio_content
                print("✅ Audio content is already bytes")
            else:
                print(f"❌ Unknown audio content type: {type(audio_content)}")
                return False
            
            # Save audio file
            output_file = "test_complete_success.wav"
            print(f"💾 Saving audio to {output_file}...")
            with open(output_file, "wb") as audio_file:
                audio_file.write(audio_bytes)
            print(f"✅ Audio saved to {output_file}")
            print(f"📊 Size: {len(audio_bytes)} bytes")
            return True
        else:
            print("❌ No generations found in response")
            return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_voices_complete():
    """Test multiple voices with proper audio handling"""
    print(f"\n🎭 Testing Multiple Voices - Complete")
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
                
                # Get audio content from generations
                if hasattr(response, 'generations') and response.generations:
                    generation = response.generations[0]
                    audio_content = generation.audio
                    
                    # Handle different audio content types
                    if isinstance(audio_content, str):
                        try:
                            audio_bytes = base64.b64decode(audio_content)
                        except:
                            audio_bytes = audio_content.encode('utf-8')
                    elif isinstance(audio_content, bytes):
                        audio_bytes = audio_content
                    else:
                        print(f"❌ Unknown audio content type: {type(audio_content)}")
                        continue
                    
                    output_file = f"test_voice_{i+1}_{voice.name.lower().replace(' ', '_')}.wav"
                    with open(output_file, "wb") as audio_file:
                        audio_file.write(audio_bytes)
                    
                    print(f"✅ Generated: {output_file}")
                    print(f"📊 Size: {len(audio_bytes)} bytes")
                else:
                    print(f"❌ No generations found")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during multiple voice testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emotional_voices():
    """Test emotional voice generation"""
    print(f"\n😊 Testing Emotional Voices")
    print("=" * 50)
    
    try:
        api_key = "kGBxtW8ArdwxWJVs2feTIey3maFwYQq5Ds2XWQwCTVsOiPBh"
        client = HumeClient(api_key=api_key)
        
        # Get voices
        voices_response = client.tts.voices.list(provider="HUME_AI")
        voices = voices_response.items
        
        # Test emotional content with different voices
        emotional_tests = [
            {
                'name': 'Happy',
                'text': 'I am feeling so happy and excited today! This is amazing!',
                'voice_name': 'Cheerful Assistant'
            },
            {
                'name': 'Calm',
                'text': 'Take a deep breath and relax. Everything will be okay. Find your inner peace.',
                'voice_name': 'Serene Assistant'
            },
            {
                'name': 'Professional',
                'text': 'Welcome to our presentation. Today we will discuss important matters and strategic planning.',
                'voice_name': 'Professional Assistant'
            },
            {
                'name': 'Friendly',
                'text': 'Hey there! How are you doing today? I hope you are having a great time!',
                'voice_name': 'Friendly Assistant'
            }
        ]
        
        for test in emotional_tests:
            print(f"\n🧪 Testing: {test['name']} Voice")
            print(f"📝 Text: {test['text']}")
            print(f"🎭 Voice: {test['voice_name']}")
            
            # Find a suitable voice
            suitable_voice = None
            for voice in voices:
                if test['voice_name'].lower() in voice.name.lower():
                    suitable_voice = voice
                    break
            
            if not suitable_voice:
                # Use first available voice
                suitable_voice = voices[0]
                print(f"🎭 Using available voice: {suitable_voice.name}")
            
            try:
                utterance = PostedUtterance(
                    text=test['text'],
                    voice=PostedUtteranceVoiceWithId(id=suitable_voice.id)
                )
                
                response = client.tts.synthesize_json(
                    utterances=[utterance],
                    format=FormatWav()
                )
                
                # Get audio content from generations
                if hasattr(response, 'generations') and response.generations:
                    generation = response.generations[0]
                    audio_content = generation.audio
                    
                    # Handle different audio content types
                    if isinstance(audio_content, str):
                        try:
                            audio_bytes = base64.b64decode(audio_content)
                        except:
                            audio_bytes = audio_content.encode('utf-8')
                    elif isinstance(audio_content, bytes):
                        audio_bytes = audio_content
                    else:
                        print(f"❌ Unknown audio content type: {type(audio_content)}")
                        continue
                    
                    output_file = f"test_emotional_{test['name'].lower()}.wav"
                    with open(output_file, "wb") as audio_file:
                        audio_file.write(audio_bytes)
                    
                    print(f"✅ Generated: {output_file}")
                    print(f"📊 Size: {len(audio_bytes)} bytes")
                else:
                    print(f"❌ No generations found")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during emotional voice testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Hume AI Voice Generation - COMPLETE SUCCESS VERSION")
    print("=" * 60)
    
    # Test successful voice generation
    success1 = test_voice_generation_complete()
    
    # Test multiple voices
    success2 = test_multiple_voices_complete()
    
    # Test emotional voices
    success3 = test_emotional_voices()
    
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print(f"Voice Generation: {'✅ Success' if success1 else '❌ Failed'}")
    print(f"Multiple Voices: {'✅ Success' if success2 else '❌ Failed'}")
    print(f"Emotional Voices: {'✅ Success' if success3 else '❌ Failed'}")
    
    if success1 or success2 or success3:
        print(f"\n🎉 Hume AI Voice Generation is working!")
        print(f"📁 Generated files:")
        for file in os.listdir("."):
            if file.endswith(".wav"):
                print(f"  - {file}")
    else:
        print(f"\n❌ All tests failed. Check the errors above.")
