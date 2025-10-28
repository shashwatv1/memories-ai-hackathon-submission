"""
Hume AI Voice Generation Client - Working Version

This module provides a working client for generating emotional voice using Hume AI's TTS API.
"""

import os
import base64
from typing import Dict, List, Optional, Any
from hume import HumeClient
from hume.tts.types import PostedUtterance, PostedUtteranceVoiceWithId, FormatWav


class HumeVoiceClient:
    """
    Working Hume AI voice generation client for emotional voice synthesis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Hume voice client
        
        Args:
            api_key: Hume AI API key
        """
        self.api_key = api_key or os.getenv("HUME_API_KEY")
        
        if not self.api_key:
            raise ValueError("HUME_API_KEY not found. Please set it in your environment or .env file")
        
        # Initialize Hume client
        self.client = HumeClient(api_key=self.api_key)
        self.tts = self.client.tts
        
        # Cache available voices
        self._voices = None
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices
        
        Returns:
            List of voice dictionaries with id, name, and tags
        """
        if self._voices is None:
            try:
                voices_response = self.tts.voices.list(provider="HUME_AI")
                self._voices = []
                for voice in voices_response.items:
                    self._voices.append({
                        'id': voice.id,
                        'name': voice.name,
                        'provider': voice.provider,
                        'tags': voice.tags,
                        'compatible_models': voice.compatible_octave_models
                    })
            except Exception as e:
                print(f"❌ Error getting voices: {e}")
                self._voices = []
        
        return self._voices
    
    def generate_voice(self, 
                      text: str, 
                      voice_id: Optional[str] = None,
                      voice_name: Optional[str] = None,
                      output_format: str = "wav") -> Dict[str, Any]:
        """
        Generate emotional voice from text using Hume AI TTS.
        
        Args:
            text: Text to convert to speech
            voice_id: Specific voice ID to use
            voice_name: Voice name to search for
            output_format: Audio format (wav, mp3, etc.)
            
        Returns:
            Dict with success status, audio data, and metadata
        """
        try:
            print(f"🎙️ Generating voice with Hume AI...")
            print(f"   Text: {text[:50]}...")
            
            # Get voice ID
            if not voice_id:
                if voice_name:
                    # Find voice by name
                    voices = self.get_available_voices()
                    for voice in voices:
                        if voice_name.lower() in voice['name'].lower():
                            voice_id = voice['id']
                            print(f"   Voice: {voice['name']} (ID: {voice_id})")
                            break
                
                if not voice_id:
                    # Use first available voice
                    voices = self.get_available_voices()
                    if voices:
                        voice_id = voices[0]['id']
                        print(f"   Voice: {voices[0]['name']} (ID: {voice_id})")
                    else:
                        raise ValueError("No voices available")
            
            # Create utterance
            utterance = PostedUtterance(
                text=text,
                voice=PostedUtteranceVoiceWithId(id=voice_id)
            )
            
            # Generate speech
            response = self.tts.synthesize_json(
                utterances=[utterance],
                format=FormatWav()
            )
            
            # Extract audio content
            if hasattr(response, 'generations') and response.generations:
                generation = response.generations[0]
                audio_content = generation.audio
                
                # Handle base64 encoded audio
                if isinstance(audio_content, str):
                    try:
                        audio_bytes = base64.b64decode(audio_content)
                    except:
                        audio_bytes = audio_content.encode('utf-8')
                elif isinstance(audio_content, bytes):
                    audio_bytes = audio_content
                else:
                    raise ValueError(f"Unknown audio content type: {type(audio_content)}")
                
                print(f"✅ Voice generated successfully!")
                print(f"📊 Audio size: {len(audio_bytes)} bytes")
                
                return {
                    "success": True,
                    "audio_data": audio_bytes,
                    "audio_size": len(audio_bytes),
                    "voice_id": voice_id,
                    "text": text,
                    "metadata": {
                        "generation_id": generation.generation_id,
                        "duration": getattr(generation, 'duration', None),
                        "encoding": getattr(generation, 'encoding', None),
                        "file_size": getattr(generation, 'file_size', None)
                    }
                }
            else:
                raise ValueError("No generations found in response")
                
        except Exception as e:
            error_msg = f"Voice generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def generate_voice_to_file(self, 
                              text: str, 
                              output_file: str,
                              voice_id: Optional[str] = None,
                              voice_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate voice and save to file
        
        Args:
            text: Text to convert to speech
            output_file: Path to save audio file
            voice_id: Specific voice ID to use
            voice_name: Voice name to search for
            
        Returns:
            Dict with success status and file path
        """
        result = self.generate_voice(text, voice_id, voice_name)
        
        if result.get('success'):
            try:
                with open(output_file, "wb") as audio_file:
                    audio_file.write(result['audio_data'])
                
                print(f"💾 Audio saved to {output_file}")
                return {
                    "success": True,
                    "file_path": output_file,
                    "file_size": result['audio_size'],
                    "voice_id": result['voice_id'],
                    "text": result['text']
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to save file: {str(e)}"
                }
        else:
            return result
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection and authentication
        
        Returns:
            Dict with connection test result
        """
        try:
            print(f"🔍 Testing Hume AI API connection...")
            
            # Test by getting voices
            voices = self.get_available_voices()
            
            return {
                "success": len(voices) > 0,
                "voices_count": len(voices),
                "voices": [v['name'] for v in voices[:5]]  # First 5 voice names
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def test_hume_voice_client():
    """Test the working Hume voice client"""
    print("🧪 Testing Hume Voice Client")
    print("=" * 50)
    
    try:
        # Initialize client
        client = HumeVoiceClient()
        print("✅ HumeVoiceClient initialized successfully")
        
        # Test connection
        print("\n🔍 Testing API connection...")
        connection_result = client.test_connection()
        if connection_result.get('success'):
            print("✅ API connection successful")
            print(f"📊 Available voices: {connection_result.get('voices_count', 0)}")
            print(f"📊 Sample voices: {connection_result.get('voices', [])}")
        else:
            print(f"❌ API connection failed: {connection_result}")
            return False
        
        # Test voice generation
        print("\n🎙️ Testing voice generation...")
        text = "Hello, this is a test of the Hume AI voice generation system."
        result = client.generate_voice(text)
        
        if result.get('success'):
            print("✅ Voice generation successful!")
            print(f"📊 Audio size: {result.get('audio_size', 0)} bytes")
            
            # Test saving to file
            print("\n💾 Testing file save...")
            file_result = client.generate_voice_to_file(text, "test_client_output.wav")
            if file_result.get('success'):
                print(f"✅ File saved: {file_result.get('file_path')}")
                return True
            else:
                print(f"❌ File save failed: {file_result.get('error')}")
                return False
        else:
            print(f"❌ Voice generation failed: {result.get('error')}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_hume_voice_client()
    if success:
        print(f"\n🎉 Hume AI Voice Client is working correctly!")
    else:
        print(f"\n⚠️  Some tests failed. Please check the issues above.")
