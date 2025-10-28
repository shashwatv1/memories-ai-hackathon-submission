"""
Hume AI Voice Generation Client

This module provides a client for generating emotional voice using Hume AI's EVI (Empathic Voice Interface).
"""

import os
import requests
import json
import time
from typing import Dict, Optional, Any


class HumeVoiceClient:
    """
    Hume AI voice generation client for emotional voice synthesis.
    """
    
    def __init__(self, api_key: Optional[str] = None, secret: Optional[str] = None):
        """
        Initialize the Hume voice client
        
        Args:
            api_key: Hume AI API key
            secret: Hume AI secret key
        """
        self.api_key = api_key or os.getenv("HUME_API_KEY")
        self.secret = secret or os.getenv("HUME_SECRET")
        
        if not self.api_key:
            raise ValueError("HUME_API_KEY not found. Please set it in your environment or .env file")
        if not self.secret:
            raise ValueError("HUME_SECRET not found. Please set it in your environment or .env file")
        
        # Hume AI API endpoints (based on typical API structure)
        self.api_base_url = "https://api.hume.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Hume-Secret": self.secret
        }
    
    def generate_voice(self, 
                      text: str, 
                      emotion: str = "neutral",
                      intensity: float = 0.5,
                      voice_style: str = "narrator",
                      output_format: str = "mp3") -> Dict[str, Any]:
        """
        Generate emotional voice from text using Hume AI EVI.
        
        Args:
            text: Text to convert to speech
            emotion: Emotional tone (neutral, happy, sad, angry, excited, calm, etc.)
            intensity: Emotional intensity (0.0-1.0)
            voice_style: Voice style (narrator, conversational, professional, etc.)
            output_format: Audio format (mp3, wav, etc.)
            
        Returns:
            Dict with success status, audio_url, and metadata
        """
        try:
            print(f"🎙️ Generating voice with Hume AI...")
            print(f"   Text: {text[:50]}...")
            print(f"   Emotion: {emotion}")
            print(f"   Intensity: {intensity}")
            print(f"   Style: {voice_style}")
            
            # Prepare request payload
            payload = {
                "text": text,
                "emotion": emotion,
                "intensity": intensity,
                "voice_style": voice_style,
                "output_format": output_format,
                "model": "evi-2"  # Empathic Voice Interface v2
            }
            
            # Make API request
            response = requests.post(
                f"{self.api_base_url}/v1/voice/generate",
                json=payload,
                headers=self.headers,
                timeout=60
            )
            
            print(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Voice generation successful!")
                print(f"📊 Response: {json.dumps(result, indent=2)}")
                return {
                    "success": True,
                    "audio_url": result.get("audio_url"),
                    "duration": result.get("duration"),
                    "emotion": emotion,
                    "intensity": intensity,
                    "voice_style": voice_style,
                    "metadata": result
                }
            else:
                error_msg = f"API request failed: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            error_msg = f"Voice generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection and authentication
        
        Returns:
            Dict with connection test result
        """
        try:
            print(f"🔍 Testing Hume AI API connection...")
            
            # Test with a simple voice generation request
            test_payload = {
                "text": "Hello, this is a test.",
                "emotion": "neutral",
                "intensity": 0.5,
                "voice_style": "narrator",
                "output_format": "mp3"
            }
            
            response = requests.post(
                f"{self.api_base_url}/v1/voice/generate",
                json=test_payload,
                headers=self.headers,
                timeout=30
            )
            
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response": response.text[:200] if response.text else "No response body"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_available_emotions(self) -> list:
        """
        Get list of available emotions for voice generation
        
        Returns:
            List of supported emotions
        """
        return [
            "neutral", "happy", "sad", "angry", "excited", "calm", 
            "melancholic", "energetic", "serious", "playful", "dramatic",
            "romantic", "mysterious", "confident", "gentle", "intense"
        ]
    
    def get_available_voice_styles(self) -> list:
        """
        Get list of available voice styles
        
        Returns:
            List of supported voice styles
        """
        return [
            "narrator", "conversational", "professional", "casual",
            "formal", "friendly", "authoritative", "warm", "cool"
        ]


def test_hume_voice_generation():
    """Test Hume AI voice generation with different parameters"""
    print("🧪 Testing Hume AI Voice Generation")
    print("=" * 60)
    
    try:
        # Initialize client
        client = HumeVoiceClient()
        print("✅ HumeVoiceClient initialized successfully")
        
        # Test connection
        print("\n🔍 Testing API connection...")
        connection_result = client.test_connection()
        if connection_result.get('success'):
            print("✅ API connection successful")
        else:
            print(f"❌ API connection failed: {connection_result}")
            return False
        
        # Test cases
        test_cases = [
            {
                'name': 'Test 1: Neutral Narration',
                'params': {
                    'text': 'Welcome to our video presentation. Today we will explore the beautiful sunset over the mountains.',
                    'emotion': 'neutral',
                    'intensity': 0.5,
                    'voice_style': 'narrator'
                }
            },
            {
                'name': 'Test 2: Happy and Energetic',
                'params': {
                    'text': 'What an amazing day! The sun is shining and everything looks wonderful!',
                    'emotion': 'happy',
                    'intensity': 0.8,
                    'voice_style': 'conversational'
                }
            },
            {
                'name': 'Test 3: Calm and Gentle',
                'params': {
                    'text': 'Take a deep breath and relax. The gentle waves are soothing your mind.',
                    'emotion': 'calm',
                    'intensity': 0.3,
                    'voice_style': 'gentle'
                }
            },
            {
                'name': 'Test 4: Dramatic and Intense',
                'params': {
                    'text': 'The storm approaches with thunderous force, shaking the very foundations of the earth.',
                    'emotion': 'dramatic',
                    'intensity': 0.9,
                    'voice_style': 'authoritative'
                }
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n🎬 {test_case['name']}")
            print(f"📝 Text: {test_case['params']['text']}")
            print(f"😊 Emotion: {test_case['params']['emotion']}")
            print(f"⚡ Intensity: {test_case['params']['intensity']}")
            print(f"🎭 Style: {test_case['params']['voice_style']}")
            print("-" * 50)
            
            try:
                result = client.generate_voice(**test_case['params'])
                
                if result.get('success'):
                    print(f"✅ Voice generated successfully!")
                    print(f"🔗 Audio URL: {result.get('audio_url', 'N/A')}")
                    print(f"⏱️  Duration: {result.get('duration', 'N/A')} seconds")
                    results.append({
                        'name': test_case['name'],
                        'success': True,
                        'result': result
                    })
                else:
                    print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                    results.append({
                        'name': test_case['name'],
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                results.append({
                    'name': test_case['name'],
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        print(f"\n📊 TEST SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for r in results if r.get('success', False))
        total = len(results)
        
        print(f"Total Tests: {total}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {total - successful}")
        print(f"Success Rate: {(successful/total)*100:.1f}%")
        
        if successful > 0:
            print(f"\n🎙️ Generated Voices:")
            for result in results:
                if result.get('success', False):
                    print(f"  - {result['name']}: {result['result'].get('audio_url', 'N/A')}")
        
        if total - successful > 0:
            print(f"\n❌ Failed Tests:")
            for result in results:
                if not result.get('success', False):
                    print(f"  - {result['name']}: {result.get('error', 'Unknown error')}")
        
        return successful == total
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False


if __name__ == "__main__":
    success = test_hume_voice_generation()
    if success:
        print(f"\n🎉 All tests passed! Hume AI voice generation is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Please check the issues above.")
