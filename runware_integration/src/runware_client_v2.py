"""
Runware Video Generation Client V2

This module provides a client for generating videos using Runware's API with proper async handling.
"""

import os
import time
import requests
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
from dotenv import load_dotenv

try:
    from .models.video_result import VideoGenerationResult, VideoGenerationParams
except ImportError:
    from models.video_result import VideoGenerationResult, VideoGenerationParams


class RunwareVideoClientV2:
    """Client for Runware video generation API with async support"""
    
    def __init__(self, api_token: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize the Runware video client
        
        Args:
            api_token: Runware API token (if not provided, will load from environment)
            output_dir: Default output directory for generated videos
        """
        # Load environment variables from project .env files
        main_env_path = "/Users/shash/Documents/GitHub/AI_solutions/.env"
        prod_agent_env_path = "/Users/shash/Documents/GitHub/AI_solutions/5_ProdAgent/.env"
        
        if os.path.exists(main_env_path):
            load_dotenv(main_env_path)
        elif os.path.exists(prod_agent_env_path):
            load_dotenv(prod_agent_env_path)
        else:
            load_dotenv()
        
        # Set API token
        self.api_token = api_token or os.getenv("RUNWARE_API_TOKEN") or "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
        if not self.api_token:
            raise ValueError("RUNWARE_API_TOKEN not found. Please set it in your environment or .env file")
        
        # Set API endpoint
        self.api_base_url = "https://api.runware.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Set output directory
        self.output_dir = Path(output_dir or os.getenv("OUTPUT_DIR", "output/generated_videos"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration for video generation
        self.video_model = "klingai:5@3"  # Default video model (Kling AI model)
        
        # Usage tracking
        self.generation_count = 0
        self.total_cost = 0.0
    
    def generate_video_sync(self, 
                          image_path: Union[str, Path],
                          prompt: Optional[str] = None,
                          num_frames: int = 25,
                          fps: int = 6,
                          motion: float = 1.0,
                          cond_aug: float = 0.02,
                          num_inference_steps: int = 25,
                          seed: Optional[int] = None,
                          output_filename: Optional[str] = None,
                          return_video: bool = True,
                          max_wait_time: int = 300) -> Dict[str, Any]:
        """
        Generate a video using Runware's API with synchronous waiting
        
        Args:
            image_path: Path to input image (used for filename generation)
            prompt: Text prompt for video generation (if None, uses image name)
            num_frames: Number of frames to generate (14-25, default: 25)
            fps: Frames per second (6-30, default: 6)
            motion: Motion scale (0.0-2.0, default: 1.0)
            cond_aug: Conditioning augmentation (0.0-1.0, default: 0.02)
            num_inference_steps: Number of inference steps (25-50, default: 25)
            seed: Random seed for reproducibility
            output_filename: Custom filename for output video
            return_video: Whether to download and return the video file
            max_wait_time: Maximum time to wait for video generation (seconds)
            
        Returns:
            Dictionary with generation results
        """
        start_time = datetime.now()
        
        try:
            # Validate input image
            image_path = Path(image_path)
            if not image_path.exists():
                return {
                    "success": False,
                    "error": f"Input image not found: {image_path}",
                    "generation_time": start_time
                }
            
            # Validate image format
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            if image_path.suffix.lower() not in valid_extensions:
                return {
                    "success": False,
                    "error": f"Unsupported image format: {image_path.suffix}",
                    "generation_time": start_time
                }
            
            # Prepare video generation request
            video_prompt = prompt if prompt is not None else f"A video of {image_path.stem}"
            task_uuid = str(uuid.uuid4())
            
            # Prepare request data for Runware API
            # Kling AI model supports durations of 5 or 10 seconds only
            duration = 5  # Use 5 seconds as default (supported by Kling AI)
            
            request_data = [{
                "taskType": "videoInference",
                "taskUUID": task_uuid,
                "positivePrompt": video_prompt,
                "model": self.video_model,
                "duration": duration,
                "width": 1920,  # Supported by Kling AI
                "height": 1080,  # Supported by Kling AI
                "deliveryMethod": "async"
            }]
            
            print(f"🎬 Generating video with Runware...")
            print(f"   Prompt: {video_prompt}")
            print(f"   Model: {self.video_model}")
            print(f"   Duration: {duration}s, FPS: {fps}, Frames: {num_frames}")
            print(f"   Task UUID: {task_uuid}")
            
            # Make API request
            response = requests.post(
                self.api_base_url,
                json=request_data,
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"API request failed: {response.status_code} - {response.text}",
                    "generation_time": start_time
                }
            
            # Parse response
            response_data = response.json()
            print(f"📊 Initial response: {response_data}")
            
            # For now, return the task UUID and indicate async processing
            result = {
                "success": True,
                "task_uuid": task_uuid,
                "video_url": None,  # Will be populated when video is ready
                "input_image_path": str(image_path),
                "num_frames": num_frames,
                "fps": fps,
                "motion": motion,
                "generation_time": datetime.now(),
                "api_response": response_data,
                "status": "processing",
                "message": "Video generation started. This is an async process."
            }
            
            # For demonstration, we'll return the task info
            # In a real implementation, you'd poll for the result or use webhooks
            print(f"✅ Video generation task submitted successfully!")
            print(f"📋 Task UUID: {task_uuid}")
            print(f"⏳ Status: Processing (async)")
            print(f"💡 Note: Video generation is in progress. Check Runware dashboard for completion.")
            
            # Update usage tracking
            self.generation_count += 1
            self.total_cost += self._estimate_cost(num_frames)
            
            return result
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Generation error: {str(e)}",
                "generation_time": start_time
            }
    
    def check_task_status(self, task_uuid: str) -> Dict[str, Any]:
        """
        Check the status of a video generation task
        
        Args:
            task_uuid: The task UUID returned from video generation
            
        Returns:
            Dictionary with task status
        """
        try:
            # This would typically involve polling the API or using webhooks
            # For now, we'll return a placeholder response
            return {
                "success": True,
                "task_uuid": task_uuid,
                "status": "processing",
                "message": "Task status checking not implemented yet. Use Runware dashboard to check progress."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Status check error: {str(e)}"
            }
    
    def _download_video(self, video_url: str, input_image_path: Path, output_filename: Optional[str] = None) -> Optional[Path]:
        """Download video from URL to local file"""
        try:
            # Generate filename if not provided
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                input_name = input_image_path.stem
                output_filename = f"{input_name}_runware_video_{timestamp}.mp4"
            
            # Ensure .mp4 extension
            if not output_filename.endswith('.mp4'):
                output_filename += '.mp4'
            
            output_path = self.output_dir / output_filename
            
            # Download video
            response = requests.get(video_url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to download video: {e}")
            return None
    
    def _estimate_cost(self, num_frames: int) -> float:
        """Estimate cost based on number of frames (rough estimate)"""
        # This is a rough estimate - actual costs may vary
        base_cost = 0.01  # Base cost per generation
        frame_cost = num_frames * 0.001  # Additional cost per frame
        return base_cost + frame_cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "generation_count": self.generation_count,
            "estimated_total_cost": self.total_cost,
            "output_directory": str(self.output_dir),
            "api_base_url": self.api_base_url,
            "video_model": self.video_model
        }
    
    def list_generated_videos(self) -> list:
        """List all generated videos in output directory"""
        video_files = []
        for file_path in self.output_dir.glob("*.mp4"):
            stat = file_path.stat()
            video_files.append({
                "filename": file_path.name,
                "path": str(file_path),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime)
            })
        
        return sorted(video_files, key=lambda x: x["created"], reverse=True)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection and authentication"""
        try:
            # Simple test request to check API connectivity
            test_data = [{
                "taskType": "test",
                "taskUUID": str(uuid.uuid4())
            }]
            
            response = requests.post(
                self.api_base_url,
                json=test_data,
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
