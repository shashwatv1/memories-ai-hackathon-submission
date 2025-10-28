"""
Video Generation Result Models

Pydantic models for Runware video generation results and parameters.
"""

from typing import Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, validator


class VideoGenerationParams(BaseModel):
    """Parameters for video generation"""
    
    image_path: Union[str, Path] = Field(..., description="Path to input image")
    prompt: Optional[str] = Field(None, description="Text prompt for video generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid unwanted elements")
    duration: int = Field(5, description="Video duration in seconds (5 or 10)")
    width: int = Field(1920, description="Video width in pixels")
    height: int = Field(1080, description="Video height in pixels")
    filename: Optional[str] = Field(None, description="Output filename")
    
    @validator('image_path')
    def validate_image_path(cls, v):
        """Validate image path exists and has valid extension"""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Image path does not exist: {path}")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        return path
    
    @validator('prompt')
    def validate_prompt(cls, v, values):
        """Generate prompt from image name if not provided"""
        if v is None and 'image_path' in values:
            image_path = Path(values['image_path'])
            return f"A video of {image_path.stem}"
        return v
    
    @validator('duration')
    def validate_duration(cls, v):
        """Validate duration is 5 or 10 seconds"""
        if v not in [5, 10]:
            raise ValueError(f"Duration must be 5 or 10 seconds, got: {v}")
        return v
    
    @validator('width', 'height')
    def validate_resolution(cls, v):
        """Validate resolution parameters"""
        if v not in [1920, 1080]:
            raise ValueError(f"Resolution must be 1920 or 1080 pixels, got: {v}")
        return v
    
    @validator('height')
    def validate_resolution_combination(cls, v, values):
        """Validate resolution combination is supported"""
        if 'width' in values:
            width = values['width']
            valid_combinations = [(1920, 1080), (1080, 1920), (1080, 1080)]
            if (width, v) not in valid_combinations:
                raise ValueError(f"Unsupported resolution combination: {width}x{v}. Supported: 1920x1080, 1080x1920, 1080x1080")
        return v


class VideoGenerationResult(BaseModel):
    """Result of video generation"""
    
    success: bool = Field(..., description="Whether generation was successful")
    video_url: Optional[str] = Field(None, description="URL of generated video")
    video_path: Optional[str] = Field(None, description="Local path to downloaded video")
    input_image_path: str = Field(..., description="Path to input image")
    prompt: str = Field(..., description="Text prompt used for generation")
    negative_prompt: str = Field(..., description="Negative prompt used for generation")
    duration: int = Field(..., description="Duration of video in seconds")
    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    generation_time: datetime = Field(..., description="Time when generation completed")
    file_size: Optional[int] = Field(None, description="Size of generated video file in bytes")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    api_response: Optional[Dict[str, Any]] = Field(None, description="Raw API response")
    download_error: Optional[str] = Field(None, description="Error during video download")
    
    @validator('generation_time', pre=True)
    def parse_generation_time(cls, v):
        """Parse generation time if it's a string"""
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }
