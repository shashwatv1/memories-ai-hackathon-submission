# Runware Video Generation Project

## Overview
This project implements video generation using Runware's API. It provides a simple interface to generate videos from text prompts with various customization options.

## Key Features
- **Text-to-Video Generation**: Generate dynamic videos from text prompts
- **4 Simple Parameters**: Control prompt, negative prompt, duration, and resolution
- **Cost-Effective**: Leverage Runware's competitive pricing
- **Local File Management**: Download and organize generated videos
- **Error Handling**: Robust error handling and retry logic
- **Usage Tracking**: Monitor API usage and costs

## Runware Benefits
- ✅ **Cost-effective** video generation
- ✅ **High-quality output** with advanced AI models
- ✅ **API access** for integration
- ✅ **Fast generation** times
- ✅ **Multiple model options**

## Requirements
- Python 3.8+
- Runware API token
- Input images (JPG, PNG, etc.) for filename generation

## Installation
```bash
pip install requests python-dotenv pydantic pillow
```

## Setup
1. Get your API token from [Runware Account Settings](https://runware.ai)
2. Set environment variable: `export RUNWARE_API_TOKEN=your_token`
3. Or add to `.env` file: `RUNWARE_API_TOKEN=your_token`

## Basic Usage
```python
from src.runware_client import RunwareVideoClient

client = RunwareVideoClient()

# Generate video with 4 simple parameters
result = client.generate_video(
    image_path="input_image.jpg",
    prompt="A beautiful sunset over mountains",
    negative_prompt="blurry, dark, low quality",
    duration=5,
    width=1920,
    height=1080
)

if result["success"]:
    print(f"Video saved to: {result['video_path']}")
```

## Project Structure
```
20250118_002_runware_video_generation/
├── README.md
├── requirements.txt
├── env.example
├── src/
│   ├── __init__.py
│   ├── runware_client.py
│   └── models/
│       ├── __init__.py
│       └── video_result.py
├── test/
│   └── test_basic_generation.py
├── demo/
│   └── demo_basic.py
├── output/
│   └── generated_videos/
└── TASK_ACTIVITY_LOG.md
```

## 4 Simple Parameters
- **prompt**: Text description of the video (required)
- **negative_prompt**: What to avoid in the video (optional, default: "blurry, low quality, distorted")
- **duration**: Video length in seconds (5 or 10, default: 5)
- **resolution**: Video dimensions (width x height, default: 1920x1080)
  - Landscape: 1920x1080
  - Portrait: 1080x1920  
  - Square: 1080x1080

## Best Practices
1. **Prompt Quality**: Use detailed, descriptive prompts for better results
2. **Negative Prompts**: Specify what to avoid (blurry, dark, low quality)
3. **Duration**: Start with 5s for testing, use 10s for final videos
4. **Resolution**: Choose based on intended use (landscape for web, portrait for mobile)
5. **Testing**: Use 5s duration for testing to save credits

## Error Handling
- API rate limits and quota exceeded
- Invalid image formats
- Network connectivity issues
- Model processing errors

## Cost Management
- Monitor usage in Runware dashboard
- Use lower frame counts for testing
- Implement retry logic for failed requests
- Cache results to avoid regeneration

## Integration Notes
- Compatible with existing media search systems
- Can be integrated with Pexels search results
- Supports batch processing
- Ready for production deployment

## Future Enhancements
- Batch video generation
- Video style transfer
- Custom model fine-tuning
- Integration with other AI services
- Advanced video editing features

## Testing
Run the comprehensive 4-parameter test suite:
```bash
python test_4_parameters.py
```

Run the end-to-end test with polling:
```bash
python test_with_polling.py
```

Run the basic demo:
```bash
python demo/demo_basic.py
```

## API Key Configuration
The system will automatically use the provided API key: `NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK`

You can also set it via environment variable:
```bash
export RUNWARE_API_TOKEN=NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK
```
