# Runware Video Generation API - Complete Documentation

## Overview
This documentation covers the complete Runware video generation API integration, including all available parameters, usage examples, and implementation details.

## Table of Contents
1. [API Authentication](#api-authentication)
2. [Video Generation Parameters](#video-generation-parameters)
3. [Task Management](#task-management)
4. [Response Handling](#response-handling)
5. [Error Handling](#error-handling)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)

## API Authentication

### API Key
```python
api_token = "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
```

### Headers
```python
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}
```

## Video Generation Parameters

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `taskType` | string | Task type for video generation | `"videoInference"` |
| `taskUUID` | string | Unique identifier for the task | `"uuid4-generated-id"` |
| `positivePrompt` | string | Text description of desired video | `"A beautiful sunset over mountains"` |
| `model` | string | AI model identifier | `"klingai:5@3"` |

### Video Configuration Parameters

| Parameter | Type | Description | Valid Values | Default |
|-----------|------|-------------|--------------|---------|
| `duration` | integer | Video duration in seconds | `5`, `10` | `5` |
| `width` | integer | Video width in pixels | `1920`, `1080`, `1080` | `1920` |
| `height` | integer | Video height in pixels | `1080`, `1920`, `1080` | `1080` |
| `deliveryMethod` | string | Processing method | `"async"`, `"sync"` | `"async"` |

### Optional Parameters

| Parameter | Type | Description | Valid Values | Notes |
|-----------|------|-------------|--------------|-------|
| `negativePrompt` | string | What to avoid in the video | Any text | Optional |
| `seed` | integer | Random seed for reproducibility | Any integer | Not supported by all models |
| `fps` | integer | Frames per second | `6-30` | Model dependent |
| `outputFormat` | string | Video format | `"mp4"` | Default format |
| `outputQuality` | integer | Video quality | `20-99` | Higher = better quality |
| `webhookURL` | string | Callback URL for completion | Valid URL | For async processing |
| `ttl` | integer | Time to live in seconds | `300-3600` | Task expiration |
| `CFGScale` | float | Guidance scale | `1.0-20.0` | Model dependent |
| `referenceImages` | array | Reference images | Base64 or URLs | For style transfer |

### Advanced Parameters

| Parameter | Type | Description | Valid Values |
|-----------|------|-------------|--------------|
| `frameImages` | array | Input frames for video | Base64 encoded images |
| `providerSettings` | object | Model-specific settings | Varies by model |
| `advancedFeatures` | object | Advanced AI features | Model dependent |
| `uploadEndpoint` | string | Custom upload endpoint | Valid URL |
| `numberResults` | integer | Number of videos to generate | `1-4` |

## Supported Models

### Video Generation Models

| Model ID | Name | Duration | Resolution | Notes |
|----------|------|----------|------------|-------|
| `klingai:5@3` | Kling AI 2.1 Master | 5s, 10s | 1920x1080, 1080x1920, 1080x1080 | Primary model |
| `klingai:5@1` | Kling AI Alternative | 5s, 10s | 1920x1080, 1080x1920, 1080x1080 | Alternative model |
| `runware:108@22` | Runware Video Model | 5s, 10s | 1920x1080, 1080x1920, 1080x1080 | Runware native |

### Model-Specific Constraints

#### Kling AI Models (`klingai:5@3`, `klingai:5@1`)
- **Duration**: Only 5 or 10 seconds
- **Resolution**: 1920x1080 (16:9), 1080x1920 (9:16), 1080x1080 (1:1)
- **Seed**: Not supported
- **FPS**: Model controlled

#### Runware Native Models (`runware:*`)
- **Duration**: 5-30 seconds
- **Resolution**: Flexible
- **Seed**: Supported
- **FPS**: Configurable

## Task Management

### Task Lifecycle

1. **Submit Task**: Send video generation request
2. **Get Task UUID**: Receive unique identifier
3. **Poll Status**: Check task completion
4. **Download Result**: Retrieve generated video

### Status Polling

```python
# Check task status
request_data = [{
    "taskType": "getResponse",
    "taskUUID": "your-task-uuid"
}]
```

### Task Status Values

| Status | Description | Action |
|--------|-------------|--------|
| `processing` | Task in progress | Continue polling |
| `success` | Task completed | Download video |
| `failed` | Task failed | Check error message |
| `error` | Task error | Retry or contact support |

## Response Handling

### Successful Response Structure

```json
{
  "data": [
    {
      "taskUUID": "uuid-here",
      "taskType": "videoInference",
      "status": "success",
      "videoUUID": "video-uuid-here",
      "seed": 9121768170631298021,
      "videoURL": "https://vm.runware.ai/video/ws/5/vi/video-uuid.mp4"
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `taskUUID` | string | Original task identifier |
| `taskType` | string | Task type |
| `status` | string | Task status |
| `videoUUID` | string | Generated video identifier |
| `seed` | integer | Random seed used |
| `videoURL` | string | Download URL for video |

## Error Handling

### Common Error Codes

| Error Code | Description | Solution |
|------------|-------------|----------|
| `invalidModel` | Model not supported | Use supported model ID |
| `invalidDuration` | Duration not supported | Use 5 or 10 seconds |
| `unsupportedDimensions` | Resolution not supported | Use supported dimensions |
| `unsupportedParameter` | Parameter not supported | Remove unsupported parameter |
| `unknownErrorWhileReadingResults` | Server error | Retry request |

### Error Response Structure

```json
{
  "data": [],
  "errors": [
    {
      "code": "error-code",
      "message": "Error description",
      "parameter": "parameter-name",
      "type": "parameter-type",
      "allowedValues": ["value1", "value2"]
    }
  ]
}
```

## Usage Examples

### Basic Video Generation

```python
import requests
import uuid

# API configuration
api_token = "NaSUWKyScWvUcc8QB1AwaR33Do8QNtcK"
api_base_url = "https://api.runware.ai/v1"
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

# Generate video
task_uuid = str(uuid.uuid4())
request_data = [{
    "taskType": "videoInference",
    "taskUUID": task_uuid,
    "positivePrompt": "A beautiful sunset over mountains",
    "model": "klingai:5@3",
    "duration": 5,
    "width": 1920,
    "height": 1080,
    "deliveryMethod": "async"
}]

response = requests.post(api_base_url, json=request_data, headers=headers)
```

### Advanced Video Generation

```python
# Advanced parameters
request_data = [{
    "taskType": "videoInference",
    "taskUUID": str(uuid.uuid4()),
    "positivePrompt": "A serene beach with gentle waves",
    "negativePrompt": "dark, scary, violent",
    "model": "klingai:5@3",
    "duration": 10,
    "width": 1920,
    "height": 1080,
    "deliveryMethod": "async",
    "outputFormat": "mp4",
    "outputQuality": 95,
    "webhookURL": "https://your-webhook-url.com/callback",
    "ttl": 1800
}]
```

### Status Polling

```python
# Poll for completion
def poll_task_status(task_uuid, max_attempts=30, delay=10):
    for attempt in range(max_attempts):
        request_data = [{
            "taskType": "getResponse",
            "taskUUID": task_uuid
        }]
        
        response = requests.post(api_base_url, json=request_data, headers=headers)
        result = response.json()
        
        if result['data'][0]['status'] == 'success':
            return result['data'][0]['videoURL']
        elif result['data'][0]['status'] in ['failed', 'error']:
            raise Exception(f"Task failed: {result['data'][0]}")
        
        time.sleep(delay)
    
    raise Exception("Task timeout")
```

### Video Download

```python
import requests
from pathlib import Path

def download_video(video_url, output_path):
    response = requests.get(video_url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return output_path
```

## Best Practices

### Performance Optimization

1. **Use Async Processing**: Set `deliveryMethod: "async"` for better performance
2. **Polling Intervals**: Use 10-15 second delays between status checks
3. **Timeout Handling**: Set reasonable timeouts (5-10 minutes)
4. **Batch Processing**: Submit multiple tasks simultaneously

### Error Handling

1. **Retry Logic**: Implement exponential backoff for failed requests
2. **Parameter Validation**: Validate all parameters before submission
3. **Status Monitoring**: Always check task status before downloading
4. **Logging**: Log all API interactions for debugging

### Cost Optimization

1. **Model Selection**: Choose appropriate models for your use case
2. **Duration**: Use shorter durations when possible (5s vs 10s)
3. **Quality Settings**: Balance quality vs cost
4. **Caching**: Cache results to avoid regeneration

### Security

1. **API Key Protection**: Store API keys securely
2. **HTTPS Only**: Always use HTTPS for API calls
3. **Input Validation**: Validate all user inputs
4. **Rate Limiting**: Implement rate limiting to avoid quota issues

## Rate Limits and Quotas

### API Limits
- **Requests per minute**: 60
- **Concurrent tasks**: 10
- **Daily quota**: Varies by plan
- **Video duration**: 5-10 seconds per video

### Monitoring Usage
```python
# Check usage statistics
def get_usage_stats():
    # Implementation depends on your tracking system
    return {
        "requests_today": 0,
        "videos_generated": 0,
        "total_cost": 0.0
    }
```

## Troubleshooting

### Common Issues

1. **"Invalid Model" Error**
   - Solution: Use supported model IDs
   - Check: `klingai:5@3`, `klingai:5@1`

2. **"Invalid Duration" Error**
   - Solution: Use only 5 or 10 seconds
   - Check: `duration: 5` or `duration: 10`

3. **"Unsupported Dimensions" Error**
   - Solution: Use supported resolutions
   - Check: 1920x1080, 1080x1920, 1080x1080

4. **Task Timeout**
   - Solution: Increase polling timeout
   - Check: Network connectivity

### Debug Information

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Log API requests
def log_request(request_data, response):
    print(f"Request: {request_data}")
    print(f"Response: {response.json()}")
```

## Support and Resources

- **Documentation**: https://runware.ai/docs
- **API Reference**: https://runware.ai/docs/en/video-inference/api-reference
- **Support**: support@runware.ai
- **GitHub**: https://github.com/Runware
- **Community**: Runware Discord/Forum

## Changelog

### Version 1.0.0
- Initial implementation
- Basic video generation
- Status polling
- Video download
- Error handling

### Future Enhancements
- Webhook support
- Batch processing
- Advanced model parameters
- Real-time streaming
