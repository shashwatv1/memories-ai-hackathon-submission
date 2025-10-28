# Runware Video Generation - Parameter Reference Guide

## Quick Reference

### Essential Parameters (Required)
```python
{
    "taskType": "videoInference",           # Always required
    "taskUUID": "unique-uuid-here",         # Always required  
    "positivePrompt": "Your video description", # Always required
    "model": "klingai:5@3",                # Always required
    "duration": 5,                         # 5 or 10 seconds
    "width": 1920,                         # 1920, 1080, or 1080
    "height": 1080,                        # 1080, 1920, or 1080
    "deliveryMethod": "async"              # "async" or "sync"
}
```

## Complete Parameter List

### 1. Core Parameters

| Parameter | Type | Required | Description | Valid Values |
|-----------|------|----------|-------------|--------------|
| `taskType` | string | ✅ | Task type identifier | `"videoInference"` |
| `taskUUID` | string | ✅ | Unique task identifier | UUID string |
| `positivePrompt` | string | ✅ | Video description | Any text |
| `model` | string | ✅ | AI model to use | `"klingai:5@3"`, `"klingai:5@1"` |

### 2. Video Configuration

| Parameter | Type | Required | Description | Valid Values | Default |
|-----------|------|----------|-------------|--------------|---------|
| `duration` | integer | ✅ | Video length in seconds | `5`, `10` | `5` |
| `width` | integer | ✅ | Video width in pixels | `1920`, `1080`, `1080` | `1920` |
| `height` | integer | ✅ | Video height in pixels | `1080`, `1920`, `1080` | `1080` |
| `deliveryMethod` | string | ✅ | Processing method | `"async"`, `"sync"` | `"async"` |

### 3. Content Parameters

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `negativePrompt` | string | ❌ | What to avoid in video | `"dark, scary, violent"` |
| `seed` | integer | ❌ | Random seed (model dependent) | `1234567890` |

### 4. Output Parameters

| Parameter | Type | Required | Description | Valid Values | Default |
|-----------|------|----------|-------------|--------------|---------|
| `outputFormat` | string | ❌ | Video format | `"mp4"` | `"mp4"` |
| `outputQuality` | integer | ❌ | Video quality (20-99) | `20-99` | `95` |
| `outputType` | string | ❌ | Response format | `"URL"`, `"dataURI"` | `"URL"` |

### 5. Advanced Parameters

| Parameter | Type | Required | Description | Valid Values |
|-----------|------|----------|-------------|--------------|
| `fps` | integer | ❌ | Frames per second | `6-30` |
| `CFGScale` | float | ❌ | Guidance scale | `1.0-20.0` |
| `ttl` | integer | ❌ | Task time-to-live (seconds) | `300-3600` |
| `webhookURL` | string | ❌ | Callback URL | Valid URL |
| `numberResults` | integer | ❌ | Number of videos | `1-4` |

### 6. Reference Parameters

| Parameter | Type | Required | Description | Format |
|-----------|------|----------|-------------|--------|
| `referenceImages` | array | ❌ | Style reference images | Base64 or URLs |
| `frameImages` | array | ❌ | Input frames | Base64 encoded |
| `providerSettings` | object | ❌ | Model-specific settings | JSON object |
| `advancedFeatures` | object | ❌ | Advanced AI features | JSON object |

## Model-Specific Parameters

### Kling AI Models (`klingai:5@3`, `klingai:5@1`)

#### Supported Parameters
```python
{
    "taskType": "videoInference",
    "taskUUID": "uuid",
    "positivePrompt": "text",
    "model": "klingai:5@3",
    "duration": 5,                    # Only 5 or 10
    "width": 1920,                    # Only 1920, 1080, 1080
    "height": 1080,                   # Only 1080, 1920, 1080
    "deliveryMethod": "async",
    "negativePrompt": "text",         # Optional
    "outputFormat": "mp4",           # Optional
    "outputQuality": 95,             # Optional
    "webhookURL": "url"              # Optional
}
```

#### Unsupported Parameters
- `seed` - Not supported by Kling AI
- `fps` - Model controlled
- `CFGScale` - Model controlled
- `referenceImages` - Not supported
- `frameImages` - Not supported

### Runware Native Models (`runware:*`)

#### Supported Parameters
```python
{
    "taskType": "videoInference",
    "taskUUID": "uuid",
    "positivePrompt": "text",
    "model": "runware:108@22",
    "duration": 5,                    # 5-30 seconds
    "width": 1920,                    # Flexible
    "height": 1080,                   # Flexible
    "deliveryMethod": "async",
    "seed": 1234567890,              # Supported
    "fps": 24,                       # Supported
    "CFGScale": 7.5,                 # Supported
    "negativePrompt": "text",
    "outputFormat": "mp4",
    "outputQuality": 95,
    "webhookURL": "url"
}
```

## Resolution Combinations

### Supported Resolutions

| Width | Height | Aspect Ratio | Use Case |
|-------|--------|--------------|----------|
| 1920 | 1080 | 16:9 | Landscape videos |
| 1080 | 1920 | 9:16 | Portrait videos |
| 1080 | 1080 | 1:1 | Square videos |

### Resolution Examples
```python
# Landscape (16:9)
{"width": 1920, "height": 1080}

# Portrait (9:16) 
{"width": 1080, "height": 1920}

# Square (1:1)
{"width": 1080, "height": 1080}
```

## Duration Options

### Supported Durations
- **5 seconds**: Standard duration, faster processing
- **10 seconds**: Extended duration, longer processing time

### Duration Examples
```python
# Short video (5 seconds)
{"duration": 5}

# Extended video (10 seconds)  
{"duration": 10}
```

## Quality Settings

### Output Quality Levels

| Quality | Value | Use Case |
|---------|-------|----------|
| Low | 20-40 | Quick previews |
| Medium | 50-70 | Standard quality |
| High | 80-95 | Production quality |
| Maximum | 95-99 | Best quality |

### Quality Examples
```python
# Low quality (faster, smaller file)
{"outputQuality": 30}

# High quality (slower, larger file)
{"outputQuality": 95}

# Maximum quality (slowest, largest file)
{"outputQuality": 99}
```

## Prompt Guidelines

### Positive Prompt Best Practices

#### Good Prompts
```python
# Detailed and specific
"A serene beach at sunset with gentle waves lapping at the shore, golden light reflecting on the water"

# Action-oriented
"A bird soaring gracefully through mountain peaks with clouds drifting below"

# Style-specific
"A cinematic shot of a thunderstorm with lightning flashing across dark clouds"
```

#### Bad Prompts
```python
# Too vague
"Nice video"

# Too complex
"A complex scene with many elements, characters, actions, and details that are hard to describe"

# Inappropriate content
"Violent or inappropriate content"
```

### Negative Prompt Examples
```python
# Common negative prompts
"dark, scary, violent, blood, gore, horror"

# Quality-related
"blurry, low quality, distorted, pixelated"

# Style-related
"cartoon, anime, drawing, painting"
```

## Error Prevention

### Common Parameter Errors

#### 1. Invalid Duration
```python
# ❌ Wrong
{"duration": 6}  # Not supported

# ✅ Correct  
{"duration": 5}  # Supported
{"duration": 10} # Supported
```

#### 2. Invalid Dimensions
```python
# ❌ Wrong
{"width": 1280, "height": 720}  # Not supported

# ✅ Correct
{"width": 1920, "height": 1080}  # Supported
{"width": 1080, "height": 1920}  # Supported
```

#### 3. Unsupported Parameters
```python
# ❌ Wrong (for Kling AI)
{"seed": 123456}  # Not supported by Kling AI

# ✅ Correct
# Remove seed parameter for Kling AI models
```

## Parameter Validation

### Pre-submission Checklist

```python
def validate_parameters(params):
    # Required parameters
    assert "taskType" in params
    assert "taskUUID" in params  
    assert "positivePrompt" in params
    assert "model" in params
    assert "duration" in params
    assert "width" in params
    assert "height" in params
    assert "deliveryMethod" in params
    
    # Validate duration
    assert params["duration"] in [5, 10]
    
    # Validate dimensions
    valid_dims = [(1920, 1080), (1080, 1920), (1080, 1080)]
    assert (params["width"], params["height"]) in valid_dims
    
    # Validate model
    assert params["model"] in ["klingai:5@3", "klingai:5@1"]
    
    return True
```

## Usage Examples

### Basic Video Generation
```python
basic_params = {
    "taskType": "videoInference",
    "taskUUID": str(uuid.uuid4()),
    "positivePrompt": "A beautiful sunset over mountains",
    "model": "klingai:5@3",
    "duration": 5,
    "width": 1920,
    "height": 1080,
    "deliveryMethod": "async"
}
```

### Advanced Video Generation
```python
advanced_params = {
    "taskType": "videoInference",
    "taskUUID": str(uuid.uuid4()),
    "positivePrompt": "A serene beach with gentle waves at sunset",
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
}
```

### High-Quality Video
```python
hq_params = {
    "taskType": "videoInference",
    "taskUUID": str(uuid.uuid4()),
    "positivePrompt": "Cinematic shot of a thunderstorm with lightning",
    "model": "klingai:5@3",
    "duration": 10,
    "width": 1920,
    "height": 1080,
    "deliveryMethod": "async",
    "outputQuality": 99
}
```

This parameter reference provides everything you need to effectively use the Runware video generation API with all available options and best practices.
