# 🔗 Integration Guide

## Overview

This guide explains how the different AI services and components work together in our content creation platform. Each integration is designed to be modular and reusable across different use cases.

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER                        │
│  Unified API | Service Registry | Error Handling           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   SERVICE ADAPTERS                         │
│  Hume Adapter | Runware Adapter | Content Adapter          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    AI SERVICES                              │
│  Hume.ai | Runware.ai | OpenAI | Spotify | Custom ML      │
└─────────────────────────────────────────────────────────────┘
```

## 1. Hume.ai Voice Generation Integration

### **Integration Pattern**
```python
# Service Adapter Pattern
class HumeVoiceAdapter:
    def __init__(self, api_key):
        self.client = HumeClient(api_key=api_key)
        self.tts = self.client.tts
    
    def generate_voice(self, text, voice_id, emotion=None):
        # 1. Validate input parameters
        # 2. Create utterance with voice configuration
        # 3. Call Hume.ai API
        # 4. Process base64 audio response
        # 5. Return standardized audio data
```

### **Key Integration Points**
- **Voice Selection**: Dynamic voice selection based on content type
- **Emotional Control**: Intensity-based emotional voice generation
- **Audio Processing**: Base64 decoding and WAV file generation
- **Error Handling**: Comprehensive error handling and retry logic

### **Usage Example**
```python
# Initialize adapter
hume_adapter = HumeVoiceAdapter(api_key="your_api_key")

# Generate emotional voice
audio_data = hume_adapter.generate_voice(
    text="Welcome to our amazing content!",
    voice_id="colton_rivers",
    emotion="friendly"
)

# Save audio file
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### **Integration Benefits**
- **High-Quality Audio**: Professional-grade voice synthesis
- **Emotional Range**: Multiple emotional states and intensities
- **Multiple Voices**: 10+ different voice options
- **Fast Processing**: 2-3 second generation times

## 2. Runware.ai Video Generation Integration

### **Integration Pattern**
```python
# Async Service Pattern
class RunwareVideoAdapter:
    def __init__(self, api_token):
        self.api_token = api_token
        self.base_url = "https://api.runware.ai/v1"
    
    async def generate_video(self, prompt, duration=5, resolution="1920x1080"):
        # 1. Validate parameters
        # 2. Submit async task
        # 3. Poll for completion
        # 4. Download video when ready
        # 5. Return video data
```

### **Key Integration Points**
- **Async Processing**: Handle long-running video generation tasks
- **Parameter Validation**: Ensure valid input parameters
- **Status Polling**: Monitor task progress and completion
- **File Management**: Download and organize generated videos

### **Usage Example**
```python
# Initialize adapter
runware_adapter = RunwareVideoAdapter(api_token="your_token")

# Generate video
video_data = await runware_adapter.generate_video(
    prompt="A beautiful sunset over mountains",
    duration=10,
    resolution="1920x1080"
)

# Save video file
with open("output.mp4", "wb") as f:
    f.write(video_data)
```

### **Integration Benefits**
- **High-Quality Videos**: Professional video generation
- **Multiple Formats**: Various resolutions and durations
- **Cost-Effective**: Competitive pricing for AI video generation
- **Reliable Processing**: Robust error handling and retry logic

## 3. Content Processing Pipeline Integration

### **Integration Pattern**
```python
# Pipeline Pattern
class ContentProcessingPipeline:
    def __init__(self):
        self.song_embedder = SongEmbedder()
        self.video_analyzer = VideoAnalyzer()
        self.image_processor = ImageProcessor()
        self.clusterer = ContentClusterer()
    
    def process_content(self, content_data):
        # 1. Extract features from content
        # 2. Generate embeddings
        # 3. Perform analysis
        # 4. Cluster similar content
        # 5. Return processed data
```

### **Key Integration Points**
- **Song Embedding**: 384-dimensional embeddings for music similarity
- **Video Analysis**: Shot detection and keyframe extraction
- **Image Processing**: Face detection and content clustering
- **Content Clustering**: Intelligent grouping of similar content

### **Usage Example**
```python
# Initialize pipeline
pipeline = ContentProcessingPipeline()

# Process video content
result = pipeline.process_content({
    "type": "video",
    "path": "input_video.mp4",
    "metadata": {"title": "Sample Video"}
})

# Get processed data
clusters = result["clusters"]
embeddings = result["embeddings"]
metadata = result["metadata"]
```

### **Integration Benefits**
- **Intelligent Analysis**: Advanced ML-based content analysis
- **Similarity Search**: Fast content matching using FAISS
- **Content Organization**: Automatic clustering and categorization
- **Quality Assessment**: Multi-dimensional content quality scoring

## 4. Cross-Service Integration Patterns

### **Unified Content Creation Workflow**
```python
class UnifiedContentCreator:
    def __init__(self):
        self.voice_adapter = HumeVoiceAdapter(api_key)
        self.video_adapter = RunwareVideoAdapter(api_token)
        self.content_pipeline = ContentProcessingPipeline()
    
    async def create_multimedia_content(self, text_prompt):
        # 1. Generate voice narration
        voice_data = self.voice_adapter.generate_voice(text_prompt)
        
        # 2. Generate video content
        video_data = await self.video_adapter.generate_video(text_prompt)
        
        # 3. Process and enhance content
        processed_data = self.content_pipeline.process_content({
            "video": video_data,
            "audio": voice_data,
            "text": text_prompt
        })
        
        # 4. Return unified content package
        return {
            "video": video_data,
            "audio": voice_data,
            "metadata": processed_data["metadata"],
            "clusters": processed_data["clusters"]
        }
```

### **Error Handling and Resilience**
```python
class ResilientIntegrationManager:
    def __init__(self):
        self.services = {
            "hume": HumeVoiceAdapter(api_key),
            "runware": RunwareVideoAdapter(api_token),
            "content": ContentProcessingPipeline()
        }
        self.fallback_services = {
            "hume": AlternativeVoiceService(),
            "runware": AlternativeVideoService()
        }
    
    async def execute_with_fallback(self, service_name, operation, *args, **kwargs):
        try:
            # Try primary service
            return await getattr(self.services[service_name], operation)(*args, **kwargs)
        except Exception as e:
            # Log error and try fallback
            logger.error(f"Primary service failed: {e}")
            return await getattr(self.fallback_services[service_name], operation)(*args, **kwargs)
```

## 5. Data Flow Integration

### **Content Creation Data Flow**
```
User Input → Content Planning → AI Generation → Processing → Enhancement → Output
     ↓              ↓              ↓            ↓           ↓          ↓
Text Prompt → Template Selection → Voice/Video → Analysis → Clustering → Final Content
     ↓              ↓              ↓            ↓           ↓          ↓
JSON Config → AI Service Calls → Raw Content → ML Analysis → Metadata → Packaged Content
```

### **Song Embedding Data Flow**
```
Song Data → Feature Extraction → Embedding Generation → FAISS Index → Similarity Search
     ↓              ↓                    ↓                ↓              ↓
Spotify API → Audio Analysis → 384-dim Vector → Search Index → Content Matching
     ↓              ↓                    ↓                ↓              ↓
Metadata → Waveform → Embedding → Index Update → Query Processing → Results
```

### **Video Processing Data Flow**
```
Video Input → Shot Detection → Keyframe Extraction → Analysis → Metadata → Clustering
     ↓              ↓                ↓               ↓         ↓          ↓
Raw Video → Scene Changes → Representative Frames → ML Analysis → JSON → Grouping
     ↓              ↓                ↓               ↓         ↓          ↓
File Upload → FFmpeg Processing → Frame Analysis → Feature Extraction → Clustering → Output
```

## 6. Configuration Management

### **Environment Configuration**
```python
# config.py
class IntegrationConfig:
    # Hume.ai Configuration
    HUME_API_KEY = os.getenv("HUME_API_KEY")
    HUME_SECRET = os.getenv("HUME_SECRET")
    
    # Runware.ai Configuration
    RUNWARE_API_TOKEN = os.getenv("RUNWARE_API_TOKEN")
    RUNWARE_BASE_URL = "https://api.runware.ai/v1"
    
    # Content Processing Configuration
    SONG_EMBEDDING_MODEL = "all-MiniLM-L12-v2"
    FAISS_INDEX_PATH = "Assets/spotify/embeddings"
    
    # Processing Configuration
    MAX_CONCURRENT_TASKS = 5
    REQUEST_TIMEOUT = 30
    RETRY_ATTEMPTS = 3
```

### **Service Discovery**
```python
class ServiceRegistry:
    def __init__(self):
        self.services = {}
        self.health_checks = {}
    
    def register_service(self, name, adapter, health_check):
        self.services[name] = adapter
        self.health_checks[name] = health_check
    
    def get_healthy_service(self, name):
        if self.health_checks[name]():
            return self.services[name]
        else:
            raise ServiceUnavailableError(f"Service {name} is not healthy")
```

## 7. Testing Integration

### **Integration Testing Strategy**
```python
class IntegrationTestSuite:
    def __init__(self):
        self.test_data = self.load_test_data()
        self.expected_results = self.load_expected_results()
    
    async def test_voice_generation(self):
        # Test Hume.ai integration
        voice_data = await self.voice_adapter.generate_voice(
            self.test_data["voice"]["text"],
            self.test_data["voice"]["voice_id"]
        )
        assert len(voice_data) > 0
        assert voice_data.startswith(b'RIFF')  # WAV file header
    
    async def test_video_generation(self):
        # Test Runware.ai integration
        video_data = await self.video_adapter.generate_video(
            self.test_data["video"]["prompt"]
        )
        assert len(video_data) > 0
        assert video_data.startswith(b'\x00\x00\x00')  # MP4 file header
    
    async def test_content_processing(self):
        # Test content processing pipeline
        result = await self.content_pipeline.process_content(
            self.test_data["content"]
        )
        assert "embeddings" in result
        assert "clusters" in result
        assert "metadata" in result
```

## 8. Monitoring and Observability

### **Integration Metrics**
```python
class IntegrationMetrics:
    def __init__(self):
        self.metrics = {
            "api_calls": Counter("api_calls_total", ["service", "endpoint"]),
            "api_duration": Histogram("api_duration_seconds", ["service"]),
            "api_errors": Counter("api_errors_total", ["service", "error_type"]),
            "content_processed": Counter("content_processed_total", ["type"])
        }
    
    def record_api_call(self, service, endpoint, duration, success):
        self.metrics["api_calls"].labels(service=service, endpoint=endpoint).inc()
        self.metrics["api_duration"].labels(service=service).observe(duration)
        if not success:
            self.metrics["api_errors"].labels(service=service, error_type="api_error").inc()
```

### **Health Checks**
```python
class HealthChecker:
    def __init__(self):
        self.checks = {
            "hume": self.check_hume_health,
            "runware": self.check_runware_health,
            "content_pipeline": self.check_content_pipeline_health
        }
    
    async def check_hume_health(self):
        try:
            # Test API connectivity
            voices = await self.hume_adapter.get_voices()
            return len(voices) > 0
        except Exception:
            return False
    
    async def check_runware_health(self):
        try:
            # Test API connectivity
            status = await self.runware_adapter.check_status()
            return status == "healthy"
        except Exception:
            return False
```

## 9. Best Practices

### **Integration Best Practices**
1. **Modular Design**: Keep integrations separate and loosely coupled
2. **Error Handling**: Implement comprehensive error handling and retry logic
3. **Configuration Management**: Use environment variables for sensitive data
4. **Testing**: Write comprehensive integration tests
5. **Monitoring**: Monitor integration health and performance
6. **Documentation**: Document all integration points and configurations

### **Performance Optimization**
1. **Async Processing**: Use async/await for I/O operations
2. **Caching**: Cache frequently accessed data
3. **Connection Pooling**: Reuse connections where possible
4. **Batch Processing**: Process multiple items together when possible
5. **Resource Management**: Properly manage resources and connections

### **Security Considerations**
1. **API Key Management**: Secure storage and rotation of API keys
2. **Input Validation**: Validate all inputs before processing
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Audit Logging**: Log all integration activities
5. **Data Encryption**: Encrypt sensitive data in transit and at rest

---

This integration guide provides a comprehensive overview of how our AI services work together to create a powerful content creation platform. Each integration is designed to be robust, scalable, and maintainable.
