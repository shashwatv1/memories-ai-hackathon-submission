# Hume AI Voice Generation Testing

This directory contains tests for Hume AI's voice generation API to understand how to integrate emotional voice synthesis into the ProdAgent system.

## Overview

The goal is to test Hume AI's EVI (Empathic Voice Interface) API to generate emotional voice narrations for video content.

## Files

- `hume_voice_client.py` - Client for Hume AI voice generation
- `test_hume_api.py` - Test script to show API calls and responses
- `requirements.txt` - Python dependencies

## API Credentials

The following environment variables are used:
- `HUME_API_KEY` - Hume AI API key
- `HUME_SECRET` - Hume AI secret key

## Testing

Run the API test to see what calls are being made:

```bash
cd /Users/shash/Documents/GitHub/AI_solutions/7_claude_code/20250118_003_hume_voice_generation
python test_hume_api.py
```

This will show you:
1. What API endpoints are being called
2. What request payloads are sent
3. What responses are received
4. Any authentication or rate limiting issues

## Expected Features

- Emotional voice generation with different emotions (happy, sad, calm, etc.)
- Intensity control (0.0-1.0)
- Voice style selection (narrator, conversational, etc.)
- Audio format options (MP3, WAV, etc.)

## Integration Goal

Once we understand the API, we'll integrate this into the ProdAgent system to add emotional voice narrations to generated videos.
