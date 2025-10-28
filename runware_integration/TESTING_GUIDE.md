# Runware 4-Parameter Testing Guide

## Overview
This guide explains how to test the Runware video generation implementation with the 4 simplified parameters: **prompt**, **negative_prompt**, **duration**, and **resolution**.

## Test Files

### 1. `test_4_parameters.py` - Parameter Validation Tests
Tests all parameter combinations and validation rules without generating actual videos.

**What it tests:**
- Prompt variations (simple, detailed, action, cinematic)
- Negative prompt functionality (with/without, empty values)
- Duration options (5s, 10s valid; 6s invalid)
- Resolution options (landscape, portrait, square valid; unsupported invalid)
- Parameter combinations (portrait+10s+negative, square+5s+detailed, etc.)
- Error scenarios (invalid duration, resolution, empty prompts)
- Validation rules (duration must be 5 or 10, resolution must be supported)

**Run command:**
```bash
python test_4_parameters.py
```

**Expected output:**
```
🧪 Starting Runware 4-Parameter Test Suite
Testing: prompt, negative_prompt, duration, resolution
============================================================

🔍 Running test_prompt_variations...
✅ test_prompt_variations: PASSED

🔍 Running test_negative_prompt...
✅ test_negative_prompt: PASSED

🔍 Running test_duration_options...
✅ test_duration_options: PASSED

🔍 Running test_resolution_options...
✅ test_resolution_options: PASSED

🔍 Running test_parameter_combinations...
✅ test_parameter_combinations: PASSED

🔍 Running test_error_scenarios...
✅ test_error_scenarios: PASSED

🔍 Running test_validation_rules...
✅ test_validation_rules: PASSED

============================================================
📊 TEST SUMMARY
============================================================
Total Tests: 7
✅ Passed: 7
❌ Failed: 0
⚠️  Errors: 0
Success Rate: 100.0%
```

### 2. `test_with_polling.py` - End-to-End Tests
Tests the complete workflow: generate → poll → download → validate.

**What it tests:**
- Basic 5s landscape video generation
- Portrait 10s video generation  
- Square 5s video generation
- Complete API workflow (submit → poll → download)
- File validation (size, format, content)
- Error handling during generation

**Run command:**
```bash
python test_with_polling.py
```

**Expected output:**
```
🚀 Runware E2E Test with Polling
Testing complete workflow: generate → poll → download → validate
============================================================

🎬 Test 1: Basic 5s Landscape Video
----------------------------------------
   📝 Prompt: A beautiful sunset over mountains
   🚫 Negative: blurry, dark, low quality
   ⏱️  Duration: 5s
   📐 Resolution: 1920x1080
   🔄 Step 1: Submitting video generation request...
   📋 Task UUID: 12345678-1234-1234-1234-123456789012
   🔍 Step 2: Polling for completion...
   ⏳ Polling task 12345678-1234-1234-1234-123456789012 (max 20 attempts, 15s delay)
   📊 Attempt 1: Status = processing
   ⏰ Waiting 15 seconds...
   📊 Attempt 2: Status = success
   🎬 Video URL: https://...
   📥 Step 3: Downloading video...
   📥 Downloading to: output/generated_videos/Basic_5s_Landscape_Video_20250118_123456.mp4
   📊 File size: 6,234,567 bytes (5.95 MB)
   ✅ Step 4: Validating video file...
✅ Basic 5s Landscape Video: SUCCESS

============================================================
📊 E2E TEST SUMMARY
============================================================
Total Tests: 3
✅ Successful: 3
❌ Failed: 0
Success Rate: 100.0%

🎬 Generated Videos:
  - Basic 5s Landscape Video: 6,234,567 bytes
  - Portrait 10s Video: 12,456,789 bytes
  - Square 5s Video: 5,123,456 bytes
```

## Test Results Interpretation

### ✅ Success Criteria
- **Parameter Tests**: All validation rules pass
- **E2E Tests**: Videos generate, download, and validate successfully
- **File Validation**: Generated videos are valid MP4 files with reasonable sizes
- **Error Handling**: Invalid parameters are properly rejected

### ❌ Failure Indicators
- **API Errors**: 400/500 status codes from Runware API
- **Validation Failures**: Invalid parameters not caught by validation
- **Download Failures**: Videos fail to download or are corrupted
- **File Issues**: Generated files are empty, wrong format, or unreasonably sized

### ⚠️ Common Issues

#### 1. API Connection Issues
```
❌ API Error: 401 - Unauthorized
```
**Solution**: Check API token is correct and active

#### 2. Parameter Validation Failures
```
❌ Duration must be 5 or 10 seconds, got: 6
```
**Solution**: Use only supported duration values (5 or 10)

#### 3. Resolution Validation Failures
```
❌ Invalid resolution: 1280x720. Supported: 1920x1080, 1080x1920, 1080x1080
```
**Solution**: Use only supported resolution combinations

#### 4. Polling Timeouts
```
❌ Max attempts (20) reached
```
**Solution**: Check Runware dashboard for task status, may need longer polling

#### 5. Download Failures
```
❌ Download failed: Connection timeout
```
**Solution**: Check network connection, try again

## Parameter Validation Rules

### Duration
- **Valid**: 5, 10
- **Invalid**: Any other number (3, 6, 15, etc.)
- **Default**: 5

### Resolution
- **Valid Combinations**:
  - Landscape: 1920x1080
  - Portrait: 1080x1920
  - Square: 1080x1080
- **Invalid**: Any other combination (1280x720, 1920x720, etc.)
- **Default**: 1920x1080

### Prompt
- **Required**: Yes (will use image name if not provided)
- **Type**: String
- **Example**: "A beautiful sunset over mountains"

### Negative Prompt
- **Required**: No (defaults to "blurry, low quality, distorted")
- **Type**: String or None
- **Example**: "blurry, dark, low quality"

## File Validation

### Expected File Sizes
- **5s videos**: 3-10 MB
- **10s videos**: 6-20 MB
- **Too small**: < 1 MB (likely corrupted)
- **Too large**: > 100 MB (unusual, may indicate issues)

### File Format
- **Extension**: .mp4
- **Codec**: H.264 (standard)
- **Container**: MP4

## Troubleshooting

### 1. Test Image Issues
If PIL is not available, the test will look for existing images:
```bash
# Install PIL if needed
pip install pillow
```

### 2. API Rate Limits
If you hit rate limits, wait between tests:
```bash
# Wait 30 seconds between E2E tests
sleep 30
python test_with_polling.py
```

### 3. Network Issues
If downloads fail, check:
- Internet connection
- Firewall settings
- Proxy configuration

### 4. File System Issues
If file operations fail, check:
- Disk space (need ~100 MB free)
- Write permissions in output directory
- File path length limits

## Performance Benchmarks

### Expected Performance
- **API Response**: < 5 seconds
- **Video Generation**: 1-3 minutes
- **File Download**: < 1 minute
- **Total E2E Time**: 2-5 minutes per video

### Resource Usage
- **Memory**: < 100 MB during tests
- **Disk**: ~50 MB for test images and videos
- **Network**: ~10-20 MB per video download

## Success Metrics

### Parameter Tests
- **Target**: 100% pass rate
- **Critical**: All validation rules must pass
- **Acceptable**: 90%+ pass rate

### E2E Tests  
- **Target**: 100% success rate
- **Critical**: Videos must download and validate
- **Acceptable**: 80%+ success rate

### File Quality
- **Target**: All files valid MP4 format
- **Critical**: No corrupted downloads
- **Acceptable**: < 10% download failures

## Next Steps After Testing

1. **If all tests pass**: Ready for integration with video generation agent
2. **If some tests fail**: Fix issues before integration
3. **If many tests fail**: Review API configuration and network setup

## Integration Notes

The 4-parameter interface is designed to be simple and robust:
- **prompt**: Core creative control
- **negative_prompt**: Quality control
- **duration**: Length control (5s for testing, 10s for production)
- **resolution**: Format control (landscape/portrait/square)

This matches the user's existing video generation agent requirements and provides the essential controls needed for video generation workflows.
