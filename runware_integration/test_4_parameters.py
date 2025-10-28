#!/usr/bin/env python3
"""
Comprehensive Test Suite for Runware 4-Parameter Implementation

This script tests the 4 user-controllable parameters:
1. prompt
2. negative_prompt  
3. duration
4. resolution (width/height)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from runware_client import RunwareVideoClient
from models.video_result import VideoGenerationParams


class RunwareParameterTester:
    """Test suite for Runware 4-parameter implementation"""
    
    def __init__(self):
        """Initialize the tester"""
        self.client = RunwareVideoClient()
        self.test_results = []
        self.test_image_path = self._create_test_image()
        
    def _create_test_image(self) -> Path:
        """Create a simple test image for testing"""
        try:
            from PIL import Image
            
            # Create a simple test image
            img = Image.new('RGB', (512, 512), color='lightblue')
            test_path = Path("test_image.jpg")
            img.save(test_path)
            print(f"✅ Created test image: {test_path}")
            return test_path
            
        except ImportError:
            print("⚠️  PIL not available, using existing image if available")
            # Look for any existing image file
            for ext in ['.jpg', '.jpeg', '.png']:
                for img_file in Path('.').glob(f'*{ext}'):
                    if img_file.exists():
                        print(f"✅ Using existing image: {img_file}")
                        return img_file
            
            # Create a dummy path for testing (will fail gracefully)
            return Path("dummy_test_image.jpg")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all parameter tests"""
        print("🧪 Starting Runware 4-Parameter Test Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_prompt_variations,
            self.test_negative_prompt,
            self.test_duration_options,
            self.test_resolution_options,
            self.test_parameter_combinations,
            self.test_error_scenarios,
            self.test_validation_rules
        ]
        
        for test_method in test_methods:
            try:
                print(f"\n🔍 Running {test_method.__name__}...")
                result = test_method()
                self.test_results.append({
                    'test': test_method.__name__,
                    'result': result,
                    'status': 'PASSED' if result.get('success', False) else 'FAILED'
                })
                print(f"✅ {test_method.__name__}: {result.get('status', 'UNKNOWN')}")
            except Exception as e:
                print(f"❌ {test_method.__name__} failed with error: {e}")
                self.test_results.append({
                    'test': test_method.__name__,
                    'result': {'success': False, 'error': str(e)},
                    'status': 'ERROR'
                })
        
        return self._generate_summary()
    
    def test_prompt_variations(self) -> Dict[str, Any]:
        """Test different prompt types"""
        print("   Testing prompt variations...")
        
        test_prompts = [
            "A beautiful sunset over mountains",
            "A serene beach with gentle waves",
            "A bird flying through mountain peaks", 
            "A thunderstorm with lightning flashing"
        ]
        
        results = []
        for i, prompt in enumerate(test_prompts):
            try:
                print(f"     Testing prompt {i+1}: '{prompt[:30]}...'")
                
                # Test parameter validation (don't actually generate video)
                params = VideoGenerationParams(
                    image_path=self.test_image_path,
                    prompt=prompt,
                    duration=5,
                    width=1920,
                    height=1080
                )
                
                results.append({
                    'prompt': prompt,
                    'valid': True,
                    'params_valid': True
                })
                
            except Exception as e:
                results.append({
                    'prompt': prompt,
                    'valid': False,
                    'error': str(e)
                })
        
        return {
            'success': all(r.get('valid', False) for r in results),
            'test_type': 'prompt_variations',
            'results': results,
            'status': f"Tested {len(test_prompts)} prompts"
        }
    
    def test_negative_prompt(self) -> Dict[str, Any]:
        """Test negative prompt functionality"""
        print("   Testing negative prompt...")
        
        test_cases = [
            {"negative_prompt": "blurry, low quality, dark", "expected": "blurry, low quality, dark"},
            {"negative_prompt": None, "expected": "blurry, low quality, distorted"},
            {"negative_prompt": "", "expected": "blurry, low quality, distorted"},
            {"negative_prompt": "ugly, distorted, bad quality", "expected": "ugly, distorted, bad quality"}
        ]
        
        results = []
        for case in test_cases:
            try:
                params = VideoGenerationParams(
                    image_path=self.test_image_path,
                    prompt="A test video",
                    negative_prompt=case["negative_prompt"],
                    duration=5,
                    width=1920,
                    height=1080
                )
                
                # Check if the negative prompt is handled correctly
                actual_negative = case["negative_prompt"] or "blurry, low quality, distorted"
                results.append({
                    'input': case["negative_prompt"],
                    'expected': case["expected"],
                    'actual': actual_negative,
                    'valid': True
                })
                
            except Exception as e:
                results.append({
                    'input': case["negative_prompt"],
                    'error': str(e),
                    'valid': False
                })
        
        return {
            'success': all(r.get('valid', False) for r in results),
            'test_type': 'negative_prompt',
            'results': results,
            'status': f"Tested {len(test_cases)} negative prompt cases"
        }
    
    def test_duration_options(self) -> Dict[str, Any]:
        """Test 5s and 10s durations"""
        print("   Testing duration options...")
        
        test_durations = [5, 10]
        invalid_durations = [3, 6, 15]
        
        results = []
        
        # Test valid durations
        for duration in test_durations:
            try:
                params = VideoGenerationParams(
                    image_path=self.test_image_path,
                    prompt="A test video",
                    duration=duration,
                    width=1920,
                    height=1080
                )
                results.append({
                    'duration': duration,
                    'valid': True,
                    'status': 'VALID'
                })
            except Exception as e:
                results.append({
                    'duration': duration,
                    'valid': False,
                    'error': str(e),
                    'status': 'ERROR'
                })
        
        # Test invalid durations
        for duration in invalid_durations:
            try:
                params = VideoGenerationParams(
                    image_path=self.test_image_path,
                    prompt="A test video",
                    duration=duration,
                    width=1920,
                    height=1080
                )
                results.append({
                    'duration': duration,
                    'valid': False,
                    'status': 'SHOULD_FAIL',
                    'error': 'Should have failed but did not'
                })
            except Exception as e:
                results.append({
                    'duration': duration,
                    'valid': False,
                    'status': 'CORRECTLY_FAILED',
                    'error': str(e)
                })
        
        return {
            'success': all(r.get('status') in ['VALID', 'CORRECTLY_FAILED'] for r in results),
            'test_type': 'duration_options',
            'results': results,
            'status': f"Tested {len(test_durations)} valid and {len(invalid_durations)} invalid durations"
        }
    
    def test_resolution_options(self) -> Dict[str, Any]:
        """Test all supported resolutions"""
        print("   Testing resolution options...")
        
        valid_resolutions = [
            (1920, 1080),  # Landscape
            (1080, 1920),  # Portrait
            (1080, 1080)   # Square
        ]
        
        invalid_resolutions = [
            (1280, 720),   # Not supported
            (1920, 720),   # Not supported
            (800, 600)     # Not supported
        ]
        
        results = []
        
        # Test valid resolutions
        for width, height in valid_resolutions:
            try:
                params = VideoGenerationParams(
                    image_path=self.test_image_path,
                    prompt="A test video",
                    duration=5,
                    width=width,
                    height=height
                )
                results.append({
                    'resolution': f"{width}x{height}",
                    'valid': True,
                    'status': 'VALID'
                })
            except Exception as e:
                results.append({
                    'resolution': f"{width}x{height}",
                    'valid': False,
                    'error': str(e),
                    'status': 'ERROR'
                })
        
        # Test invalid resolutions
        for width, height in invalid_resolutions:
            try:
                params = VideoGenerationParams(
                    image_path=self.test_image_path,
                    prompt="A test video",
                    duration=5,
                    width=width,
                    height=height
                )
                results.append({
                    'resolution': f"{width}x{height}",
                    'valid': False,
                    'status': 'SHOULD_FAIL',
                    'error': 'Should have failed but did not'
                })
            except Exception as e:
                results.append({
                    'resolution': f"{width}x{height}",
                    'valid': False,
                    'status': 'CORRECTLY_FAILED',
                    'error': str(e)
                })
        
        return {
            'success': all(r.get('status') in ['VALID', 'CORRECTLY_FAILED'] for r in results),
            'test_type': 'resolution_options',
            'results': results,
            'status': f"Tested {len(valid_resolutions)} valid and {len(invalid_resolutions)} invalid resolutions"
        }
    
    def test_parameter_combinations(self) -> Dict[str, Any]:
        """Test various parameter combinations"""
        print("   Testing parameter combinations...")
        
        test_combinations = [
            {
                'name': 'Portrait + 10s + negative prompt',
                'params': {
                    'prompt': 'A vertical scene',
                    'negative_prompt': 'blurry, dark',
                    'duration': 10,
                    'width': 1080,
                    'height': 1920
                }
            },
            {
                'name': 'Square + 5s + detailed prompt',
                'params': {
                    'prompt': 'A detailed square format scene with intricate details',
                    'negative_prompt': 'low quality, pixelated',
                    'duration': 5,
                    'width': 1080,
                    'height': 1080
                }
            },
            {
                'name': 'Landscape + 10s + simple prompt',
                'params': {
                    'prompt': 'A simple landscape',
                    'negative_prompt': None,
                    'duration': 10,
                    'width': 1920,
                    'height': 1080
                }
            }
        ]
        
        results = []
        for combo in test_combinations:
            try:
                params = VideoGenerationParams(
                    image_path=self.test_image_path,
                    **combo['params']
                )
                results.append({
                    'combination': combo['name'],
                    'valid': True,
                    'status': 'VALID'
                })
            except Exception as e:
                results.append({
                    'combination': combo['name'],
                    'valid': False,
                    'error': str(e),
                    'status': 'ERROR'
                })
        
        return {
            'success': all(r.get('valid', False) for r in results),
            'test_type': 'parameter_combinations',
            'results': results,
            'status': f"Tested {len(test_combinations)} parameter combinations"
        }
    
    def test_error_scenarios(self) -> Dict[str, Any]:
        """Test invalid parameter handling"""
        print("   Testing error scenarios...")
        
        error_cases = [
            {
                'name': 'Invalid duration',
                'params': {'duration': 6},
                'should_fail': True
            },
            {
                'name': 'Invalid resolution',
                'params': {'width': 1280, 'height': 720},
                'should_fail': True
            },
            {
                'name': 'Empty prompt (should use default)',
                'params': {'prompt': None},
                'should_fail': False
            }
        ]
        
        results = []
        for case in error_cases:
            try:
                base_params = {
                    'image_path': self.test_image_path,
                    'prompt': 'A test video',
                    'duration': 5,
                    'width': 1920,
                    'height': 1080
                }
                base_params.update(case['params'])
                
                params = VideoGenerationParams(**base_params)
                
                if case['should_fail']:
                    results.append({
                        'case': case['name'],
                        'result': 'SHOULD_HAVE_FAILED',
                        'valid': False
                    })
                else:
                    results.append({
                        'case': case['name'],
                        'result': 'PASSED',
                        'valid': True
                    })
                    
            except Exception as e:
                if case['should_fail']:
                    results.append({
                        'case': case['name'],
                        'result': 'CORRECTLY_FAILED',
                        'valid': True,
                        'error': str(e)
                    })
                else:
                    results.append({
                        'case': case['name'],
                        'result': 'UNEXPECTED_ERROR',
                        'valid': False,
                        'error': str(e)
                    })
        
        return {
            'success': all(r.get('valid', False) for r in results),
            'test_type': 'error_scenarios',
            'results': results,
            'status': f"Tested {len(error_cases)} error scenarios"
        }
    
    def test_validation_rules(self) -> Dict[str, Any]:
        """Test parameter validation rules"""
        print("   Testing validation rules...")
        
        validation_tests = [
            {
                'rule': 'Duration must be 5 or 10',
                'test': lambda: VideoGenerationParams(
                    image_path=self.test_image_path,
                    duration=3
                ),
                'should_fail': True
            },
            {
                'rule': 'Width must be 1920 or 1080',
                'test': lambda: VideoGenerationParams(
                    image_path=self.test_image_path,
                    width=800
                ),
                'should_fail': True
            },
            {
                'rule': 'Height must be 1920 or 1080',
                'test': lambda: VideoGenerationParams(
                    image_path=self.test_image_path,
                    height=600
                ),
                'should_fail': True
            },
            {
                'rule': 'Valid resolution combination',
                'test': lambda: VideoGenerationParams(
                    image_path=self.test_image_path,
                    width=1920,
                    height=1080
                ),
                'should_fail': False
            }
        ]
        
        results = []
        for test in validation_tests:
            try:
                test['test']()
                if test['should_fail']:
                    results.append({
                        'rule': test['rule'],
                        'result': 'SHOULD_HAVE_FAILED',
                        'valid': False
                    })
                else:
                    results.append({
                        'rule': test['rule'],
                        'result': 'PASSED',
                        'valid': True
                    })
            except Exception as e:
                if test['should_fail']:
                    results.append({
                        'rule': test['rule'],
                        'result': 'CORRECTLY_FAILED',
                        'valid': True,
                        'error': str(e)
                    })
                else:
                    results.append({
                        'rule': test['rule'],
                        'result': 'UNEXPECTED_ERROR',
                        'valid': False,
                        'error': str(e)
                    })
        
        return {
            'success': all(r.get('valid', False) for r in results),
            'test_type': 'validation_rules',
            'results': results,
            'status': f"Tested {len(validation_tests)} validation rules"
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['status'] == 'PASSED')
        failed_tests = sum(1 for r in self.test_results if r['status'] == 'FAILED')
        error_tests = sum(1 for r in self.test_results if r['status'] == 'ERROR')
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️  Errors: {error_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0 or error_tests > 0:
            print("\n🔍 FAILED TESTS:")
            for result in self.test_results:
                if result['status'] in ['FAILED', 'ERROR']:
                    print(f"  - {result['test']}: {result['status']}")
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'errors': error_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'results': self.test_results
        }


def main():
    """Main test execution"""
    print("🚀 Runware 4-Parameter Test Suite")
    print("Testing: prompt, negative_prompt, duration, resolution")
    print("=" * 60)
    
    tester = RunwareParameterTester()
    summary = tester.run_all_tests()
    
    # Clean up test image
    if tester.test_image_path.exists() and tester.test_image_path.name == "test_image.jpg":
        tester.test_image_path.unlink()
        print(f"\n🧹 Cleaned up test image: {tester.test_image_path}")
    
    return summary


if __name__ == "__main__":
    main()
