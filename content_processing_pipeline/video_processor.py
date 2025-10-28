from decorators import *
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import subprocess
from IPython.display import display, HTML
from shot_detector import ShotDetectionEnhancer

def process_single_video(video_path, motion_threshold=5, min_segment_duration=2.0, 
                         sample_rate=5, motion_leniency=3, smoothing_window=5, 
                         save_segments=True, visualize=True, enable_shot_detection=True, 
                         shot_threshold=0.7):
    """
    Analyzes a single video for motion, identifies good segments, and saves them.
    
    Parameters:
    - video_path: Path to the video file
    - motion_threshold: Threshold for motion detection (higher values mean more motion is allowed)
    - min_segment_duration: Minimum duration in seconds for a segment to be kept
    - sample_rate: Analyze every Nth frame for efficiency
    - motion_leniency: Number of consecutive high-motion frames to tolerate before breaking a segment
    - smoothing_window: Size of rolling window for motion smoothing
    - save_segments: Whether to save the extracted segments
    - visualize: Whether to generate visualizations
    - enable_shot_detection: Whether to apply shot detection to split segments at scene boundaries
    - shot_threshold: Histogram similarity threshold for shot detection (0-1, lower = more sensitive)
    
    Returns:
    - Dictionary with analysis results and segment information
    """
    print(f"Processing: {os.path.basename(video_path)}")
    
    # Analyze video for motion
    metrics_df, sample_frames = analyze_video_motion(video_path, sample_rate)
    
    if metrics_df.empty:
        print(f"Error: Failed to analyze {os.path.basename(video_path)}")
        return None
    
    # Apply smoothing to motion scores
    if smoothing_window > 1:
        metrics_df['smooth_motion'] = metrics_df['motion_score'].rolling(window=smoothing_window, center=True).mean()
        # Fill NaN values at the beginning and end
        metrics_df['smooth_motion'] = metrics_df['smooth_motion'].fillna(metrics_df['motion_score'])
    else:
        metrics_df['smooth_motion'] = metrics_df['motion_score']
    
    # Identify good segments with leniency
    good_segments = identify_good_segments_with_leniency(
        metrics_df, 
        motion_threshold=motion_threshold,
        min_duration=min_segment_duration,
        leniency=motion_leniency
    )
    
    # Apply shot detection enhancement if enabled
    if enable_shot_detection and good_segments:
        try:
            enhancement_result = ShotDetectionEnhancer.enhance_video_processing(
                video_path, good_segments, 
                hist_threshold=shot_threshold,
                min_shot_duration=min_segment_duration
            )
            good_segments = enhancement_result['enhanced_segments']
            print(f"Shot detection: Split into {len(good_segments)} single-shot segments")
        except Exception as e:
            print(f"Shot detection failed, using original segments: {e}")
    
    # Create result object
    result = {
        'video_name': os.path.basename(video_path),
        'full_path': video_path,
        'duration': metrics_df.iloc[-1]['timestamp'] if not metrics_df.empty else 0,
        'avg_motion_score': metrics_df['motion_score'].mean() if not metrics_df.empty else 0,
        'pct_high_motion': len(metrics_df[metrics_df['smooth_motion'] > motion_threshold]) / len(metrics_df) * 100 if not metrics_df.empty else 0,
        'good_segments': good_segments,
        'metrics_df': metrics_df,
        'sample_frames': sample_frames
    }
    
    # Save segments if requested
    if save_segments and good_segments:
        output_files = save_video_segments(result)
        result['output_files'] = output_files
    
    # Visualize if requested
    if visualize:
        visualize_motion_analysis(
            os.path.basename(video_path), 
            metrics_df, 
            good_segments, 
            sample_frames,
            use_smooth_motion=True  # Use the smoothed motion values
        )
        
        # Display segment information
        display(HTML(generate_segments_table(good_segments, os.path.basename(video_path))))
        display(HTML(video_report_summary(os.path.basename(video_path), metrics_df, good_segments)))
    
    return result

def process_video_folder(folder_path, motion_threshold=5, min_segment_duration=2.0, 
                        sample_rate=5, motion_leniency=3, smoothing_window=5,
                        save_segments=True, visualize=True, enable_shot_detection=True, 
                        shot_threshold=0.7):
    """
    Processes all videos in a folder, analyzes them for motion, and saves good segments.
    
    Parameters:
    - folder_path: Path to the folder containing videos
    - motion_threshold: Threshold for motion detection
    - min_segment_duration: Minimum duration in seconds for a segment to be kept
    - sample_rate: Analyze every Nth frame for efficiency
    - motion_leniency: Number of consecutive high-motion frames to tolerate
    - smoothing_window: Size of rolling window for motion smoothing
    - save_segments: Whether to save the extracted segments
    - visualize: Whether to generate visualizations
    
    Returns:
    - List of results for each processed video
    """
    print(f"Looking for videos in {folder_path}")
    
    # Find video files
    video_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check for common video extensions
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')):
                full_path = os.path.join(root, file)
                print(f"Found video: {full_path}")
                video_paths.append(full_path)
    
    if not video_paths:
        print(f"No videos found in {folder_path}")
        return []
    
    print(f"Found {len(video_paths)} videos to analyze")
    
    # Process each video
    all_results = []
    for video_path in video_paths:
        result = process_single_video(
            video_path,
            motion_threshold=motion_threshold,
            min_segment_duration=min_segment_duration,
            sample_rate=sample_rate,
            motion_leniency=motion_leniency,
            smoothing_window=smoothing_window,
            save_segments=save_segments,
            visualize=visualize,
            enable_shot_detection=enable_shot_detection,
            shot_threshold=shot_threshold
        )
        
        if result:
            all_results.append(result)
    
    # Generate summary table
    if all_results and visualize:
        print("\n### Video Analysis Summary ###")
        display(HTML(generate_summary_table(all_results)))
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame([
            {
                'video_name': r['video_name'],
                'duration': r['duration'],
                'avg_motion_score': r['avg_motion_score'],
                'pct_high_motion': r['pct_high_motion'],
                'num_good_segments': len(r['good_segments']),
                'usable_duration': sum(segment['duration'] for segment in r['good_segments']),
                'pct_usable': (sum(segment['duration'] for segment in r['good_segments']) / r['duration'] * 100) if r['duration'] > 0 else 0
            } 
            for r in all_results
        ])
        
        # results_df.to_csv('video_analysis_results.csv', index=False)
        tracking_dir = os.path.join(config.User_ID, config.Chat_ID)
        os.makedirs(tracking_dir, exist_ok=True)
        results_csv_path = os.path.join(tracking_dir, 'video_analysis_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print("\nResults saved to video_analysis_results.csv")
    
    return all_results

# Helper functions

def analyze_video_motion(video_path, sample_rate=5):
    """
    Analyzes video frames for motion.
    Returns frame-by-frame metrics and sample frames.
    """
    video_path = str(video_path)
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return pd.DataFrame(), []
    
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # For progress tracking
    n_frames_to_process = total_frames // sample_rate
    
    metrics = []
    sample_frames = []
    frame_count = 0
    prev_frame = None
    
    print(f"Analyzing {os.path.basename(video_path)} - {n_frames_to_process} frames to process")
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            # Save key frames for visualization (every 50th analyzed frame)
            if frame_count % (sample_rate * 50) == 0:
                sample_frames.append({
                    'frame': frame_count,
                    'timestamp': frame_count/fps,
                    'image': frame.copy()
                })
            
            # Calculate motion score for this frame
            motion_score = detect_motion(frame, prev_frame) if prev_frame is not None else 0
            
            # Add to metrics
            metrics.append({
                'frame': frame_count,
                'timestamp': frame_count/fps,
                'motion_score': motion_score
            })
        
        prev_frame = frame.copy()
        frame_count += 1
    
    video.release()
    return pd.DataFrame(metrics), sample_frames

def detect_motion(current_frame, previous_frame):
    """
    Detects motion between consecutive frames using optical flow.
    Higher values indicate more motion.
    """
    if current_frame is None or previous_frame is None:
        return 0
        
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude of 2D vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Average magnitude as motion score
    motion_score = np.mean(magnitude)
    
    return motion_score

def identify_good_segments_with_leniency(metrics_df, motion_threshold=5, min_duration=2.0, leniency=3, sample_rate=5):
    """
    Identifies segments with acceptable motion, applying leniency for brief motion spikes.
    
    Parameters:
    - metrics_df: DataFrame with motion metrics
    - motion_threshold: Threshold for motion detection
    - min_duration: Minimum duration for a segment to be kept
    - leniency: Number of consecutive high-motion frames to tolerate
    - sample_rate: Sample rate used for analysis (for duration calculation)
    
    Returns:
    - List of good segments (dictionaries with start_time, end_time, duration)
    """
    if metrics_df.empty:
        return []
    
    # Create a new column that identifies potential breaks in segments
    # A break is identified only if we have more than 'leniency' consecutive high-motion frames
    metrics_df['high_motion'] = metrics_df['smooth_motion'] > motion_threshold
    
    # Initialize a counter for consecutive high motion frames
    consecutive_high_motion = 0
    # Initialize a column for actual breaks
    metrics_df['segment_break'] = False
    
    # Iterate through frames to identify actual breaks
    for i in range(len(metrics_df)):
        if metrics_df.iloc[i]['high_motion']:
            consecutive_high_motion += 1
            if consecutive_high_motion > leniency:
                metrics_df.loc[metrics_df.index[i], 'segment_break'] = True
        else:
            consecutive_high_motion = 0
    
    # Now identify good segments based on segment_break column
    segments = []
    in_segment = True
    start_idx = 0
    
    for i in range(len(metrics_df)):
        if metrics_df.iloc[i]['segment_break'] and in_segment:
            # End of a good segment
            end_idx = i - 1
            
            if end_idx >= start_idx:  # Valid segment
                start_time = metrics_df.iloc[start_idx]['timestamp']
                end_time = metrics_df.iloc[end_idx]['timestamp']
                duration = end_time - start_time
                
                if duration >= min_duration:
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'start_frame': metrics_df.iloc[start_idx]['frame'],
                        'end_frame': metrics_df.iloc[end_idx]['frame']
                    })
            
            in_segment = False
        elif not metrics_df.iloc[i]['segment_break'] and not in_segment:
            # Start of a new good segment
            start_idx = i
            in_segment = True
    
    # Handle the last segment if we're still in a good segment at the end
    if in_segment and start_idx < len(metrics_df):
        end_idx = len(metrics_df) - 1
        start_time = metrics_df.iloc[start_idx]['timestamp']
        end_time = metrics_df.iloc[end_idx]['timestamp']
        duration = end_time - start_time
        
        if duration >= min_duration:
            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'start_frame': metrics_df.iloc[start_idx]['frame'],
                'end_frame': metrics_df.iloc[end_idx]['frame']
            })
    
    return segments


def save_video_segments(result, output_dir=None):
    """
    Intelligently saves the good segments from a video to individual files.
    Tries stream copying first for speed, falls back to re-encoding only if needed.
    
    Parameters:
    - result: Result dictionary from process_single_video
    - output_dir: Directory to save segments (if None, uses same directory as input)
    
    Returns:
    - List of output file paths
    """
    video_path = result['full_path']
    good_segments = result['good_segments']
    
    if not good_segments:
        print(f"No good segments to save for {os.path.basename(video_path)}")
        return []
    
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = []
    
    # First, check source video format and codec info
    try:
        print(f"Analyzing source video: {os.path.basename(video_path)}")
        format_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,profile,width,height',
            '-of', 'json',
            video_path
        ]
        format_output = subprocess.run(format_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(format_output.stdout)
        
        # Get audio info
        audio_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'json',
            video_path
        ]
        audio_output = subprocess.run(audio_cmd, capture_output=True, text=True)
        has_audio = False
        
        try:
            audio_info = json.loads(audio_output.stdout)
            has_audio = 'streams' in audio_info and len(audio_info['streams']) > 0
            audio_codec = audio_info['streams'][0]['codec_name'] if has_audio else None
            print(f"Audio codec: {audio_codec if has_audio else 'No audio stream'}")
        except:
            print("Could not determine audio codec, assuming no audio")
        
        # Check if video codec is standard and well-supported
        if 'streams' in video_info and video_info['streams']:
            video_codec = video_info['streams'][0]['codec_name']
            print(f"Video codec: {video_codec}")
            
            # Define well-supported formats
            standard_video_codecs = ['h264', 'avc1', 'avc', 'libx264']
            standard_audio_codecs = ['aac', 'mp3', 'mp4a']
            
            use_stream_copy = video_codec.lower() in standard_video_codecs
            if has_audio:
                use_stream_copy = use_stream_copy 
                # and audio_codec.lower() in standard_audio_codecs
                
            print(f"Using stream copy: {use_stream_copy}")
        else:
            # Codec information unavailable, default to re-encoding
            print("Could not determine video codec, defaulting to re-encoding")
            use_stream_copy = False
    except Exception as e:
        print(f"Error checking video format: {e}")
        print("Defaulting to re-encoding for safety")
        use_stream_copy = False
    
    # Now process each segment
    for i, segment in enumerate(good_segments):
        start_time = segment['start_time']
        duration = segment['duration']
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_segment_{i+1}.mp4")
        
        # Skip if file already exists
        if os.path.exists(output_file):
            print(f"Segment file already exists: {output_file}")
            output_files.append(output_file)
            continue
        
        success = False
        
        # Try stream copying first if codec is standard
        if use_stream_copy:
            try:
                print(f"Attempting stream copy for segment {i+1}/{len(good_segments)} ({start_time:.2f}s - {start_time+duration:.2f}s)")
                
                copy_cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ss', f"{start_time:.2f}",
                    '-t', f"{duration:.2f}",
                    '-c:v', 'copy',       # Stream copy for video
                    '-c:a', 'copy' if has_audio else '-an',  # Stream copy for audio if present
                    '-avoid_negative_ts', '1',  # Helps with keyframe alignment
                    '-movflags', '+faststart',  # Optimize for web streaming
                    '-y',                 # Overwrite if exists
                    output_file
                ]
                
                subprocess.run(copy_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                
                # Verify the output file is valid
                verify_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    output_file
                ]
                subprocess.run(verify_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # If we get here, stream copy was successful
                output_files.append(output_file)
                print(f"Saved with stream copy: {output_file}")
                success = True
                
            except subprocess.CalledProcessError as e:
                print(f"Stream copy failed: {e}")
                # If file was created but is invalid, remove it
                if os.path.exists(output_file):
                    os.remove(output_file)
                print("Falling back to re-encoding...")
            except Exception as e:
                print(f"Error during stream copy: {e}")
                if os.path.exists(output_file):
                    os.remove(output_file)
        
        # Fall back to re-encoding if stream copy failed or wasn't attempted
        if not success:
            try:
                print(f"Re-encoding segment {i+1}/{len(good_segments)} ({start_time:.2f}s - {start_time+duration:.2f}s)")
                
                encode_cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ss', f"{start_time:.2f}",
                    '-t', f"{duration:.2f}",
                    '-c:v', 'libx264',  # Re-encode video
                    '-profile:v', 'high422',  # Main profile for better compatibility
                    '-preset', 'fast',  # Balance between speed and quality
                    '-crf', '23',       # Reasonable quality setting
                    '-c:a', 'aac' if has_audio else '-an',  # Re-encode audio if present
                    '-b:a', '128k' if has_audio else '',  # Audio bitrate if needed
                    '-movflags', '+faststart',  # Optimize for web streaming
                    '-y',               # Overwrite if exists
                    output_file
                ]
                
                # Clean up empty elements in command
                encode_cmd = [x for x in encode_cmd if x]
                
                subprocess.run(encode_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output_files.append(output_file)
                print(f"Saved with re-encoding: {output_file}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error extracting segment with re-encoding: {e}")
                print(f"Error output: {e.stderr.decode() if hasattr(e, 'stderr') else 'No error output'}")
                
                # Try one last fallback with ultrafast preset and lower quality
                try:
                    print("Attempting last-resort extraction with ultrafast preset...")
                    
                    fallback_cmd = [
                        'ffmpeg',
                        '-i', video_path,
                        '-ss', f"{start_time:.2f}",
                        '-t', f"{duration:.2f}",
                        '-c:v', 'libx264',
                        '-preset', 'ultrafast',  # Fastest possible preset
                        '-crf', '28',           # Lower quality for speed
                        '-c:a', 'aac' if has_audio else '-an',
                        '-b:a', '96k' if has_audio else '',
                        '-movflags', '+faststart',
                        '-y',
                        output_file
                    ]
                    
                    # Clean up empty elements in command
                    fallback_cmd = [x for x in fallback_cmd if x]
                    
                    subprocess.run(fallback_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    output_files.append(output_file)
                    print(f"Saved with fallback encoding: {output_file}")
                    
                except subprocess.CalledProcessError as e2:
                    print(f"All extraction methods failed for this segment")
    
    return output_files

def visualize_motion_analysis(video_name, metrics_df, good_segments, sample_frames=None, use_smooth_motion=True):
    """
    Creates visualization of motion metrics and identified good segments.
    """
    if metrics_df.empty:
        print(f"No data to visualize for {video_name}")
        return
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot metrics
    timestamps = metrics_df['timestamp']
    
    # Plot motion scores - use smoothed version if available and requested
    motion_column = 'smooth_motion' if 'smooth_motion' in metrics_df.columns and use_smooth_motion else 'motion_score'
    ax.plot(timestamps, metrics_df[motion_column], 'g-', label='Motion Score')
    
    # Also plot raw motion as a thin line if using smoothed
    if motion_column == 'smooth_motion':
        ax.plot(timestamps, metrics_df['motion_score'], 'g-', alpha=0.3, linewidth=0.8, label='Raw Motion')
    
    ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Motion Threshold')  # Motion threshold
    ax.set_ylabel('Motion Score')
    ax.set_xlabel('Time (seconds)')
    ax.set_title(f'Motion Analysis: {video_name}')
    ax.legend()
    
    # Highlight good segments
    for segment in good_segments:
        ax.axvspan(segment['start_time'], segment['end_time'], alpha=0.2, color='green')
    
    # Highlight high motion areas (areas to remove)
    ax.fill_between(timestamps, 0, metrics_df[motion_column], 
                    where=metrics_df[motion_column] > 5, 
                    color='red', alpha=0.3)
    
    plt.tight_layout()
    
    # Show sample frames if available
    if sample_frames and len(sample_frames) > 0:
        # Create a new figure for frames
        num_frames = min(len(sample_frames), 4)  # Show up to 4 frames
        fig2, axes = plt.subplots(1, num_frames, figsize=(16, 3))
        
        # Handle case with only one frame
        if num_frames == 1:
            axes = [axes]
        
        for i in range(num_frames):
            frame = sample_frames[i]['image']
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(frame_rgb)
            axes[i].axis('off')
            
            # Check if this frame is in a good segment
            frame_time = sample_frames[i]['timestamp']
            in_good_segment = False
            for segment in good_segments:
                if segment['start_time'] <= frame_time <= segment['end_time']:
                    in_good_segment = True
                    break
            
            title_color = 'green' if in_good_segment else 'red'
            axes[i].set_title(f"Time: {frame_time:.1f}s", color=title_color)
        
        plt.tight_layout()
    
    plt.show()

def generate_segments_table(good_segments, video_name):
    """
    Generates an HTML table of good segments.
    """
    html = f"""
    <style>
        .segments-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin-bottom: 20px;
        }}
        .segments-table th, .segments-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .segments-table th {{
            background-color: #f2f2f2;
        }}
        .segments-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
    <h3>Usable Segments for {video_name}</h3>
    <table class="segments-table">
        <tr>
            <th>Segment #</th>
            <th>Start Time</th>
            <th>End Time</th>
            <th>Duration</th>
        </tr>
    """
    
    if not good_segments:
        html += '<tr><td colspan="4" style="text-align:center;">No usable segments found</td></tr>'
    else:
        for i, segment in enumerate(good_segments):
            html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{segment['start_time']:.2f}s</td>
                <td>{segment['end_time']:.2f}s</td>
                <td>{segment['duration']:.2f}s</td>
            </tr>
            """
    
    html += "</table>"
    
    # Add summary statistics
    if good_segments:
        total_duration = sum(segment['duration'] for segment in good_segments)
        avg_duration = total_duration / len(good_segments)
        html += f"""
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <strong>Total Segments:</strong> {len(good_segments)}<br>
            <strong>Total Usable Duration:</strong> {total_duration:.2f}s<br>
            <strong>Average Segment Duration:</strong> {avg_duration:.2f}s
        </div>
        """
    
    return html

def video_report_summary(video_name, metrics_df, good_segments):
    """
    Generates a summary report for a video.
    """
    if metrics_df.empty:
        return f"<p>No data available for {video_name}</p>"
    
    # Calculate total video duration
    total_duration = metrics_df.iloc[-1]['timestamp']
    
    # Calculate usable footage duration
    usable_duration = sum(segment['duration'] for segment in good_segments)
    
    # Calculate percentage of usable footage
    pct_usable = (usable_duration / total_duration) * 100 if total_duration > 0 else 0
    
    # Use smoothed motion if available
    motion_column = 'smooth_motion' if 'smooth_motion' in metrics_df.columns else 'motion_score'
    
    # Calculate percentage of high motion frames
    pct_high_motion = len(metrics_df[metrics_df[motion_column] > 5]) / len(metrics_df) * 100 if not metrics_df.empty else 0
    
    html = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px; border: 1px solid #ddd; padding: 20px; border-radius: 5px;">
        <h2 style="color: #333;">Video Analysis Summary: {video_name}</h2>
        
        <div style="margin-bottom: 20px;">
            <table style="width:100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Total Duration</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{total_duration:.2f} seconds</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Usable Footage</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{usable_duration:.2f} seconds ({pct_usable:.1f}%)</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Number of Usable Segments</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{len(good_segments)}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">High Motion Frames</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{pct_high_motion:.1f}% of frames</td>
                </tr>
            </table>
        </div>
        
        <div style="margin-top: 20px;">
            <h3>Recommendation</h3>
    """
    
    if pct_usable < 30:
        html += "<p style='color: red;'>This video has limited usable footage due to excessive motion. Consider reshooting with more stable camera work.</p>"
    elif pct_usable < 70:
        html += "<p style='color: orange;'>This video has moderate usable footage. Trimming out high motion segments will improve quality significantly.</p>"
    else:
        html += "<p style='color: green;'>This video has good usable footage. Minor trimming of high motion segments will result in high-quality content.</p>"
    
    html += """
        </div>
    </div>
    """
    
    return html

def generate_summary_table(results):
    """
    Generates an HTML summary table for all analyzed videos.
    """
    html = """
    <style>
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
        }
        .summary-table th, .summary-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .summary-table th {
            background-color: #f2f2f2;
        }
        .summary-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .good { color: green; }
        .bad { color: red; }
        .warning { color: orange; }
    </style>
    <table class="summary-table">
        <tr>
            <th>Video</th>
            <th>Duration</th>
            <th>Usable Segments</th>
            <th>Usable Duration</th>
            <th>Usable %</th>
            <th>High Motion %</th>
        </tr>
    """
    
    for result in results:
        video_name = result['video_name']
        total_duration = result['duration']
        usable_segments = len(result['good_segments'])
        
        # Calculate usable duration
        usable_duration = sum(segment['duration'] for segment in result['good_segments'])
        pct_usable = (usable_duration / total_duration) * 100 if total_duration > 0 else 0
        
        # Determine color class based on usable percentage
        if pct_usable >= 70:
            usable_class = "good"
        elif pct_usable >= 30:
            usable_class = "warning"
        else:
            usable_class = "bad"
            
        html += f"""
        <tr>
            <td>{video_name}</td>
            <td>{total_duration:.2f}s</td>
            <td>{usable_segments}</td>
            <td>{usable_duration:.2f}s</td>
            <td class="{usable_class}">{pct_usable:.1f}%</td>
            <td>{result['pct_high_motion']:.1f}%</td>
        </tr>
        """
    
    html += "</table>"
    return html

from GoogleServiceAPI import GoogleDriveServiceOperations
@log_execution
@handle_errors
@sync_with_drive(
    inputs=[
        'processed_videos.json'
    ],
    outputs=[
        'processed_videos.json',
        'video_analysis_results.csv'
    ]
)
def process_videos_with_segmentation(
    service_account_file: str = 'service-account.json',
    target_folder_id: str = '1be5p41JtvBbSxKpBaxrcotet0RZCzt5Y',  # Your shared drive folder ID
    motion_threshold: int = 5,
    min_segment_duration: float = 1.0,
    motion_leniency: int = 2,
    smoothing_window: int = 3,
    resume: bool = True
) -> Dict[str, Any]:
    """
    Downloads videos from Google Drive links, processes them with segmentation,
    and uploads the resulting segments to a folder structure:
    {target_folder}/User_ID/Chat_ID/Processed_Videos/
    
    Args:
        service_account_file: Path to service account credentials file
        target_folder_id: ID of the target folder (shared drive folder)
        motion_threshold: Threshold for motion detection
        min_segment_duration: Minimum duration in seconds for a segment to be kept
        motion_leniency: Number of consecutive high-motion frames to tolerate
        smoothing_window: Size of rolling window for motion smoothing
        resume: Whether to resume from previous run if interrupted
        
    Returns:
        Dictionary with processing results
    """
    import tempfile
    import shutil
    from pathlib import Path
    import time
    
    # Get the links directly from config
    import config
    drive_urls = config.links
    User_ID = config.User_ID
    Chat_ID = config.Chat_ID
    
    if not drive_urls:
        return {
            "status": "error",
            "message": "No Drive URLs provided in configuration"
        }
    
    # Create a simple tracking file if resume is enabled
    # processed_videos = set()
    # # tracking_file = "processed_videos.json"
    # tracking_dir = os.path.join(User_ID, Chat_ID)
    # os.makedirs(tracking_dir, exist_ok=True)
    # tracking_file = os.path.join(tracking_dir, "processed_videos.json")
    
    # # Load previously processed videos if resuming
    # if resume and os.path.exists(tracking_file):
    #     try:
    #         with open(tracking_file, 'r') as f:
    #             processed_videos = set(json.load(f))
    #             print(f"Resuming previous run. Found {len(processed_videos)} already processed videos.")
    #     except Exception as e:
    #         print(f"Warning: Could not load tracking data: {str(e)}")
    
    # Create a simple tracking file if resume is enabled
    processed_videos = set()
    # tracking_file = "processed_videos.json"
    tracking_dir = os.path.join(User_ID, Chat_ID)
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_file = os.path.join(tracking_dir, "processed_videos.json")

    # Load previously processed videos if resuming
    if resume and os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r') as f:
                processed_videos = set(json.load(f))
                print(f"Resuming previous run. Found {len(processed_videos)} already processed videos.")
        except Exception as e:
            print(f"Warning: Could not load tracking file: {str(e)}")

    # Get list of videos already in Jushn folder to skip
    # existing_video_names = set()
    # try:
    #     drive_ops = GoogleDriveServiceOperations(service_account_file)
    #     # jushn_folder_id = find_folder_by_name(drive_ops, target_folder_id, "Jushn")
    #     jushn_folder_id = find_folder_by_name(drive_ops.service, "Jushn", target_folder_id)

        
    #     if jushn_folder_id:
    #         video_query = f"'{jushn_folder_id}' in parents and mimeType contains 'video/' and trashed=false"
    #         existing_videos = drive_ops.list_files(query=video_query)
            
    #         if existing_videos['status'] == 'success' and existing_videos['data'].get('files'):
    #             existing_video_names = {file['name'] for file in existing_videos['data']['files']}
    #             print(f"Found {len(existing_video_names)} videos already in Jushn folder - will skip these during processing")
    # except Exception as e:
    #     print(f"Warning: Could not check Jushn folder: {str(e)}")
    # Get list of videos already in Jushn folder to skip
    existing_video_names = set()
    try:
        # Initialize Drive operations
        drive_ops = GoogleDriveServiceOperations(service_account_file)
        
        # Find ALL Jushn folders recursively and collect videos from them
        jushn_folders_query = f"name='Jushn' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        jushn_folders_result = drive_ops.list_files(query=jushn_folders_query)

        if jushn_folders_result['status'] == 'success' and jushn_folders_result['data'].get('files'):
            for jushn_folder in jushn_folders_result['data']['files']:
                jushn_folder_id = jushn_folder['id']
                
                # Check for videos in this Jushn folder
                video_query = f"'{jushn_folder_id}' in parents and mimeType contains 'video/' and trashed=false"
                existing_videos = drive_ops.list_files(query=video_query)
                
                if existing_videos['status'] == 'success' and existing_videos['data'].get('files'):
                    for video in existing_videos['data']['files']:
                        existing_video_names.add(video['name'])
            
            if existing_video_names:
                print(f"Found {len(existing_video_names)} videos already in Jushn folders - will skip these during processing")
            else:
                print("No existing videos found in any Jushn folders")
        else:
            print("No Jushn folders found")
    except Exception as e:
        print(f"Warning: Could not check Jushn folder: {str(e)}")
    
    try:
        # Initialize Drive operations
        # drive_ops = GoogleDriveServiceOperations(service_account_file)
        
        # Verify target folder exists
        try:
            folder_info = drive_ops.service.files().get(
                fileId=target_folder_id,
                fields="name,driveId",
                supportsAllDrives=True
            ).execute()
            folder_name = folder_info.get('name', 'Unknown')
            drive_id = folder_info.get('driveId')
            
            print(f"Target folder found: {folder_name} (ID: {target_folder_id})")
            if drive_id:
                print(f"Folder is in shared drive with ID: {drive_id}")
        except Exception as e:
            raise ValueError(f"Failed to access target folder: {str(e)}")
        
        # Create or find User_ID folder in target folder
        user_folder_id = find_or_create_folder(drive_ops.service, User_ID, target_folder_id)
        print(f"User folder ID: {user_folder_id}")
        
        # Create or find Chat_ID folder in User_ID folder
        chat_folder_id = find_or_create_folder(drive_ops.service, Chat_ID, user_folder_id)
        print(f"Chat folder ID: {chat_folder_id}")
        
        # Create or find Processed_Videos folder in Chat_ID folder
        processed_folder_name = "Processed_Videos"
        processed_folder_id = find_or_create_folder(drive_ops.service, processed_folder_name, chat_folder_id)
        print(f"Processed videos folder ID: {processed_folder_id}")
        
        # Create a temporary directory for downloads and processing
        temp_dir = tempfile.mkdtemp(prefix="video_processing_")
        downloads_dir = os.path.join(temp_dir, "downloads")
        segments_dir = os.path.join(temp_dir, "segments")
        os.makedirs(downloads_dir, exist_ok=True)
        os.makedirs(segments_dir, exist_ok=True)
        
        # Process timestamp for logs and folder names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results tracking
        processing_log = {
            "process_id": f"video_processing_{timestamp}",
            "start_time": datetime.now().isoformat(),
            "source_drives": drive_urls,
            "downloaded_videos": [],
            "processed_segments": [],
            "uploaded_files": [],
            "errors": []
        }
        
        # Track folder structure in log
        processing_log["folder_structure"] = {
            "target_folder": {
                "id": target_folder_id,
                "name": folder_name
            },
            "user_folder": {
                "id": user_folder_id,
                "name": User_ID
            },
            "chat_folder": {
                "id": chat_folder_id,
                "name": Chat_ID
            },
            "processed_folder": {
                "id": processed_folder_id,
                "name": processed_folder_name,
                "link": f"https://drive.google.com/drive/folders/{processed_folder_id}"
            }
        }
        
        # Process each drive URL
        for url_index, url in enumerate(drive_urls):
            try:
                folder_id = url.split('folders/')[-1].split('?')[0]
                
                # Get folder name
                folder_info = drive_ops.service.files().get(
                    fileId=folder_id,
                    fields='name',
                    supportsAllDrives=True
                ).execute()
                folder_name = folder_info.get('name', f"Folder_{url_index+1}")
                
                print(f"Processing folder: {folder_name}")
                
                # Get video files
                # query = "mimeType contains 'video/' and trashed=false"
                # files = drive_ops.list_files(
                #     query=f"'{folder_id}' in parents and {query}"
                # )
                
                # if files['status'] != 'success' or not files['data'].get('files'):
                #     print(f"No video files found in folder: {folder_name}")
                #     continue
                
                # # Track progress
                # total_videos = len(files['data'].get('files', []))
                # videos_processed = 0
                
                # # Download and process videos
                # for file in files['data'].get('files', []):
                # Get all video files recursively
                all_files = []
                print(f"Recursively searching for videos in folder: {folder_name}")

                def get_videos_recursive(current_folder_id):
                    # Get direct video files
                    query = "mimeType contains 'video/' and trashed=false"
                    files_response = drive_ops.list_files(
                        query=f"'{current_folder_id}' in parents and {query}"
                    )
                    
                    if files_response['status'] == 'success' and files_response['data'].get('files'):
                        all_files.extend(files_response['data']['files'])
                    
                    # Get subfolders
                    folders_query = f"'{current_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                    folders_response = drive_ops.list_files(query=folders_query)
                    
                    if folders_response['status'] == 'success' and folders_response['data'].get('files'):
                        for subfolder in folders_response['data']['files']:
                            print(f"Searching subfolder: {subfolder['name']}")
                            get_videos_recursive(subfolder['id'])

                # Start recursive search
                get_videos_recursive(folder_id)

                # No videos found in this folder or its subfolders
                if not all_files:
                    print(f"No video files found in folder: {folder_name} or its subfolders")
                    continue

                # Track progress
                total_videos = len(all_files)
                videos_processed = 0

                # Download and process videos
                for file in all_files:
                    try:
                        video_id = file['id']
                        video_name = file['name']
                        
                        # Skip if already processed (for resumption)
                        # if resume and video_id in processed_videos:
                        #     print(f"Skipping already processed video: {video_name}")
                        #     videos_processed += 1
                        #     continue

                        # Skip if already processed (for resumption)
                        if resume and video_id in processed_videos:
                            print(f"Skipping already processed video: {video_name}")
                            videos_processed += 1
                            continue

                        # Skip if video already exists in Jushn folder
                        if video_name in existing_video_names:
                            print(f"Skipping video - exists in Jushn folder: {video_name}")
                            videos_processed += 1
                            continue
                        
                        # Skip non-video files
                        if not any(video_name.lower().endswith(ext) for ext in 
                                   ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')):
                            continue
                        
                        videos_processed += 1
                        print(f"Processing video ({videos_processed}/{total_videos}): {video_name}")
                        
                        # Create a folder for this video
                        video_folder = os.path.join(downloads_dir, video_name.replace('.', '_'))
                        os.makedirs(video_folder, exist_ok=True)
                        
                        # Download the video
                        video_path = os.path.join(video_folder, video_name)
                        download_result = drive_ops.download_file(video_id, video_path)
                        
                        if download_result['status'] != 'success':
                            raise Exception(f"Failed to download video: {download_result.get('message')}")
                        
                        # Log the download
                        processing_log['downloaded_videos'].append({
                            "file_name": video_name,
                            "drive_id": video_id,
                            "folder_name": folder_name,
                            "local_path": video_path
                        })
                        
                        # Process video with segmentation
                        print(f"Running segmentation on {video_name}")
                        
                        # Set the output directory for segments
                        video_segments_dir = os.path.join(segments_dir, video_name.replace('.', '_'))
                        os.makedirs(video_segments_dir, exist_ok=True)
                        
                        # Downscale the video for faster processing
                        downscaled_video_path = downscale_video_for_processing(video_path, target_width=960)

                        # Process the downscaled video for motion analysis ONLY
                        result = process_single_video(
                            downscaled_video_path,  # Use downscaled version for analysis
                            motion_threshold=motion_threshold,
                            min_segment_duration=min_segment_duration,
                            sample_rate=10,
                            motion_leniency=motion_leniency,
                            smoothing_window=smoothing_window,
                            save_segments=False,  # Don't save segments yet, just analyze
                            visualize=False,
                            enable_shot_detection=enable_shot_detection,
                            shot_threshold=shot_threshold
                        )

                        # If analysis was successful, now extract segments from the original video
                        if result and 'good_segments' in result and result['good_segments']:
                            # Modify the result to use the original video path for segment extraction
                            result['full_path'] = video_path  # Point to original video 
                            
                            # Now extract segments from the original high-quality video
                            output_files = save_video_segments(result)
                            result['output_files'] = output_files
                            print(f"Extracted {len(output_files)} segments from original video")

                        # Clean up downscaled video
                        if os.path.exists(downscaled_video_path) and downscaled_video_path != video_path:
                            try:
                                os.remove(downscaled_video_path)
                                print(f"Removed downscaled video after analysis")
                            except Exception as e:
                                print(f"Warning: Could not remove downscaled video: {str(e)}")

                        # # Process the video
                        # result = process_single_video(
                        #     video_path,
                        #     motion_threshold=motion_threshold,
                        #     min_segment_duration=min_segment_duration,
                        #     sample_rate=5,
                        #     motion_leniency=motion_leniency,
                        #     smoothing_window=smoothing_window,
                        #     save_segments=True,
                        #     visualize=False
                        # )
                        
                        if result is None or not result.get('good_segments'):
                            print(f"No good segments found in {video_name}")
                            
                            # Mark as processed even if no segments found
                            if resume:
                                processed_videos.add(video_id)
                                with open(tracking_file, 'w') as f:
                                    json.dump(list(processed_videos), f)
                            
                            # Clean up this video's files
                            try:
                                shutil.rmtree(video_folder)
                            except Exception as e:
                                print(f"Warning: Failed to clean up files for {video_name}: {str(e)}")
                            
                            continue
                        
                        # Upload segments to the processed_folder_id
                        segments_uploaded = 0
                        total_segments = len(result.get('output_files', []))
                        
                        if 'output_files' in result and result['output_files']:
                            for segment_path in result['output_files']:
                                # Check if segment file exists before attempting upload
                                if not os.path.exists(segment_path):
                                    print(f"Warning: Segment file not found: {segment_path}")
                                    continue
                                    
                                # Allow a moment for file system to complete writing
                                time.sleep(0.5)

                                # Check video length - Skip if longer than 60 seconds
                                try:
                                    cap = cv2.VideoCapture(segment_path)
                                    fps = cap.get(cv2.CAP_PROP_FPS)
                                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    duration = frame_count / fps if fps > 0 else 0
                                    cap.release()
                                    
                                    if duration > 600:  # Skip videos longer than 600 seconds (10 minute)
                                        print(f"Segment {os.path.basename(segment_path)} duration is {duration:.2f}s - moving to Long_Videos folder")
        
                                        # Create or find Long_Videos folder inside the same parent folder
                                        long_videos_folder_name = "Long_Videos"
                                        long_videos_folder_id = find_or_create_folder(drive_ops.service, long_videos_folder_name, chat_folder_id)
                                        print(f"Long videos folder ID: {long_videos_folder_id}")
                                        
                                        # Clean up the video name
                                        clean_video_name = os.path.splitext(video_name)[0].replace('.mp4', '').replace('.MP4', '')
                                        clean_video_name = ''.join(c for c in clean_video_name if c.isalnum() or c in ' _-')
                                        
                                        # Upload to Long_Videos folder instead
                                        upload_result = drive_ops.upload_file(
                                            segment_path,
                                            parent_folder_id=long_videos_folder_id,
                                            file_name=f"{clean_video_name}_{segment_name}"
                                        )
                                        
                                        if upload_result['status'] == 'success':
                                            processing_log['uploaded_files'].append({
                                                "original_video": video_name,
                                                "segment_name": segment_name,
                                                "drive_id": upload_result['data']['id'],
                                                "drive_link": f"https://drive.google.com/file/d/{upload_result['data']['id']}/view",
                                                "folder": "Long_Videos"
                                            })
                                            print(f"Uploaded long segment to Long_Videos folder: {segment_name}")
                                        else:
                                            processing_log['errors'].append({
                                                "type": "upload_error",
                                                "video": video_name,
                                                "segment": segment_name,
                                                "error": upload_result.get('message')
                                            })
                                            print(f"Error uploading long segment: {segment_name}")

                                        continue
                                except Exception as e:
                                    print(f"Error checking video duration for {segment_path}: {str(e)}")
                                    # Proceed with upload anyway if duration check fails

                                
                                segment_name = os.path.basename(segment_path)
                                segments_uploaded += 1
                                
                                # Upload to User_ID/Chat_ID/Processed_Videos folder
                                try:
                                    # upload_result = drive_ops.upload_file(
                                    #     segment_path,
                                    #     parent_folder_id=processed_folder_id,
                                    #     file_name=f"{folder_name}_{segment_name}"
                                    # )

                                    # Clean up the video name by removing extensions and special characters
                                    clean_video_name = os.path.splitext(video_name)[0].replace('.mp4', '').replace('.MP4', '')
                                    clean_video_name = ''.join(c for c in clean_video_name if c.isalnum() or c in ' _-')

                                    upload_result = drive_ops.upload_file(
                                        segment_path,
                                        parent_folder_id=processed_folder_id,
                                        file_name=f"{clean_video_name}_{segment_name}"
                                    )   
                                    
                                    if upload_result['status'] == 'success':
                                        processing_log['uploaded_files'].append({
                                            "original_video": video_name,
                                            "segment_name": segment_name,
                                            "drive_id": upload_result['data']['id'],
                                            "drive_link": f"https://drive.google.com/file/d/{upload_result['data']['id']}/view"
                                        })
                                        
                                        print(f"Uploaded segment {segments_uploaded}/{total_segments}: {segment_name}")
                                    else:
                                        processing_log['errors'].append({
                                            "type": "upload_error",
                                            "video": video_name,
                                            "segment": segment_name,
                                            "error": upload_result.get('message')
                                        })
                                        print(f"Error uploading segment: {segment_name}")
                                except Exception as e:
                                    print(f"Error uploading segment {segment_name}: {str(e)}")
                            
                            # Mark video as processed after all segments are uploaded
                            if resume:
                                processed_videos.add(video_id)
                                with open(tracking_file, 'w') as f:
                                    json.dump(list(processed_videos), f)
                            
                            # Clean up after each video is fully processed
                            try:
                                print(f"Cleaning up temporary files for: {video_name}")
                                shutil.rmtree(video_folder)
                                print(f"Temporary files deleted for: {video_name}")
                            except Exception as e:
                                print(f"Warning: Failed to clean up files for {video_name}: {str(e)}")
                        
                    except Exception as e:
                        error_msg = f"Error processing video {video_name}: {str(e)}"
                        print(error_msg)
                        processing_log['errors'].append({
                            "type": "processing_error",
                            "video": video_name,
                            "error": error_msg
                        })
            
            except Exception as e:
                error_msg = f"Error processing folder URL {url}: {str(e)}"
                print(error_msg)
                processing_log['errors'].append({
                    "type": "folder_error",
                    "url": url,
                    "error": error_msg
                })
        
        # Finalize processing log
        processing_log["end_time"] = datetime.now().isoformat()
        processing_log["duration"] = (datetime.fromisoformat(processing_log["end_time"]) - 
                                     datetime.fromisoformat(processing_log["start_time"])).total_seconds()
        
        # Save processing log
        log_path = os.path.join(temp_dir, f"processing_log_{timestamp}.json")
        with open(log_path, 'w') as f:
            json.dump(processing_log, f, indent=2)
            
        # Upload processing log to the processed_folder_id
        log_upload = drive_ops.upload_file(
            log_path,
            parent_folder_id=processed_folder_id,
            file_name=f"processing_log_{timestamp}.json"
        )
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {str(e)}")
        
        # If successful, clear the tracking data
        if not resume and os.path.exists(tracking_file) and len(processing_log['errors']) == 0:
            try:
                os.remove(tracking_file)
                print("Processing completed successfully. Tracking data cleared.")
            except Exception:
                pass
        
        return {
            "status": "success",
            "process_id": processing_log["process_id"],
            "videos_downloaded": len(processing_log['downloaded_videos']),
            "segments_uploaded": len(processing_log['uploaded_files']),
            "errors": len(processing_log['errors']),
            "folder_structure": {
                "user_id": User_ID,
                "chat_id": Chat_ID,
                "processed_folder": processed_folder_name,
                "processed_folder_id": processed_folder_id,
                "processed_folder_link": f"https://drive.google.com/drive/folders/{processed_folder_id}"
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to process videos: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }

def find_or_create_folder(service, folder_name, parent_id):
    """Find folder by exact name and parent, create if doesn't exist"""
    try:
        # Escape special characters in folder name for the query
        # Most importantly, escape single quotes that would break the query
        escaped_name = folder_name.replace("'", "\\'")
        
        query = f"name='{escaped_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
        
        print(f"Searching for folder '{folder_name}' in parent '{parent_id}'")
        
        # Execute the query
        results = service.files().list(
            q=query,
            fields='files(id, name)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()

        # Handle existing folder
        if results.get('files'):
            # If multiple folders exist with same name under same parent, use the first one
            if len(results['files']) > 1:
                print(f"Multiple folders named '{folder_name}' found under same parent. Using first one.")
            
            folder_id = results['files'][0]['id']
            print(f"Found existing folder: {folder_name} (ID: {folder_id})")
            return folder_id
        
        # Folder doesn't exist, create it
        print(f"Creating new folder: {folder_name} in parent {parent_id}")
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        
        folder = service.files().create(
            body=file_metadata,
            fields='id',
            supportsAllDrives=True
        ).execute()
        
        folder_id = folder.get('id')
        print(f"Created new folder: {folder_name} (ID: {folder_id})")
        return folder_id
        
    except Exception as e:
        print(f"Error in find_or_create_folder for '{folder_name}': {str(e)}")
        # Try a simpler approach as fallback
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            }
            
            folder = service.files().create(
                body=file_metadata,
                fields='id',
                supportsAllDrives=True
            ).execute()
            
            folder_id = folder.get('id')
            print(f"Created folder (after error): {folder_name} (ID: {folder_id})")
            return folder_id
        except Exception as create_error:
            print(f"Failed to create folder: {str(create_error)}")
            raise


import os
import subprocess
import numpy as np
import tempfile
import json
from datetime import datetime
import cv2

def analyze_audio_quality(video_path, silent_threshold=0.01):
    """
    Analyzes the audio quality of a video file.
    
    Parameters:
    - video_path: Path to the video file
    - silent_threshold: RMS threshold below which audio is considered silent
    
    Returns:
    - Dictionary with audio quality metrics
    """
    try:
        # Create a temporary file for the audio extraction
        temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        # Extract audio using ffmpeg
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian format
            '-ar', '44100',  # 44.1 kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite if exists
            temp_audio_path
        ]
        
        # Run the command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Get audio duration
        duration_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        
        duration_output = subprocess.run(duration_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration_info = json.loads(duration_output.stdout)
        duration = float(duration_info['format']['duration'])
        
        # Analyze audio levels using ffmpeg silencedetect filter
        silence_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-af', f'silencedetect=noise={silent_threshold}:d=0.5',  # Detect silence with threshold for at least 0.5s
            '-f', 'null',
            '-'
        ]
        
        silence_output = subprocess.run(silence_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        silence_output_str = silence_output.stderr.decode()
        
        # Parse the silence detection output
        silence_intervals = []
        for line in silence_output_str.split('\n'):
            if 'silence_start' in line:
                start_time = float(line.split('silence_start: ')[1].strip())
                silence_intervals.append({'start': start_time, 'end': None})
            elif 'silence_end' in line and silence_intervals and silence_intervals[-1]['end'] is None:
                parts = line.split('silence_end: ')[1].strip().split(' | ')
                end_time = float(parts[0])
                silence_intervals[-1]['end'] = end_time
                
        # If the last silence interval doesn't have an end, set it to the duration
        if silence_intervals and silence_intervals[-1]['end'] is None:
            silence_intervals[-1]['end'] = duration
        
        # Calculate silence percentage
        silence_duration = sum(interval['end'] - interval['start'] for interval in silence_intervals)
        silence_percentage = (silence_duration / duration) * 100 if duration > 0 else 0
        
        # Calculate average volume using ffmpeg volumedetect filter
        volume_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-filter:a', 'volumedetect',
            '-f', 'null',
            '-'
        ]
        
        volume_output = subprocess.run(volume_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        volume_output_str = volume_output.stderr.decode()
        
        # Parse mean volume
        mean_volume = None
        max_volume = None
        for line in volume_output_str.split('\n'):
            if 'mean_volume' in line:
                mean_volume = float(line.split('mean_volume: ')[1].split(' dB')[0])
            elif 'max_volume' in line:
                max_volume = float(line.split('max_volume: ')[1].split(' dB')[0])
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        return {
            'duration': duration,
            'silence_percentage': silence_percentage,
            'mean_volume': mean_volume,
            'max_volume': max_volume,
            'silence_intervals': silence_intervals,
            'is_poor_quality': silence_percentage > 60 or (mean_volume is not None and mean_volume < -30)
        }
        
    except Exception as e:
        print(f"Error analyzing audio for {video_path}: {str(e)}")
        return {
            'error': str(e),
            'is_poor_quality': True  # Assume poor quality if analysis fails
        }

def split_video_into_subsegments(video_path, output_dir, max_duration=5.0):
    """
    Splits a video into subsegments of maximum specified duration.
    
    Parameters:
    - video_path: Path to the video file
    - output_dir: Directory to save the subsegments
    - max_duration: Maximum duration of each subsegment in seconds
    
    Returns:
    - List of paths to the created subsegments
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        if duration <= max_duration:
            print(f"Video {os.path.basename(video_path)} is already shorter than {max_duration}s, no splitting needed")
            return [video_path]
        
        # Calculate number of segments needed
        num_segments = int(np.ceil(duration / max_duration))
        
        # Base name for output files
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        ext = os.path.splitext(video_path)[1]
        
        subsegment_paths = []
        
        for i in range(num_segments):
            start_time = i * max_duration
            this_duration = min(max_duration, duration - start_time)
            
            output_file = os.path.join(output_dir, f"{base_name}_sub{i+1}{ext}")
            
            # Use ffmpeg to extract segment
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', f"{start_time:.2f}",
                '-t', f"{this_duration:.2f}",
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-y',  # Overwrite if exists
                output_file
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subsegment_paths.append(output_file)
            print(f"Created subsegment {i+1}/{num_segments} for {os.path.basename(video_path)}")
        
        return subsegment_paths
        
    except Exception as e:
        print(f"Error splitting video {video_path}: {str(e)}")
        return []

import os
import subprocess
import numpy as np
import tempfile
import json
from datetime import datetime
import cv2

def analyze_audio_quality(video_path, silent_threshold=0.01):
    """
    Analyzes the audio quality of a video file.
    
    Parameters:
    - video_path: Path to the video file
    - silent_threshold: RMS threshold below which audio is considered silent
    
    Returns:
    - Dictionary with audio quality metrics
    """
    try:
        # Create a temporary file for the audio extraction
        temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        # Extract audio using ffmpeg
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian format
            '-ar', '44100',  # 44.1 kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite if exists
            temp_audio_path
        ]
        
        # Run the command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Get audio duration
        duration_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        
        duration_output = subprocess.run(duration_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration_info = json.loads(duration_output.stdout)
        duration = float(duration_info['format']['duration'])
        
        # Analyze audio levels using ffmpeg silencedetect filter
        silence_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-af', f'silencedetect=noise={silent_threshold}:d=0.5',  # Detect silence with threshold for at least 0.5s
            '-f', 'null',
            '-'
        ]
        
        silence_output = subprocess.run(silence_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        silence_output_str = silence_output.stderr.decode()
        
        # Parse the silence detection output
        silence_intervals = []
        for line in silence_output_str.split('\n'):
            if 'silence_start' in line:
                start_time = float(line.split('silence_start: ')[1].strip())
                silence_intervals.append({'start': start_time, 'end': None})
            elif 'silence_end' in line and silence_intervals and silence_intervals[-1]['end'] is None:
                parts = line.split('silence_end: ')[1].strip().split(' | ')
                end_time = float(parts[0])
                silence_intervals[-1]['end'] = end_time
                
        # If the last silence interval doesn't have an end, set it to the duration
        if silence_intervals and silence_intervals[-1]['end'] is None:
            silence_intervals[-1]['end'] = duration
        
        # Calculate silence percentage
        silence_duration = sum(interval['end'] - interval['start'] for interval in silence_intervals)
        silence_percentage = (silence_duration / duration) * 100 if duration > 0 else 0
        
        # Calculate average volume using ffmpeg volumedetect filter
        volume_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-filter:a', 'volumedetect',
            '-f', 'null',
            '-'
        ]
        
        volume_output = subprocess.run(volume_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        volume_output_str = volume_output.stderr.decode()
        
        # Parse mean volume
        mean_volume = None
        max_volume = None
        for line in volume_output_str.split('\n'):
            if 'mean_volume' in line:
                mean_volume = float(line.split('mean_volume: ')[1].split(' dB')[0])
            elif 'max_volume' in line:
                max_volume = float(line.split('max_volume: ')[1].split(' dB')[0])
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        return {
            'duration': duration,
            'silence_percentage': silence_percentage,
            'mean_volume': mean_volume,
            'max_volume': max_volume,
            'silence_intervals': silence_intervals,
            'is_poor_quality': silence_percentage > 60 or (mean_volume is not None and mean_volume < -30)
        }
        
    except Exception as e:
        print(f"Error analyzing audio for {video_path}: {str(e)}")
        return {
            'error': str(e),
            'is_poor_quality': True  # Assume poor quality if analysis fails
        }

def split_video_into_subsegments(video_path, output_dir, max_duration=5.0):
    """
    Splits a video into subsegments of maximum specified duration.
    
    Parameters:
    - video_path: Path to the video file
    - output_dir: Directory to save the subsegments
    - max_duration: Maximum duration of each subsegment in seconds
    
    Returns:
    - List of paths to the created subsegments
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []  # Return empty list, indicating no subsegments created

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        if duration <= max_duration:
            print(f"Video {os.path.basename(video_path)} is already shorter than {max_duration}s, no splitting needed")
            return [video_path]
        
        # Calculate number of segments needed
        num_segments = int(np.ceil(duration / max_duration))
        
        # Clean base name for output files - ensure we don't duplicate .mp4 extension
        base_name = os.path.basename(video_path)
        if base_name.lower().endswith('.mp4'):
            base_name = os.path.splitext(base_name)[0]
        
        # Make sure there are no duplicate .mp4 in the name
        base_name = base_name.replace('.mp4', '').replace('.MP4', '')
        
        subsegment_paths = []
        
        for i in range(num_segments):
            start_time = i * max_duration
            this_duration = min(max_duration, duration - start_time)
            
            output_file = os.path.join(output_dir, f"{base_name}_sub{i+1}.mp4")
            
            # Use ffmpeg to extract segment
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', f"{start_time:.2f}",
                '-t', f"{this_duration:.2f}",
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-y',  # Overwrite if exists
                output_file
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subsegment_paths.append(output_file)
            print(f"Created subsegment {i+1}/{num_segments} for {os.path.basename(video_path)}")
        
        return subsegment_paths
        
    except Exception as e:
        print(f"Error splitting video {video_path}: {str(e)}")
        return []

def process_segments_by_audio_quality(segments_dir, min_length=5.0, silence_threshold=0.01, poor_volume_threshold=-30, 
                                     silence_percentage_threshold=60, max_subsegment_duration=4.9):
    """
    Processes video segments by audio quality, splitting those with poor audio into subsegments.
    
    Parameters:
    - segments_dir: Directory containing the video segments
    - min_length: Minimum length in seconds to consider for audio quality check
    - silence_threshold: RMS threshold for silence detection
    - poor_volume_threshold: Mean volume threshold in dB below which audio is considered poor
    - silence_percentage_threshold: Percentage of silence above which audio is considered poor
    - max_subsegment_duration: Maximum duration of subsegments in seconds
    
    Returns:
    - Dictionary with processing results
    """
    print(f"Processing segments for audio quality in: {segments_dir}")
    
    # Find all video files in the segments directory
    video_files = []
    for root, dirs, files in os.walk(segments_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        print(f"No video segments found in {segments_dir}")
        return {
            "status": "error",
            "message": "No video segments found"
        }
    
    print(f"Found {len(video_files)} video segments to process")
    
    # Process each video segment
    results = {
        "start_time": datetime.now().isoformat(),
        "segments_processed": [],
        "segments_split": [],
        "errors": []
    }
    
    for video_path in video_files:
        try:
            print(f"Processing segment: {os.path.basename(video_path)}")
            
            # Get video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            segment_result = {
                "segment_name": os.path.basename(video_path),
                "duration": duration,
                "above_min_length": duration >= min_length,
                "processed": False,
                "split": False
            }
            
            # Skip segments shorter than min_length
            if duration < min_length:
                print(f"Skipping segment {os.path.basename(video_path)} - duration {duration:.2f}s below threshold")
                results["segments_processed"].append(segment_result)
                continue
            
            # Analyze audio quality
            audio_quality = analyze_audio_quality(video_path, silent_threshold=silence_threshold)
            segment_result.update({
                "audio_analysis": audio_quality,
                "processed": True
            })
            
            # Check if audio quality is poor and needs splitting
            is_poor_quality = audio_quality.get('is_poor_quality', False)
            if is_poor_quality:
                print(f"Poor audio quality detected in {os.path.basename(video_path)} - splitting into subsegments")
                
                # Create a directory for the subsegments
                subsegments_dir = os.path.join(
                    os.path.dirname(video_path),
                    f"{os.path.splitext(os.path.basename(video_path))[0]}_subsegments"
                )
                
                # Split the video
                subsegment_paths = split_video_into_subsegments(
                    video_path, 
                    subsegments_dir, 
                    max_duration=max_subsegment_duration
                )
                
                segment_result.update({
                    "split": True,
                    "num_subsegments": len(subsegment_paths),
                    "subsegments_dir": subsegments_dir
                })
                
                results["segments_split"].append(segment_result)
            
            results["segments_processed"].append(segment_result)
            
        except Exception as e:
            error_msg = f"Error processing segment {os.path.basename(video_path)}: {str(e)}"
            print(error_msg)
            results["errors"].append({
                "segment": os.path.basename(video_path),
                "error": error_msg
            })
    
    # Finalize results
    results["end_time"] = datetime.now().isoformat()
    results["total_segments"] = len(results["segments_processed"])
    results["segments_with_poor_audio"] = len(results["segments_split"])
    
    # Save results to a log file
    log_path = os.path.join(segments_dir, f"audio_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAudio processing complete. Results saved to {log_path}")
    print(f"Processed {results['total_segments']} segments, split {results['segments_with_poor_audio']} with poor audio quality")
    
    return results

@log_execution
@handle_errors
@sync_with_drive(
    inputs=[
        'processed_audio_segments.json'
    ],
    outputs=[
        'processed_audio_segments.json',
        'audio_processing_log.json'
    ]
)
def process_segments_with_audio_check(
    service_account_file: str = 'service-account.json',
    target_folder_id: str = None,  # Will use from segmentation result if None
    User_ID: str = None,           # Will use from config if None
    Chat_ID: str = None,           # Will use from config if None
    min_audio_segment_length: float = 5.0,
    silence_threshold: float = 0.01,
    poor_volume_threshold: float = -30,
    silence_percentage_threshold: float = 60,
    max_subsegment_duration: float = 4.9,
    resume: bool = True,
    force_split: bool = True  # Force split all videos regardless of audio quality
  # Add resume parameter

) -> Dict[str, Any]:
    """
    This function is meant to be called directly after process_videos_with_segmentation
    to check audio quality of segments and split poor-quality ones.
    
    It reads necessary parameters from config.py like the original function.
    
    Args:
        service_account_file: Path to service account credentials file
        target_folder_id: ID of the target folder (will use from config if None)
        User_ID: User ID for folder structure (will use from config if None)
        Chat_ID: Chat ID for folder structure (will use from config if None)
        min_audio_segment_length: Minimum segment length to consider for audio quality check
        silence_threshold: RMS threshold for silence detection
        poor_volume_threshold: Mean volume threshold in dB below which audio is considered poor
        silence_percentage_threshold: Percentage of silence above which audio is considered poor
        max_subsegment_duration: Maximum duration of subsegments in seconds
        
    Returns:
        Dictionary with processing results
    """
    # Import necessary modules
    import shutil
    import tempfile
    
    # Import config
    import config
    
    # Use values from config if not provided
    if User_ID is None:
        User_ID = config.User_ID
    if Chat_ID is None:
        Chat_ID = config.Chat_ID
    if target_folder_id is None:
        target_folder_id = config.target_folder_id if hasattr(config, 'target_folder_id') else '1be5p41JtvBbSxKpBaxrcotet0RZCzt5Y'
    
    print(f"Processing segments with audio quality check for User: {User_ID}, Chat: {Chat_ID}")


    # Create tracking directory and file
    tracking_dir = os.path.join(User_ID, Chat_ID)
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_file = os.path.join(tracking_dir, "processed_audio_segments.json")
    
    # Load previously processed segments if resuming
    processed_segments = set()
    if resume and os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r') as f:
                processed_segments = set(json.load(f))
                print(f"Resuming previous run. Found {len(processed_segments)} already processed segments.")
        except Exception as e:
            print(f"Warning: Could not load tracking data: {str(e)}")
    
    try:
        # Initialize Drive operations
        drive_ops = GoogleDriveServiceOperations(service_account_file)
        
        # Find the processed videos folder
        # First, find user folder
        user_folder_id = find_folder_by_name(drive_ops.service, User_ID, target_folder_id)
        if not user_folder_id:
            raise ValueError(f"Could not find user folder for {User_ID}")
        
        # Then, find chat folder
        chat_folder_id = find_folder_by_name(drive_ops.service, Chat_ID, user_folder_id)
        if not chat_folder_id:
            raise ValueError(f"Could not find chat folder for {Chat_ID}")
        
        # Finally, find Processed_Videos folder
        processed_folder_id = find_folder_by_name(drive_ops.service, "Processed_Videos", chat_folder_id)
        if not processed_folder_id:
            raise ValueError(f"Could not find Processed_Videos folder")
        
        print(f"Found Processed_Videos folder: {processed_folder_id}")
        
        # Create a temporary directory for downloaded segments
        temp_dir = tempfile.mkdtemp(prefix="audio_processing_")
        segments_dir = os.path.join(temp_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)
        
        # Get all video files in the processed folder
        query = "mimeType contains 'video/' and trashed=false"
        files = drive_ops.list_files(
            query=f"'{processed_folder_id}' in parents and {query}"
        )
        
        if files['status'] != 'success' or not files['data'].get('files'):
            print(f"No video segments found in processed folder")
            return {
                "status": "error",
                "message": "No video segments found in processed folder"
            }
        
        # Process each segment one at a time (download, process, upload, clean up)
        total_segments = len(files['data'].get('files', []))
        print(f"Found {total_segments} video segments to process")
        
        # Initialize results tracking
        segments_processed = 0
        segments_with_poor_audio = 0
        subsegments_created = 0
        subsegments_uploaded = 0
        all_errors = []
        
        # Create a log for overall results
        overall_results = {
            "start_time": datetime.now().isoformat(),
            "segments_processed": [],
            "segments_split": [],
            "errors": []
        }
        
        # Process each video one at a time
        for file_idx, file in enumerate(files['data'].get('files', [])):
            segment_dir = None
            try:
                video_id = file['id']
                video_name = file['name']

                # Skip if already processed
                if resume and video_id in processed_segments:
                    print(f"Skipping already processed segment: {video_name}")
                    continue
                
                print(f"\nProcessing segment {file_idx+1}/{total_segments}: {video_name}")
                
                # Create a temporary directory for this segment
                segment_dir = os.path.join(temp_dir, f"segment_{file_idx}")
                os.makedirs(segment_dir, exist_ok=True)
                
                # Create a path for the downloaded segment
                segment_path = os.path.join(segment_dir, video_name)
                
                # 1. Download the segment
                print(f"Downloading segment: {video_name}")
                download_result = drive_ops.download_file(video_id, segment_path)
                
                if download_result['status'] != 'success':
                    error_msg = f"Failed to download segment {video_name}: {download_result.get('message')}"
                    print(error_msg)
                    all_errors.append({"segment": video_name, "error": error_msg})
                    if segment_dir and os.path.exists(segment_dir):
                        shutil.rmtree(segment_dir)
                    continue
                
                # 2. Check video duration
                cap = cv2.VideoCapture(segment_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                segment_result = {
                    "segment_name": video_name,
                    "duration": duration,
                    "above_min_length": duration >= min_audio_segment_length,
                    "processed": False,
                    "split": False
                }
                
                # Skip segments shorter than min_length
                if duration < min_audio_segment_length:
                    print(f"Skipping segment - duration {duration:.2f}s below threshold of {min_audio_segment_length}s")
                    overall_results["segments_processed"].append(segment_result)
                    segments_processed += 1
                    
                    # Clean up this segment's directory
                    if segment_dir and os.path.exists(segment_dir):
                        shutil.rmtree(segment_dir)
                    continue
                
                # # 3. Analyze audio quality
                # print(f"Analyzing audio quality...")
                # audio_quality = analyze_audio_quality(segment_path, silent_threshold=silence_threshold)
                # segment_result.update({
                #     "audio_analysis": audio_quality,
                #     "processed": True
                # })
                
                # # 4. Check if audio quality is poor and needs splitting
                # is_poor_quality = audio_quality.get('is_poor_quality', False)

                # If force_split is enabled, skip audio quality analysis
                if force_split:
                    print(f"Force split enabled - skipping audio quality analysis")
                    segment_result.update({
                        "processed": True,
                        "force_split": True
                    })
                    is_poor_quality = False  # Not needed but set for consistency
                    audio_quality = {"is_poor_quality": False}  # Default value
                else:
                    # 3. Analyze audio quality
                    print(f"Analyzing audio quality...")
                    audio_quality = analyze_audio_quality(segment_path, silent_threshold=silence_threshold)
                    segment_result.update({
                        "audio_analysis": audio_quality,
                        "processed": True
                    })
                    # 4. Check if audio quality is poor
                    is_poor_quality = audio_quality.get('is_poor_quality', False)
                
                # if is_poor_quality:
                #     print(f"Poor audio quality detected - splitting into subsegments")
                #     segments_with_poor_audio += 1
                
                if is_poor_quality or force_split:
                    if force_split:
                        print(f"Force split enabled - splitting video into subsegments")
                    else:
                        print(f"Poor audio quality detected - splitting into subsegments")
                    segments_with_poor_audio += 1
                    
                    # Create a directory for the subsegments
                    subsegments_dir = os.path.join(
                        segment_dir,
                        f"{os.path.splitext(video_name)[0]}_subsegments"
                    )
                    
                    # Split the video
                    subsegment_paths = split_video_into_subsegments(
                        segment_path, 
                        subsegments_dir, 
                        max_duration=max_subsegment_duration
                    )
                    
                    segment_result.update({
                        "split": True,
                        "num_subsegments": len(subsegment_paths),
                        "subsegments_dir": subsegments_dir
                    })
                    
                    overall_results["segments_split"].append(segment_result)
                    
                    # 5. Upload subsegments
                    if subsegment_paths:
                        print(f"Uploading {len(subsegment_paths)} subsegments to Google Drive")
                        
                        for sub_idx, subsegment_path in enumerate(subsegment_paths):
                            try:
                                subsegment_name = os.path.basename(subsegment_path)
                                
                                # Upload to processed videos folder
                                # upload_result = drive_ops.upload_file(
                                #     subsegment_path,
                                #     parent_folder_id=processed_folder_id,
                                #     file_name=subsegment_name
                                # )

                                # Clean the subsegment name (similar to how we do in segmentation)
                                clean_name = os.path.splitext(subsegment_name)[0].replace('.mp4', '').replace('.MP4', '')
                                clean_name = ''.join(c for c in clean_name if c.isalnum() or c in ' _-')
                                clean_name = f"{clean_name}.mp4"  # Re-add extension

                                upload_result = drive_ops.upload_file(
                                    subsegment_path,
                                    parent_folder_id=processed_folder_id,
                                    file_name=clean_name
                                )
                                
                                if upload_result['status'] == 'success':
                                    subsegments_uploaded += 1
                                    print(f"Uploaded subsegment {sub_idx+1}/{len(subsegment_paths)}: {subsegment_name}")
                                else:
                                    error_msg = f"Failed to upload subsegment {subsegment_name}: {upload_result.get('message')}"
                                    print(error_msg)
                                    all_errors.append({"segment": subsegment_name, "error": error_msg})
                            except Exception as e:
                                error_msg = f"Error uploading subsegment {os.path.basename(subsegment_path)}: {str(e)}"
                                print(error_msg)
                                all_errors.append({"segment": os.path.basename(subsegment_path), "error": error_msg})
                                
                        subsegments_created += len(subsegment_paths)
                
                # Add to overall results
                overall_results["segments_processed"].append(segment_result)
                segments_processed += 1
                
                # 6. Move the original segment to trash if it was split into subsegments
                if (is_poor_quality or force_split) and subsegments_uploaded > 0:
                    try:
                        print(f"Moving original segment to trash: {video_name}")
                        trash_result = drive_ops.service.files().update(
                            fileId=video_id,
                            body={"trashed": True},
                            supportsAllDrives=True
                        ).execute()
                        print(f"Original segment moved to trash")
                    except Exception as e:
                        print(f"Warning: Could not move segment to trash: {str(e)}")
                
                # 7. Clean up this segment's directory
                if segment_dir and os.path.exists(segment_dir):
                    shutil.rmtree(segment_dir)
                print(f"Temporary files cleaned up for: {video_name}")

                if resume:
                    processed_segments.add(video_id)
                    with open(tracking_file, 'w') as f:
                        json.dump(list(processed_segments), f)
                
            except Exception as e:
                error_msg = f"Error processing segment {file.get('name', 'unknown')}: {str(e)}"
                print(error_msg)
                all_errors.append({"segment": file.get('name', 'unknown'), "error": error_msg})
                
                # Make sure we clean up the directory even if there's an error
                if segment_dir and os.path.exists(segment_dir):
                    try:
                        shutil.rmtree(segment_dir)
                    except Exception as cleanup_error:
                        print(f"Failed to clean up segment directory: {str(cleanup_error)}")
        
        # Update the overall results
        overall_results["end_time"] = datetime.now().isoformat()
        overall_results["total_segments"] = segments_processed
        overall_results["segments_with_poor_audio"] = segments_with_poor_audio
        overall_results["subsegments_created"] = subsegments_created
        overall_results["subsegments_uploaded"] = subsegments_uploaded
        overall_results["errors"] = all_errors
        
        # Save overall results to a log file
        log_path = os.path.join(temp_dir, f"audio_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        # Upload the overall log
        try:
            upload_result = drive_ops.upload_file(
                log_path,
                parent_folder_id=processed_folder_id,
                file_name=os.path.basename(log_path)
            )
            print(f"Uploaded audio processing log: {os.path.basename(log_path)}")
        except Exception as e:
            print(f"Error uploading audio processing log: {str(e)}")
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print("Cleaned up all temporary directories")
        except Exception as e:
            print(f"Failed to clean up temporary directories: {str(e)}")
            
        # Return combined results
        return {
            "status": "success",
            "audio_processing": {
                "segments_processed": segments_processed,
                "segments_with_poor_audio": segments_with_poor_audio,
                "subsegments_created": subsegments_created,
                "subsegments_uploaded": subsegments_uploaded
            },
            "processed_folder_link": f"https://drive.google.com/drive/folders/{processed_folder_id}"
        }
        
    except Exception as e:
        error_msg = f"Error in audio quality processing: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }
        

# Helper function to find a folder by name
def find_folder_by_name(service, folder_name, parent_id):
    """Find folder by exact name and parent"""
    try:
        # Escape special characters in folder name for the query
        escaped_name = folder_name.replace("'", "\\'")
        
        query = f"name='{escaped_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
        
        # Execute the query
        results = service.files().list(
            q=query,
            fields='files(id, name)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()

        # Handle existing folder
        if results.get('files'):
            # If multiple folders exist with same name under same parent, use the first one
            if len(results['files']) > 1:
                print(f"Multiple folders named '{folder_name}' found under same parent. Using first one.")
            
            folder_id = results['files'][0]['id']
            return folder_id
            
        return None
        
    except Exception as e:
        print(f"Error in find_folder_by_name for '{folder_name}': {str(e)}")
        return None

# Function that integrates with the existing process_videos_with_segmentation function
def process_videos_with_audio_quality_check(
    service_account_file: str = 'service-account.json',
    target_folder_id: str = '1be5p41JtvBbSxKpBaxrcotet0RZCzt5Y',
    motion_threshold: int = 5,
    min_segment_duration: float = 1.0,
    motion_leniency: int = 2,
    smoothing_window: int = 3,
    min_audio_segment_length: float = 5.0,
    silence_threshold: float = 0.01,
    poor_volume_threshold: float = -30,
    silence_percentage_threshold: float = 60,
    max_subsegment_duration: float = 4.9,
    resume: bool = True
):
    """
    Processes videos with segmentation based on motion and then checks audio quality,
    splitting segments with poor audio into smaller subsegments.
    
    Args:
        service_account_file: Path to service account credentials file
        target_folder_id: ID of the target folder (shared drive folder)
        motion_threshold: Threshold for motion detection
        min_segment_duration: Minimum duration in seconds for a segment to be kept
        motion_leniency: Number of consecutive high-motion frames to tolerate
        smoothing_window: Size of rolling window for motion smoothing
        min_audio_segment_length: Minimum segment length to consider for audio quality check
        silence_threshold: RMS threshold for silence detection
        poor_volume_threshold: Mean volume threshold in dB below which audio is considered poor
        silence_percentage_threshold: Percentage of silence above which audio is considered poor
        max_subsegment_duration: Maximum duration of subsegments in seconds
        resume: Whether to resume from previous run if interrupted
        
    Returns:
        Dictionary with processing results
    """
    # First process videos with segmentation
    segmentation_result = process_videos_with_segmentation(
        service_account_file=service_account_file,
        target_folder_id=target_folder_id,
        motion_threshold=motion_threshold,
        min_segment_duration=min_segment_duration,
        motion_leniency=motion_leniency,
        smoothing_window=smoothing_window,
        resume=resume
    )
    
    if segmentation_result.get('status') != 'success':
        print("Video segmentation failed, cannot proceed with audio quality check")
        return segmentation_result
    
    print("Video segmentation completed successfully. Now checking audio quality of segments.")
    
    # Get the processed videos folder from the segmentation result
    processed_folder_id = segmentation_result.get('folder_structure', {}).get('processed_folder_id')
    
    if not processed_folder_id:
        print("Could not find processed videos folder ID in segmentation result")
        return {
            "status": "error",
            "message": "Could not find processed videos folder ID",
            "segmentation_result": segmentation_result
        }
    
    # Create a temporary directory for downloaded segments
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="audio_processing_")
    segments_dir = os.path.join(temp_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    try:
        # Initialize Drive operations
        drive_ops = GoogleDriveServiceOperations(service_account_file)
        
        # Get all video files in the processed folder
        query = "mimeType contains 'video/' and trashed=false"
        files = drive_ops.list_files(
            query=f"'{processed_folder_id}' in parents and {query}"
        )
        
        if files['status'] != 'success' or not files['data'].get('files'):
            print(f"No video segments found in processed folder")
            return {
                "status": "error",
                "message": "No video segments found in processed folder",
                "segmentation_result": segmentation_result
            }
        
        # Download all segments for audio processing
        downloaded_segments = []
        for file in files['data'].get('files', []):
            try:
                video_id = file['id']
                video_name = file['name']
                
                # Create a path for the downloaded segment
                segment_path = os.path.join(segments_dir, video_name)
                
                # Download the segment
                download_result = drive_ops.download_file(video_id, segment_path)
                
                if download_result['status'] == 'success':
                    downloaded_segments.append({
                        "segment_name": video_name,
                        "segment_id": video_id,
                        "local_path": segment_path
                    })
                    print(f"Downloaded segment: {video_name}")
                else:
                    print(f"Failed to download segment {video_name}: {download_result.get('message')}")
            except Exception as e:
                print(f"Error downloading segment {file.get('name', 'unknown')}: {str(e)}")
        
        print(f"Downloaded {len(downloaded_segments)} segments for audio quality checking")
        
        # Process the downloaded segments for audio quality
        audio_processing_result = process_segments_by_audio_quality(
            segments_dir,
            min_length=min_audio_segment_length,
            silence_threshold=silence_threshold,
            poor_volume_threshold=poor_volume_threshold,
            silence_percentage_threshold=silence_percentage_threshold,
            max_subsegment_duration=max_subsegment_duration
        )
        
        # Upload the subsegments back to the processed folder
        uploaded_subsegments = []
        
        # Find all newly created subsegments
        subsegments = []
        for root, dirs, files in os.walk(segments_dir):
            if "subsegments" in root:  # Only look in subsegment directories
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')):
                        subsegments.append(os.path.join(root, file))
        
        print(f"Found {len(subsegments)} subsegments to upload")
        
        # Upload each subsegment
        for subsegment_path in subsegments:
            try:
                subsegment_name = os.path.basename(subsegment_path)
                
                # Upload to processed videos folder
                upload_result = drive_ops.upload_file(
                    subsegment_path,
                    parent_folder_id=processed_folder_id,
                    file_name=subsegment_name
                )
                
                if upload_result['status'] == 'success':
                    uploaded_subsegments.append({
                        "subsegment_name": subsegment_name,
                        "drive_id": upload_result['data']['id'],
                        "drive_link": f"https://drive.google.com/file/d/{upload_result['data']['id']}/view"
                    })
                    print(f"Uploaded subsegment: {subsegment_name}")
                else:
                    print(f"Failed to upload subsegment {subsegment_name}: {upload_result.get('message')}")
            except Exception as e:
                print(f"Error uploading subsegment {os.path.basename(subsegment_path)}: {str(e)}")
        
        # Upload the audio processing log
        try:
            log_files = [f for f in os.listdir(segments_dir) if f.startswith("audio_processing_log_") and f.endswith(".json")]
            if log_files:
                log_path = os.path.join(segments_dir, log_files[0])
                upload_result = drive_ops.upload_file(
                    log_path,
                    parent_folder_id=processed_folder_id,
                    file_name=os.path.basename(log_path)
                )
                print(f"Uploaded audio processing log: {os.path.basename(log_path)}")
        except Exception as e:
            print(f"Error uploading audio processing log: {str(e)}")
        
        # Clean up temporary directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary directory")
        except Exception as e:
            print(f"Failed to clean up temporary directory: {str(e)}")
        
        # Return combined results
        return {
            "status": "success",
            "segmentation_result": segmentation_result,
            "audio_processing": {
                "segments_processed": len(audio_processing_result.get("segments_processed", [])),
                "segments_with_poor_audio": len(audio_processing_result.get("segments_split", [])),
                "subsegments_created": len(subsegments),
                "subsegments_uploaded": len(uploaded_subsegments)
            },
            "processed_folder_link": segmentation_result.get('folder_structure', {}).get('processed_folder_link')
        }
        
    except Exception as e:
        print(f"Error in audio quality processing: {str(e)}")
        return {
            "status": "error",
            "message": f"Error in audio quality processing: {str(e)}",
            "segmentation_result": segmentation_result
        }
    
def clean_repaired_videos(directory=None):
    """
    Find and delete all temporary repaired video files.
    
    Args:
        directory: Directory to search in. If None, will search in the current directory.
    
    Returns:
        int: Number of files deleted
    """
    import os
    
    if directory is None:
        directory = os.getcwd()
    
    count = 0
    deleted_files = []
    
    # Walk through the directory and all subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.startswith("repaired_") and any(filename.endswith(ext) for ext in 
                                                    ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']):
                file_path = os.path.join(root, filename)
                try:
                    # Check if file is not in use
                    os.stat(file_path)
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    count += 1
                except Exception as e:
                    print(f"Could not delete {file_path}: {str(e)}")
    
    print(f"Deleted {count} repaired video files")
    for file in deleted_files:
        print(f"  - {file}")
    
    return count


def downscale_video_for_processing(original_video_path, target_width=960, min_width_to_downscale=1280):
    """
    Creates a downscaled version of a video for faster processing, but only if
    the original video is large enough to benefit from downscaling.
    
    Parameters:
    - original_video_path: Path to the original high-resolution video
    - target_width: Target width for the longer side (width or height)
    - min_width_to_downscale: Minimum width/height threshold to perform downscaling
    
    Returns:
    - Path to the downscaled video or original video if small enough
    """
    import os
    import subprocess
    import cv2
    import time
    
    # Check video dimensions first
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {original_video_path}")
        return original_video_path
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Determine if the video is large enough to benefit from downscaling
    max_dimension = max(width, height)
    if max_dimension <= min_width_to_downscale:
        print(f"Video {os.path.basename(original_video_path)} is already small enough ({width}x{height}), skipping downscaling")
        return original_video_path
    
    # Create output filename for downscaled version
    base_dir = os.path.dirname(original_video_path)
    base_name = os.path.basename(original_video_path)
    output_path = os.path.join(base_dir, f"downscaled_{base_name}")
    
    # Skip if already exists
    if os.path.exists(output_path):
        print(f"Using existing downscaled video: {output_path}")
        return output_path
    
    # Make sure target dimension is an even number (required by most codecs)
    if target_width % 2 != 0:
        target_width = target_width - 1
    
    # Determine if portrait or landscape and set scale filter accordingly
    if height > width:  # Portrait
        # For portrait videos, ensure output dimensions are even numbers
        scale_filter = f"scale='trunc(oh*a/2)*2:{target_width}'"
        print(f"Detected portrait video ({width}x{height}), scaling to height {target_width}")
    else:  # Landscape
        # For landscape videos, ensure output dimensions are even numbers
        scale_filter = f"scale='{target_width}:trunc(ow/a/2)*2'"
        print(f"Detected landscape video ({width}x{height}), scaling to width {target_width}")
    
    # Run ffmpeg to create downscaled version
    cmd = [
        'ffmpeg',
        '-i', original_video_path,
        '-vf', scale_filter,
        '-c:v', 'libx264',
        '-crf', '28',  # Use higher CRF (lower quality) for analysis version
        '-preset', 'veryfast',  # Use fast encoding preset
        '-an',  # No audio needed for motion analysis
        '-y',  # Overwrite if exists
        output_path
    ]
    
    print(f"Creating downscaled version of {base_name} for analysis...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Add a small delay to ensure file is fully written
        time.sleep(0.5)
        print(f"Downscaled video created at: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error creating downscaled video: {e}")
        print(f"Error output: {e.stderr.decode()}")
        print(f"Falling back to original video for analysis")
        return original_video_path  # Fall back to original if downscaling fails