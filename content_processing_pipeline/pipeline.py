import json
import os
import config
from decorators import *
from helper_functions import HelperFunctions
from GPT_assistant import GPT_assistant  
from concurrent.futures import ThreadPoolExecutor, as_completed

from image_processor import (cluster_story_pipeline, upload_images_to_drive, 
                           create_insta_images, delete_folder, create_insta_video)
from lut import apply_lut_with_mapping

import instagram_filter

# config.User_ID = config.config.User_ID
# config.Chat_ID = config.config.Chat_ID
# config.Reel_ID = config.config.Reel_ID
# user_prompt = config.user_prompt
# songname = config.songname


RATING_TO_EMBEDDING = {
    'all': 'embeddings_all.pkl',
    'average': 'embeddings_average.pkl', 
    'good': 'embeddings_good.pkl',
    'very_good': 'embeddings_very_good.pkl',
    'best': 'embeddings_best.pkl'
}

RATING_TO_CLUSTERS = {
    'all': 'all_reel_clusters.json',
    'average': 'average_reel_clusters.json',
    'good': 'good_reel_clusters.json',
    'very_good': 'very_good_reel_clusters.json',
    'best': 'best_reel_clusters.json'
}


@log_execution
@handle_errors
def initialize_template():
    # global config.Reel_ID
    
    try:
        # Load song embeddings data
        spotify_base = Path("Assets/spotify/embeddings") 
        with open(spotify_base / config.embeding_path, 'rb') as f:
            data = pickle.load(f)
        
        model = data['model']
        index = data['index']
        song_names = data['song_names']
        
        # Generate embedding for user prompt
        query_embedding = model.encode([config.user_prompt])
        
        # Search for best matching song
        distances, indices = index.search(query_embedding.astype('float32'), k=5)  # Get top 5 matches
        
        # Try songs until we find one with valid files
        song_name = None
        audio_file = None
        
        for idx, dist in zip(indices[0], distances[0]):
            potential_song = song_names[idx]
            potential_audio = find_spotify_file(potential_song, 'audio')
            if potential_audio:
                song_name = potential_song
                audio_file = potential_audio
                match_score = 1 / (1 + dist)
                print(f"Selected song: {song_name} (score: {match_score:.2f})")
                break
                
        if not audio_file:
            raise ValueError("No suitable song found with available audio file")
            
        sanitized_song = song_name.lower().replace(' ', '_')
        
        # Load spotify data for analysis and beats
        spotify_data = load_spotify_data()
        
        # Get analysis data
        analysis = spotify_data['analysis'].get(sanitized_song, {})
        audio_start_time = analysis.get('start_time', 0.0)
        section_times = analysis.get('sections', [])

        # Get beats data
        beats = spotify_data['beats'].get(sanitized_song, [])
        
        # If no beats from spotify_data, try finding the beats file
        if not beats:
            beats_file = find_spotify_file(song_name, 'beats')
            if beats_file:
                with open(beats_file, 'r') as f:
                    beats_data = json.load(f)
                    beats = [bar['time'] for bar in beats_data]

        # Process beats - filter and adjust by start time
        adjusted_beats = [
            round(beat - audio_start_time, 2)
            for beat in beats
            if beat >= audio_start_time
        ]
        
        # Limit beats to reasonable number
        max_beats = 20  # You can adjust this based on your needs
        beats = adjusted_beats[:max_beats]
        
        # If no section times, try finding analysis file
        if not section_times:
            analysis_file = find_spotify_file(song_name, 'analysis')
            if analysis_file:
                with open(analysis_file, 'r') as f:
                    full_analysis = json.load(f)
                    sections = full_analysis.get('sections', [])
                    section_times = [
                        round(section['start'] - audio_start_time, 2)
                        for section in sections 
                        if section['start'] >= audio_start_time
                    ]
        
        # Find lyrics file
        lyrics_file = find_spotify_file(song_name, 'lyrics')
        
        # Create template structure
        template = {
            "template": {
                "user_prompt": config.user_prompt,
                "prompt_id": "user",
                "song_beats": beats,
                "section_times": section_times,
                "audio_start_time": audio_start_time,
                "updated_prompt": config.user_prompt,
                "aspect_ratio": [16, 9],
                "resolution": 1080,
                "fps": "30",
                "audio": str(audio_file) if audio_file else "",
                "srt": str(lyrics_file) if lyrics_file else ""
            }
        }

        # Generate unique config.Reel_ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_words = config.user_prompt.split()
        prompt_prefix = '_'.join(prompt_words[:3])[:15]
        # Remove special characters and replace spaces with underscores
        prompt_prefix = re.sub(r'[^a-zA-Z0-9_]', '', prompt_prefix)
        # Add a short random suffix to ensure uniqueness
        import random
        import string
        # random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        config.Reel_ID = f"Reel_user_{prompt_prefix}"
        # config.Reel_ID = f"Reel_user_{timestamp}"
        
        # Create directory and save template
        reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
        os.makedirs(reel_path, exist_ok=True)
        
        template_path = os.path.join(reel_path, 'video_template.json')
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=4)
            
        print(f"Created template for user request with song {song_name}")
        print(f"Found {len(beats)} beats and {len(section_times)} sections")
        
        return True
        
    except Exception as e:
        print(f"Error creating template: {str(e)}")
        raise

# Example usage:
# initialize_template()
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import matplotlib.pyplot as plt
from PIL import Image
import torch


@log_execution
@handle_errors
def get_embedding_search_results():
    """
    Searches for the most relevant media items based on the given query using a two-stage approach:
    1. Initial embedding search to select top 100 candidates
    2. Cross-encoder refinement to re-rank the top candidates
    """
    base_path = os.path.join(config.User_ID, config.Chat_ID)

    # Define mapping for different process filters to their respective pkl files
    RATING_TO_PKL = {
        'all': 'embeddings_all.pkl',
        'average': 'embeddings_average.pkl',
        'good': 'embeddings_good.pkl',
        'very_good': 'embeddings_very_good.pkl',
        'best': 'embeddings_best.pkl'
    }
    
    # Get the correct pkl file based on process filter
    pkl_file = RATING_TO_PKL.get(config.rating_filter or 'all', 'embeddings_all.pkl')
    preprocessed_data_path = os.path.join(base_path, pkl_file)
    
    images_folder = os.path.join(base_path, 'Media', 'Images')

    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    with open(os.path.join(reel_path, 'video_template.json'), 'r') as f:
        video_template = json.load(f)
    
    # print(video_template)
    song_beats_length = len(video_template["template"]["song_beats"])

    query = video_template["template"]["updated_prompt"]
    
    # Load preprocessed data
    with open(preprocessed_data_path, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    model = preprocessed_data['model']
    index = preprocessed_data['index']
    file_paths = preprocessed_data['file_paths']
    embedding_data = preprocessed_data['embedding_data']
    
    # Generate query embedding
    query_embedding = model.encode(query)
    
    # Perform initial search to get top 100 candidates
    initial_top_k = 100
    top_k = song_beats_length + 10
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), initial_top_k)
    
    # Prepare candidates for cross-encoder
    candidates = []
    for i, idx in enumerate(indices[0]):
        file_path = file_paths[idx]
        candidates.append({
            'file_path': file_path,
            'initial_similarity': 1 - distances[0][i],
            'description': embedding_data[file_path]['description'],
            'faces': embedding_data[file_path]['faces']
        })
    
    import torch
    
    device = 'cpu'  # Force CPU usage to avoid MPS device issues
    # Initialize cross-encoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
    
    # Prepare input pairs for cross-encoder
    input_pairs = [[query, f"This image contains the following people: {' '.join(candidate['faces'])} and Description: {candidate['description']}"] for candidate in candidates]
    
    # Compute cross-encoder scores
    cross_encoder_scores = cross_encoder.predict(input_pairs)
    
    # Combine candidates with cross-encoder scores and sort
    results = sorted(zip(candidates, cross_encoder_scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Prepare final results
    search_results = []
    for candidate, score in results:
        search_results.append({
            'file_path': candidate['file_path'],
            'similarity': float(score),  # Use cross-encoder score as final similarity
            'description': candidate['description'],
            'faces': candidate['faces']
        })
    
    # Save the template as JSON in the new location
    results_path = os.path.join(reel_path, 'search_results.json')
    with open(results_path, 'w') as f:
        json.dump(search_results, f, indent=4)

    print(f"Saved {len(search_results)} search results to {results_path}")
    



# # Example usage:
# generate_prompt()

# config.User_ID = "1_Shash"
# config.Chat_ID = "8_Tester_2000_500"
# config.Reel_ID = "8_Monika_happy"

from tqdm import tqdm
import io
import os
import json
from GPT_assistant import GPT_assistant
from PIL import Image, ExifTags
import cv2
import contextlib

@log_execution
@handle_errors
def generate_two_prompts(preserve_video_length=True):
    def find_file(directory, filename):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith(filename):
                    return os.path.join(root, file)
        return None

    def get_aspect_ratio_category(width, height):
        ratio = width / height
        # print(f"Width: {width}, Height: {height}, Aspect ratio: {ratio:.2f}")  # Debug print
        if ratio > 1.1:
            return "landscape"
        elif ratio < 0.91:  # Changed from 0.9 to 0.91
            return "portrait"
        else:
            return "square"

    def get_image_info(file_path):
        # with Image.open(file_path) as img:
        img = HelperFunctions.load_image(file_path)
        if img is not None:
            img = Image.fromarray(img)            
            # Handle EXIF orientation
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break

                exif = img._getexif()
                if exif is not None:
                    exif = dict(exif.items())
                    orientation_value = exif.get(orientation)

                    if orientation_value == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError, TypeError):
                # No EXIF data or orientation tag
                pass

            width, height = img.size
            # print(f"Width: {width}, Height: {height}")  # Debug print
            return get_aspect_ratio_category(width, height)

    def get_video_info(file_path):
        # video = cv2.VideoCapture(file_path)
        video = HelperFunctions.load_video(file_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        video.release()
        return get_aspect_ratio_category(width, height), duration

    def simplify_file_name(file_path):
        base_name = os.path.basename(file_path)
        name_parts = base_name.split('-')
        simplified_name = name_parts[0]
        extension = os.path.splitext(base_name)[1]
        return f"{simplified_name}{extension}"
    
    def process_descriptions_parallel(items: list, max_workers: int = 10) -> dict:
        """
        Process descriptions in parallel using ThreadPoolExecutor
        
        Args:
            items: List of items containing descriptions
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dict mapping file paths to shortened descriptions
        """
        def shorten_single_description(args: tuple) -> tuple:
            file_path, description = args
            assistant_id = 'asst_nhcUwe2bOgc1iEx8aGf7WQIx'
            try:
                gpt_assistant = GPT_assistant(assistant_id)
                thread = gpt_assistant.create_thread()
                gpt_assistant.add_message_to_thread(thread, description)
                run = gpt_assistant.run_thread_on_assistant(thread)
                response = gpt_assistant.check_run_status_and_respond(thread, run)
                
                # # Redirect stdout to suppress printing
                # with open(os.devnull, 'w') as fnull:
                #     with contextlib.redirect_stdout(fnull):
                #         response = gpt_assistant.check_run_status_and_respond(thread, run)
                
                return file_path, response.strip()
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return file_path, description  # Return original description on error

        # Create list of tasks
        tasks = [(item['file_path'], item['description']) 
                for item in items if 'description' in item]

        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create future tasks
            future_to_path = {
                executor.submit(shorten_single_description, task): task[0]
                for task in tasks
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_path), 
                            total=len(tasks), 
                            desc="Shortening descriptions"):
                file_path, shortened = future.result()
                results[file_path] = shortened

        return results

    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    
    # Load the video template
    with open(os.path.join(reel_path, 'video_template.json'), 'r') as f:
        video_template = json.load(f)
    
    # Load the search results
    with open(os.path.join(reel_path, 'search_results.json'), 'r') as f:
        search_results = json.load(f)
    
    # Extract relevant information
    user_prompt = video_template['template']['user_prompt']
    beats = video_template['template']['song_beats']
    
    # Process search results to include aspect ratio and length
    image_dir = os.path.join(config.User_ID, config.Chat_ID, 'Media', 'Images')
    video_dir = os.path.join(config.User_ID, config.Chat_ID, 'Media', 'Videos')
    
    # First process all non-GPT operations
    for item in tqdm(search_results, desc="Processing media info", unit="item"):
        original_file_path = item['file_path']
        simplified_file_name = simplify_file_name(original_file_path)
        item['file_path'] = simplified_file_name
        
        image_path = find_file(image_dir, original_file_path.split('-')[0])
        video_path = find_file(video_dir, original_file_path.split('-')[0])
        
        if image_path:
            item['orientation'] = get_image_info(image_path)
        elif video_path:
            item['orientation'], item['duration'] = get_video_info(video_path)
        else:
            item['orientation'] = "unknown"

        # Remove unnecessary fields
        item.pop('faces', None)
        item.pop('similarity', None)

    # Then process descriptions in parallel
    if any('description' in item for item in search_results):
        shortened_descriptions = process_descriptions_parallel(search_results)
        
        # Update the descriptions in search_results
        for item in search_results:
            if 'description' in item:
                item['description'] = shortened_descriptions[item['file_path']]

    # Create the first prompt for story creation
    story_prompt = {
        "instructions": """
You are an AI assistant tasked with creating a visual story by selecting and ordering images and videos for a sequence. You will be provided with a set of image and video descriptions.

Input:
- A user query providing context for the desired visual story
- A set of image and video descriptions, each containing:
  * file_path: Unique identifier for the image or video
  * description: Detailed analysis of the content
  * aspect_ratio: 'portrait', 'landscape', or 'square'
  * duration: The length of the video (for videos only)

Task:
1. Analyze the provided image and video descriptions in relation to the user query.
2. Create a cohesive story based on the available visual content and the given theme.
3. Select and order the images and videos to best represent this story, ensuring:
   - A diverse selection of visuals that capture key moments
   - Important images or videos are placed at significant points in the sequence
   - The overall flow maintains viewer engagement

Output:
Provide a JSON array where each object contains:
- file_path: The unique identifier of the selected image or video
Only return JSON, no need for any explanation or steps.

Example output format:
[
  {
    "file_path": "1da78e1d.jpg"
  },
  {
    "file_path": "33b13788.mp4"
  },
  ...
]

Important notes:
- Aim to include most or all of the available media in your sequence to provide a comprehensive story.
- Only exclude images or videos that are significantly off-topic or of poor quality.
- Focus on creating a compelling narrative through your selection and ordering of visuals.
- Prioritize images and videos with higher overall ratings and relevance to ensure better quality content.
- Ensure the sequence tells a coherent story related to the user query.


Please provide your visual story sequence based on the given information and considerations.
        """,
        "user_prompt": user_prompt,
        "images": search_results
    }

    # Create the second prompt for beat alignment
    beat_alignment_prompt = {
        "instructions": """
You are an AI assistant tasked with aligning a pre-selected sequence of images and videos with the beats of an audio track. You will be provided with a list of audio beat timings.

Input:
- A list of audio beat timings in seconds

Task:
1. Align the pre-selected visual elements with the provided audio beats.
2. Assign end times to each visual element, ensuring they align with or are close to the provided beat timings.

Output:
Provide a JSON array where each object contains:
- file_path: The unique identifier of the selected image or video
- end_time: The time in seconds when this visual element should end
Only return JSON, no need for any explanation or steps.

Example output format:
[
  {
    "file_path": "1da78e1d.jpg",
    "end_time": 2.0,
  },
  {
    "file_path": "33b13788.mp4",
    "end_time": 5.6,

  },
  ...
]

Important notes:
- The first visual should start at 0 seconds, and each subsequent visual should start immediately after the previous one ends.
- There should be no element such that the end_time of file is more than the last beat.
- Use the audio beat timings as a guide for pacing and transitions in your visual story.
- Videos can occupy multiple betas and tend to enhance visual appeal; use them for critical and longer moments in the story.
- The end_time for each visual should align with or be close to one of the provided beat timings.
- Consider the duration of videos when assigning end times.
- Ensure all images stay for mostly the same duration.

Please provide your beat-aligned visual sequence based on the given information and considerations.
        """,
        "beats": beats
    }

    # For preserve_video_length=True condition
    if preserve_video_length:
        beat_alignment_prompt = {
            "instructions": """
        You are an AI assistant tasked with creating a sequence of images and videos with appropriate timing for a visual story. You will be provided with a list of audio beat timings as reference points.

        Input:
        - A list of audio beat timings in seconds (for reference only)

        Task:
        1. Arrange the pre-selected visual elements in a logical sequence.
        2. Assign end times to each visual element, ensuring videos play in their entirety.

        Output:
        Provide a JSON array where each object contains:
        - file_path: The unique identifier of the selected image or video
        - end_time: The time in seconds when this visual element should end
        Only return JSON, no need for any explanation or steps.

        Example output format:
        [
        {
            "file_path": "1da78e1d.jpg",
            "end_time": 2.0,
        },
        {
            "file_path": "33b13788.mp4",
            "end_time": 12.5,
        },
        ...
        ]

        Important notes:
        - The first visual should start at 0 seconds, and each subsequent visual should start immediately after the previous one ends.
        - CRITICAL: For video files, assign them enough time to play completely.
        - IMPORTANT: Position longer videos (over 5 seconds) either at the beginning or the end of the sequence for better viewer experience.
        - Images should typically stay visible for 1-2 seconds.
        - There should be no element such that the end_time exceeds the last beat time in the provided list.
        - The absolute priority is to keep videos intact at their full length and using longer video at only the start or end.
        - Use the beat timings only as loose reference points for pacing.

        Please provide your sequence that prioritizes full video playback while maintaining a coherent visual story with longer videos at the beginning or end.
        """,
            "beats": beats
        }
    
    # Save the prompts to JSON files
    story_prompt_path = os.path.join(reel_path, 'gpt_story_prompt.json')
    with open(story_prompt_path, 'w') as f:
        json.dump(story_prompt, f, indent=2)
    
    beat_alignment_prompt_path = os.path.join(reel_path, 'gpt_beat_alignment_prompt.json')
    with open(beat_alignment_prompt_path, 'w') as f:
        json.dump(beat_alignment_prompt, f, indent=2)
    
    print(f"Story prompt saved to {story_prompt_path}")
    print(f"Beat alignment prompt saved to {beat_alignment_prompt_path}")

# generate_two_prompts()

import json
import os

@log_execution
@handle_errors
def fill_elements_ordered():
    # Load the video template to get the beat times
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    with open(os.path.join(reel_path, 'video_template.json'), 'r') as f:
        video_template = json.load(f)
    
    beat_times = video_template['template']['song_beats']

    # Load the search results
    with open(os.path.join(reel_path, 'search_results.json'), 'r') as f:
        search_results = json.load(f)

    # Create the new format
    new_format = []
    for index, image in enumerate(search_results):
        if index < len(beat_times):
            new_format.append({
                "file_path": image['file_path'],  # Assuming 'id' is the correct key and adding '.jpg' extension
                "end_time": beat_times[index]
            })

    # Save the updated format
    with open(os.path.join(reel_path, 'element_sequence.json'), 'w') as f:
        json.dump(new_format, f, indent=2)

    print(f"Element sequence saved to {os.path.join(reel_path, 'element_sequence.json')}")



import json
import os

@log_execution
@handle_errors
def create_video_elements():
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    
    # Load the search results with slots
    # with open(os.path.join(reel_path, 'search_results_with_slots.json'), 'r') as f:
    #     search_results = json.load(f)
    
    # Load the element sequence
    with open(os.path.join(reel_path, 'element_sequence.json'), 'r') as f:
        element_sequence = json.load(f)
    # print(search_results)
    
    # Load the video template
    template_path = os.path.join(reel_path, 'video_template.json')
    with open(template_path, 'r') as f:
        video_template = json.load(f)
    
    # beats = video_template['template']['song_beats']
    fps = int(video_template['template']['fps'])
    
    elements = []
    
    # def find_file(directory, truncated_filename):
    #     truncated_name = truncated_filename.split('.')[0]  # Remove extension if present
        
    #     for root, dirs, files in os.walk(directory):
    #         for file in files:
    #             if file.startswith(truncated_name):
    #                 return os.path.join(root, file)
    #     return None
    
    def find_file(directory, truncated_filename):
        truncated_name = truncated_filename.split('.')[0]  # Remove extension if present
        # Just do exact match
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith(truncated_name):
                    return os.path.join(root, file)
        return None

    def find_closest_file(directory, truncated_filename):
        from difflib import SequenceMatcher
        truncated_name = truncated_filename.split('.')[0]
        highest_similarity = 0
        closest_file = None
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_name = file.split('.')[0]
                similarity = SequenceMatcher(None, truncated_name, file_name).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    closest_file = os.path.join(root, file)
        
        if closest_file:
            logging.info(f"Using closest matching file {os.path.basename(closest_file)} instead of {truncated_filename}")
        return closest_file

    # Initialize cumulative timeline tracker for NEW format
    cumulative_time = 0.0

    # for item in search_results:
    for index, item in enumerate(element_sequence):
        # Calculate frame_start and frame_end
        if 'offset' in item:
            # NEW FORMAT - offset/duration exists
            offset = item['offset']
            duration = item['duration']
            frame_start = int(cumulative_time * fps)
            frame_end = int((cumulative_time + duration) * fps) - 1
            cumulative_time += duration
        elif 'start_time' in item:
            # LEGACY FORMAT - start_time/end_time (for backward compatibility)
            duration = item['end_time'] - item['start_time']
            offset = item.get('start_time', 0.0)  # Use start_time as offset for legacy
            frame_start = int(cumulative_time * fps)
            frame_end = int((cumulative_time + duration) * fps) - 1
            cumulative_time += duration
        else:
            # OLD FORMAT - end_time is cumulative position
            offset = 0.0
            if index == 0:
                frame_start = 0
            else:
                frame_start = int(element_sequence[index - 1]['end_time'] * fps)
            frame_end = int(item['end_time'] * fps) - 1  # Subtract 1 to avoid overlap
            duration = (frame_end - frame_start + 1) / fps
        # if 'assigned_slot' in item and item['assigned_slot'] is not None:
            # Search for the file without extension
        image_dir = os.path.join(config.User_ID, config.Chat_ID, 'Media', 'Images')
        video_dir = os.path.join(config.User_ID, config.Chat_ID, 'Media', 'Videos')
        
        image_path = find_file(image_dir, item['file_path'])
        video_path = find_file(video_dir, item['file_path'])

        # Add this block before is_video check:
        # Add this block before is_video check:
        if image_path is None and video_path is None:
            image_path = find_closest_file(image_dir, item['file_path'])
            if image_path is not None:
                logging.info(f"File not found: {item['file_path']}, replaced with: {os.path.basename(image_path)}")
            if image_path is None:  # If no close image match
                video_path = find_closest_file(video_dir, item['file_path'])
                if video_path is not None:
                    logging.info(f"File not found: {item['file_path']}, replaced with: {os.path.basename(video_path)}")

        
        is_video = video_path is not None
        
        # slot = item['assigned_slot'] - 1  # Convert to 0-indexed
        
        # # Calculate frame_start and frame_end
        # if slot == 0:
        #     frame_start = 0
        # else:
        #     frame_start = int(beats[slot - 1] * fps)
        
        # frame_end = int(beats[slot] * fps) - 1  # Subtract 1 to avoid overlap
        
        if is_video:
            full_filename = os.path.basename(video_path) if video_path else item['file_path']

            element = {
                "element_type": "video",
                "description": "Main video clip with effects",
                "frame_start": frame_start,
                "frame_end": frame_end,
                "media_path": full_filename,
                "clip_start_time": offset,  # Seconds to skip from source start
                "clip_end_time": offset + duration,  # End position in source
                "clip_start_frame": 0,
                "clip_end_frame": 1000,
                "layer_order": 2,
                "opacity": 1.0,
                "rotation": 0,
                "effects": "blur",
                "transitions": [],  # New field
                "audio_settings": {
                    "volume": 1.0,
                    "mute": False
                }
            }
        else:
            full_filename = os.path.basename(image_path) if image_path else item['file_path']

            element = {
                "element_type": "image",
                "description": "Overlay logo with effects",
                "frame_start": frame_start,
                "frame_end": frame_end,
                "media_path": full_filename,
                "layer_order": 2,
                "opacity": 1,
                "rotation": 0,
                "transitions": [],  # New field
                "effects": "glow"
            }
        
        elements.append(element)
    
    # Add the elements to the video template
    video_template['template']['elements'] = elements
    
    # Save the updated video template
    with open(template_path, 'w') as f:
        json.dump(video_template, f, indent=2)
    
    print(f"Video elements added to video template at {template_path}")

@log_execution
@handle_errors
def add_default_transitions():
    """
    Add default transitions and set aspect ratio based on majority orientation
    """
    # Fixed paths
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    template_path = os.path.join(reel_path, 'video_template.json')
    
    # Load image descriptions and element sequence
    with open(os.path.join(reel_path, 'gpt_story_prompt.json'), 'r') as f:
        story_prompt = json.load(f)
    
    with open(os.path.join(reel_path, 'element_sequence.json'), 'r') as f:
        element_sequence = json.load(f)
    
    # Get list of files in sequence
    sequence_files = [item['file_path'].split('.')[0] for item in element_sequence]  # Remove extension
    
    # Count orientations only for files in sequence
    landscape_count = sum(
        1 for img in story_prompt['images'] 
        if img['file_path'].split('.')[0] in sequence_files 
        and img['orientation'] == 'landscape'
    )
    portrait_count = sum(
        1 for img in story_prompt['images'] 
        if img['file_path'].split('.')[0] in sequence_files 
        and img['orientation'] == 'portrait'
    )
    
    # Load the template
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    # Set aspect ratio based on majority
    if landscape_count >= portrait_count:
        template['template']['aspect_ratio'] = [16, 9]
    else:
        template['template']['aspect_ratio'] = [9, 16]
    
    # Get fps from template for calculating duration
    fps = int(template['template']['fps'])
    transition_duration = int(fps)/6  # 1/6 second transition

    # Process each element
    for element in template['template']['elements']:
        frame_start = element['frame_start']
        frame_end = element['frame_end']
        
        # Add fade in and fade out transitions
        element['transitions'] = [
            {
                "type": "fade_in",
                "frame_start": frame_start,
                "frame_end": frame_start + transition_duration
            },
            {
                "type": "fade_out",
                "frame_start": frame_end - transition_duration,
                "frame_end": frame_end
            }
        ]
    
    # Save the updated template
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Added transitions to {len(template['template']['elements'])} elements")
    print(f"Counted orientations for {len(sequence_files)} files: {landscape_count} landscape, {portrait_count} portrait")
    print(f"Set aspect ratio to {template['template']['aspect_ratio']}")

def add_transitions():
    """
    Add default transitions and set aspect ratio based on majority orientation
    """
    # Fixed paths
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    template_path = os.path.join(reel_path, 'video_template.json')
    
    # Load image descriptions and element sequence
    with open(os.path.join(reel_path, 'gpt_story_prompt.json'), 'r') as f:
        story_prompt = json.load(f)
    
    with open(os.path.join(reel_path, 'element_sequence.json'), 'r') as f:
        element_sequence = json.load(f)
    
    # Get list of files in sequence
    sequence_files = [item['file_path'].split('.')[0] for item in element_sequence]  # Remove extension
    
    # Count orientations only for files in sequence
    landscape_count = sum(
        1 for img in story_prompt['images'] 
        if img['file_path'].split('.')[0] in sequence_files 
        and img['orientation'] == 'landscape'
    )
    portrait_count = sum(
        1 for img in story_prompt['images'] 
        if img['file_path'].split('.')[0] in sequence_files 
        and img['orientation'] == 'portrait'
    )
    
    # Create GPT Data Dict
    gpt_data_dict = {}
    for img in story_prompt['images']: 
        if img['file_path'].split('.')[0] in sequence_files:
            gpt_data_dict[img['file_path']] = img['description']
    
    # GPT Assistant
    data = json.dumps(gpt_data_dict, indent=2)
    assistant_id = 'asst_HvQLjAcy7l6KaIDPkluu3LWk'
    gpt_assistant = GPT_assistant(assistant_id)
    thread = gpt_assistant.create_thread()
    gpt_assistant.add_message_to_thread(thread, data)
    run = gpt_assistant.run_thread_on_assistant(thread)
    response = gpt_assistant.check_run_status_and_respond(thread, run)
    gpt_response = response.replace("```json\n", "").replace("\n```", "")
    print(gpt_response)
    gpt_response = json.loads(gpt_response)
    print(gpt_response)
    
    # Load the template
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    # Set aspect ratio based on majority
    if landscape_count >= portrait_count:
        template['template']['aspect_ratio'] = [16, 9]
    else:
        template['template']['aspect_ratio'] = [9, 16]
    
    # Get fps from template for calculating duration
    fps = int(template['template']['fps'])
    transition_duration = int(fps)/6  # 1/6 second transition


    # Process each element
    for element in template['template']['elements']:
        frame_start = element['frame_start']
        frame_end = element['frame_end']
        
        media_path = element["media_path"].split('-')[0] + '.' + element["media_path"].split('-')[-1].split('.')[-1]
        print(media_path)
        transition = gpt_response.get(media_path, "No description available")
        print(transition)
        
        # Add fade in and fade out transitions
        try:
            element['transitions'] = [
                {
                    "type": transition["entry"],
                    "frame_start": frame_start,
                    "frame_end": frame_start + transition_duration
                },
                {
                    "type": transition["exit"],
                    "frame_start": frame_end - transition_duration,
                    "frame_end": frame_end
                }
            ]
        except:
            element['transitions'] = [
                {
                    "type": "fade_in",
                    "frame_start": frame_start,
                    "frame_end": frame_start + transition_duration
                },
                {
                    "type": "fade_out",
                    "frame_start": frame_end - transition_duration,
                    "frame_end": frame_end
                }
            ]
    
    # Save the updated template
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Added transitions to {len(template['template']['elements'])} elements")
    print(f"Counted orientations for {len(sequence_files)} files: {landscape_count} landscape, {portrait_count} portrait")
    print(f"Set aspect ratio to {template['template']['aspect_ratio']}")

# Example usage:
# create_video_elements()
import os
import json
from PIL import Image, ExifTags
import shutil
from tqdm import tqdm
from transitions import TransitionManager


@log_execution
@handle_errors
def create_frames_from_template_image():

    # Fixed paths
    base_path = os.path.join(config.User_ID, config.Chat_ID)
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    
    # Apply LUT processing if lut.json exists
    lut_mapping_path = os.path.join(reel_path, 'lut.json')
    if os.path.exists(lut_mapping_path):
        print("🎨 Applying LUT processing to images...")
        success = apply_lut_with_mapping()
        if success:
            print("✅ LUT processing completed successfully")
        else:
            print("⚠️ LUT processing failed, continuing with original images")


    def resize_and_crop_image(
        input_path,
        output_path,
        target_size,
        opacity=1.0,
        rotation=0,
        center_weight=5.0,
        face_weight=50.0,
        padding_weight=1.0
    ):
        # Assume base_path is defined globally or pass it as an additional parameter if needed
        # base_path = os.path.join(config.User_ID, config.Chat_ID)
        json_path = os.path.join(base_path, 'image_face_assignments.json')

        # with Image.open(input_path) as img:
        img = HelperFunctions.load_image(input_path)
        if img is not None:
            img = Image.fromarray(img)

            # Handle EXIF orientation
            try:
                # Get orientation tag code
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break

                exif = img._getexif()
                if exif is not None:
                    exif = dict(exif.items())
                    orientation_value = exif.get(orientation, None)

                    if orientation_value == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError, TypeError):
                # No EXIF data or orientation tag
                pass

            # Apply rotation
            if rotation != 0:
                img = img.rotate(rotation, expand=True)

            # Apply opacity
            if opacity < 1.0:
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                alpha = img.split()[3]
                alpha = Image.eval(alpha, lambda a: int(a * opacity))
                img.putalpha(alpha)

            orig_width, orig_height = img.size
            # After loading image
            # orig_width, orig_height = img.size
            initial_width = orig_width   # Add this
            initial_height = orig_height # Add this

            # Load face data
            with open(json_path, 'r') as f:
                all_face_data = json.load(f)

            # Normalize keys in all_face_data to filenames without extensions
            face_data_normalized = {}
            for key, value in all_face_data.items():
                # Replace backslashes with forward slashes and extract the filename
                filename = os.path.splitext(os.path.basename(key.replace('\\', '/')))[0]
                face_data_normalized[filename] = value

            # Normalize input filename to match keys in face_data_normalized
            # Extract filename from input_path
            input_filename_no_ext = os.path.splitext(os.path.basename(input_path))[0]

            # Retrieve image data and faces
            image_data = face_data_normalized.get(input_filename_no_ext, {})
            faces = image_data.get('faces', [])

            # Add this print statement
            # print(f"Found {len(faces)} faces in image '{input_filename_no_ext}'.")
            # for idx, face in enumerate(faces, start=1):
                # print(f"Face {idx}: {face}")


            # Pad the image if necessary
            # needs_padding = initial_width != initial_height


            # Extract target dimensions first
            target_width, target_height = target_size
            
            # Check if source and target have same orientation
            source_is_landscape = orig_width > orig_height
            target_is_landscape = target_width > target_height
            same_orientation = (source_is_landscape == target_is_landscape)

            # Only pad when doing cross-orientation transformations
            needs_padding = initial_width != initial_height and not same_orientation

            if needs_padding:
                max_dim = max(orig_width, orig_height)
            # if orig_width != orig_height:
                # max_dim = max(orig_width, orig_height)
                # Handle RGBA images for opacity support
                # if img.mode == 'RGBA':
                #     padded_img = Image.new('RGBA', (max_dim, max_dim), (0, 0, 0, 0))
                # else:
                #     padded_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
                
                # Calculate average color from image
                img_array = np.array(img)
                avg_color = tuple(int(c) for c in np.mean(img_array, axis=(0,1)))
                if img.mode == 'RGBA':
                    padded_img = Image.new('RGBA', (max_dim, max_dim), avg_color + (0,))
                else:
                    padded_img = Image.new('RGB', (max_dim, max_dim), avg_color)
                x_offset = (max_dim - orig_width) // 2
                y_offset = (max_dim - orig_height) // 2
                padded_img.paste(img, (x_offset, y_offset))
                img = padded_img
                orig_width, orig_height = img.size

                # Adjust face bounding boxes
                for face in faces:
                    face['bbox'][0] += x_offset
                    face['bbox'][1] += y_offset
                    face['bbox'][2] += x_offset
                    face['bbox'][3] += y_offset

            target_ratio = target_width / target_height

            # Now compute x_coords and y_coords
            y_coords, x_coords = np.mgrid[0:orig_height, 0:orig_width]
            x_center = (orig_width - 1) / 2
            y_center = (orig_height - 1) / 2

            # Distance from center normalized between 0 and 1
            distance_from_center = np.sqrt(
                ((x_coords - x_center) / (orig_width / 2)) ** 2 +
                ((y_coords - y_center) / (orig_height / 2)) ** 2
            )

            # Center importance (higher at center, lower at edges)
            center_importance = np.exp(-center_weight * distance_from_center ** 2)

            # Initialize importance map
            importance_map = center_importance.copy()

            if needs_padding:
                padding_mask = np.ones((max_dim, max_dim))
                padding_mask[y_offset:y_offset+initial_height, 
                            x_offset:x_offset+initial_width] = 0
                importance_map[padding_mask == 1] = 0
            

            # Add face importance by radiating from face centers
            for face in faces:
                x1, y1, x2, y2 = map(int, face['bbox'])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_width - 1, x2)
                y2 = min(orig_height - 1, y2)

                # Compute face center and size
                face_center_x = (x1 + x2) / 2
                face_center_y = (y1 + y2) / 2
                face_width = x2 - x1
                face_height = y2 - y1
                face_size = max(face_width, face_height)

                # Create a Gaussian importance map centered at the face
                sigma = face_size * 0.5  # Adjust the spread based on face size
                if sigma == 0:
                    sigma = 1  # Avoid division by zero
                gaussian = np.exp(-(
                    ((x_coords - face_center_x) ** 2 + (y_coords - face_center_y) ** 2) /
                    (2 * sigma ** 2)
                ))
                importance_map += face_weight * gaussian

            
            
            # Ensure that importance map is not zero to avoid division by zero
            # total_importance = np.sum(importance_map)
            # if total_importance == 0:
            #     total_importance = 1

            # total_importance = np.sum(np.maximum(importance_map, 0))  # Only positive values
            total_importance = np.sum(np.abs(importance_map))  # Change maximum to abs

            x_mass = np.sum(importance_map * x_coords) / total_importance
            y_mass = np.sum(importance_map * y_coords) / total_importance

            

            # Determine initial crop dimensions
            if target_ratio > 1:
                crop_width = min(orig_width, orig_height * target_ratio)
                crop_height = crop_width / target_ratio
            else:
                crop_height = min(orig_height, orig_width / target_ratio)
                crop_width = crop_height * target_ratio

            # Determine initial crop box coordinates
            x1 = int(round(x_mass - crop_width / 2))
            y1 = int(round(y_mass - crop_height / 2))
            x2 = int(round(x1 + crop_width))
            y2 = int(round(y1 + crop_height))

            # Ensure the crop rectangle stays within the image boundaries
            x1 = max(0, min(x1, orig_width - crop_width))
            y1 = max(0, min(y1, orig_height - crop_height))
            x2 = int(round(x1 + crop_width))
            y2 = int(round(y1 + crop_height))

            # After initial crop box coordinates
            if needs_padding:
                if x1 < x_offset: 
                    x1 = x_offset
                    x2 = min(orig_width, x1 + int(crop_width))
                if x2 > (x_offset + initial_width):
                    x2 = x_offset + initial_width
                    x1 = max(0, x2 - int(crop_width))
                if y1 < y_offset:
                    y1 = y_offset
                    y2 = min(orig_height, y1 + int(crop_height))
                if y2 > (y_offset + initial_height):
                    y2 = y_offset + initial_height
                    y1 = max(0, y2 - int(crop_height))
            
            # Finally ensure bounds
            x1 = max(0, min(x1, orig_width - crop_width))
            y1 = max(0, min(y1, orig_height - crop_height))
            x2 = int(round(x1 + crop_width))
            y2 = int(round(y1 + crop_height))

            # Crop and resize
            img_cropped = img.crop((x1, y1, x2, y2))
            img_resized = img_cropped.resize(target_size, Image.LANCZOS)

            # Apply opacity again after resizing if needed
            if opacity < 1.0 and img_resized.mode == 'RGBA':
                alpha = img_resized.split()[3]
                alpha = Image.eval(alpha, lambda a: int(a * opacity))
                img_resized.putalpha(alpha)

            # Determine the output format based on the extension of output_path
            _, ext = os.path.splitext(output_path)
            ext = ext.lower()

            # Map the file extension to the corresponding PIL format
            ext_to_format = {
                '.jpg': 'JPEG',
                '.jpeg': 'JPEG',
                '.png': 'PNG',
                '.bmp': 'BMP',
                '.gif': 'GIF',
                # Add other formats if needed
            }

            # Get the format corresponding to the file extension
            output_format = ext_to_format.get(ext, 'PNG')  # Default to 'PNG' if extension not recognized

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save the image using the correct format
            img_resized.save(output_path, format=output_format)

        return output_path

    
    template_path = os.path.join(base_path, config.Reel_ID, 'video_template.json')

    # Load the template
    with open(template_path, 'r') as f:
        template = json.load(f)['template']

    # Calculate resolution based on aspect ratio and given resolution
    aspect_ratio = template['aspect_ratio']
    height = template['resolution']
    width = int((height * aspect_ratio[0]) / aspect_ratio[1])
    resolution = (width, height)


    # Get video details
    fps = int(template['fps'])
    total_frames = int(fps * max(element['frame_end'] for element in template['elements']))

    # Process each element
    for element in tqdm(template['elements'], desc="Processing Image Frames"):
        if element['element_type'] == 'image':
            # Create Images folder if it doesn't exist
            layer_order = element['layer_order']
            opacity = element['opacity']
            rotation = element['rotation']
            images_folder = os.path.join(reel_path, 'Frames', f"{layer_order}_Images")
            os.makedirs(images_folder, exist_ok=True)
            # Resize the image
            input_path = os.path.join(base_path, "Media", "Images", element['media_path'])
            output_path = os.path.join(images_folder, f"resized_{element['media_path']}")

            # resize_and_crop_image(input_path, output_path, target_size=resolution,opacity=opacity,rotation=rotation)
            resize_and_crop_image(input_path, output_path, target_size=resolution, opacity=opacity, rotation=rotation)


            # Create frames
            start_frame = element['frame_start']
            end_frame = element['frame_end']
            base_frame = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)


            # Create each frame with transitions
            for frame_num in range(start_frame, end_frame + 1):
                frame_path = os.path.join(images_folder, f"render_{frame_num:04d}.png")
                
                # If element has transitions, apply them
                if 'transitions' in element and element['transitions']:
                    processed_frame = TransitionManager.process_frame_with_transitions(
                        base_frame.copy(),  # Use copy to avoid modifying original
                        frame_num,
                        element['transitions']
                    )
                    cv2.imwrite(frame_path, processed_frame)
                else:
                    # No transitions, just copy the frame as before
                    shutil.copy(output_path, frame_path)

            # for frame_num in range(start_frame, end_frame + 1):
            #     frame_path = os.path.join(images_folder, f"render_{frame_num:04d}.png")
            #     shutil.copy(output_path, frame_path)

    # print(f"Created {total_frames} frames in {images_folder}")
    print(f"Resolution: {width}x{height}")



# Usage
# create_frames_from_template_image()
import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

@log_execution
@handle_errors
def create_frames_from_template_video():
    
    # Fixed paths
    base_path = os.path.join(config.User_ID, config.Chat_ID)
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    
    # Apply LUT processing if lut.json exists
    lut_mapping_path = os.path.join(reel_path, 'lut.json')
    if os.path.exists(lut_mapping_path):
        print("🎨 Applying LUT processing to videos...")
        success = apply_lut_with_mapping()
        if success:
            print("✅ LUT processing completed successfully")
        else:
            print("⚠️ LUT processing failed, continuing with original videos")

    def process_video_frame(frame, size, opacity=1.0, rotation=0):
        # Convert BGR to RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Apply rotation
        if rotation != 0:
            image = image.rotate(rotation, expand=True)

        # Apply opacity
        if opacity < 1.0:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            data = np.array(image)
            data[:, :, 3] = (data[:, :, 3] * opacity).astype(int)
            image = Image.fromarray(data)

        # Resize and crop logic
        # img_ratio = image.width / image.height
        # target_ratio = size[0] / size[1]
        
        # if img_ratio > target_ratio:
        #     new_width = int(target_ratio * image.height)
        #     offset = (image.width - new_width) // 2
        #     resize_image = image.crop((offset, 0, image.width - offset, image.height))
        # elif img_ratio < target_ratio:
        #     new_height = int(image.width / target_ratio)
        #     offset = (image.height - new_height) // 2
        #     resize_image = image.crop((0, offset, image.width, image.height - offset))
        # else:
        #     resize_image = image
        
        # resize_image = resize_image.resize(size, Image.LANCZOS)
        
        # return np.array(resize_image)
    

        # Replace the resize and crop logic in process_video_frame with this:
        img_ratio = image.width / image.height
        target_ratio = size[0] / size[1]
        
        # Calculate average color for background padding
        img_array = np.array(image)
        avg_color = tuple(int(c) for c in np.mean(img_array, axis=(0,1)))
        
        # Use 60% preservation 
        # preserve_percentage = 60
        # weight = preserve_percentage / 100.0
        # intermediate_ratio = (img_ratio * weight) + (target_ratio * (1 - weight))

        # Check if source and target have same orientation
        source_is_landscape = img_ratio > 1.0
        target_is_landscape = target_ratio > 1.0
        same_orientation = (source_is_landscape == target_is_landscape)

        if same_orientation:
            # Same orientation: force full fill to avoid padding strips
            intermediate_ratio = target_ratio
        else:
            # Different orientations: use 60% preservation for better composition
            preserve_percentage = 60
            weight = preserve_percentage / 100.0
            intermediate_ratio = (img_ratio * weight) + (target_ratio * (1 - weight))
        
        
        # Determine dimensions for cropping and final image
        if img_ratio > target_ratio:  # Image is wider than target
            if intermediate_ratio > target_ratio:
                new_height = int(size[0] / intermediate_ratio)
                new_width = size[0]
                crop_height = image.height
                crop_width = int(crop_height * intermediate_ratio)
            else:
                new_height = size[1]
                new_width = size[0]
                crop_height = image.height
                crop_width = int(crop_height * target_ratio)
        else:  # Image is taller than target
            if intermediate_ratio < target_ratio:
                new_width = int(size[1] * intermediate_ratio)
                new_height = size[1]
                crop_width = image.width
                crop_height = int(crop_width / intermediate_ratio)
            else:
                new_width = size[0]
                new_height = size[1]
                crop_width = image.width
                crop_height = int(crop_width / target_ratio)
        
        # Crop to intermediate ratio
        if crop_width < image.width:
            left = (image.width - crop_width) // 2
            right = left + crop_width
            top = 0
            bottom = image.height
        elif crop_height < image.height:
            top = (image.height - crop_height) // 2
            bottom = top + crop_height
            left = 0
            right = image.width
        else:
            left, top, right, bottom = 0, 0, image.width, image.height
        
        cropped_image = image.crop((left, top, right, bottom))
        
        # Create background with average color
        # background = Image.new('RGB' if image.mode != 'RGBA' else 'RGBA', size, avg_color + ((255,) if image.mode == 'RGBA' else ()))
        
        # # Resize and center
        # resized_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)
        # x_position = (size[0] - new_width) // 2
        # y_position = (size[1] - new_height) // 2
        # background.paste(resized_image, (x_position, y_position))
        
        # return np.array(background)
    
        # For same orientation, resize directly to fill entire target (no background needed)
        if same_orientation:
            # Direct resize to target size - no padding
            resized_image = cropped_image.resize(size, Image.LANCZOS)
            return np.array(resized_image)
        else:
            # Create background with average color for cross-orientation cases
            background = Image.new('RGB' if image.mode != 'RGBA' else 'RGBA', size, avg_color + ((255,) if image.mode == 'RGBA' else ()))
            
            # Resize and center
            resized_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)
            x_position = (size[0] - new_width) // 2
            y_position = (size[1] - new_height) // 2
            background.paste(resized_image, (x_position, y_position))
            
            return np.array(background)

    # Fixed paths
    base_path = os.path.join(config.User_ID, config.Chat_ID)
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    template_path = os.path.join(base_path, config.Reel_ID, 'video_template.json')

    # Load the template
    with open(template_path, 'r') as f:
        template = json.load(f)['template']

    # Calculate resolution based on aspect ratio and given resolution
    aspect_ratio = template['aspect_ratio']
    height = template['resolution']
    width = int((height * aspect_ratio[0]) / aspect_ratio[1])
    resolution = (width, height)

    # Get video details
    fps = int(template['fps'])

    # Process each video element
    for element in tqdm(template['elements'], desc="Processing Video Frames"):
        if element['element_type'] == 'video':
            # Create Video Frames folder
            layer_order = element['layer_order']
            opacity = element['opacity']
            rotation = element['rotation']
            frames_folder = os.path.join(reel_path, 'Frames', f"{layer_order}_Videos")
            os.makedirs(frames_folder, exist_ok=True)

            # Open the video file (ignoring extension)
            video_folder = os.path.join(base_path, "Media", "Videos")
            video_name = os.path.splitext(element['media_path'])[0]
            video_path = None
            for file in os.listdir(video_folder):
                if file.startswith(video_name):
                    video_path = os.path.join(video_folder, file)
                    break
            
            if not video_path:
                print(f"Video file not found for {video_name}")
                continue

            # cap = cv2.VideoCapture(video_path)
            cap = HelperFunctions.load_video(video_path)

            
            # Get video properties
            video_fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate start/end frames from source video times
            clip_start_time = element.get('clip_start_time', 0.0)
            clip_end_time = element.get('clip_end_time', 0.0)

            # Convert times to frame numbers in source video
            if clip_end_time > 0:
                # NEW FORMAT: Use time-based extraction
                start_frame = int(clip_start_time * video_fps)
                end_frame = int(clip_end_time * video_fps)
            else:
                # OLD FORMAT: Use hardcoded frame values (backward compatibility)
                start_frame = element.get('clip_start_frame', 0)
                end_frame = total_video_frames - 1

            # Validate bounds
            end_frame = min(end_frame, total_video_frames - 1)

            # Calculate frame selection parameters
            frame_ratio = video_fps / fps

            last_valid_frame = None
            for template_frame in range(element['frame_start'], element['frame_end'] + 1):
                # Calculate the corresponding frame in the original video
                video_frame = int(start_frame + (template_frame - element['frame_start']) * frame_ratio)

                # Replace the if condition and subsequent code with this:
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
                ret, frame = cap.read()

                if ret:
                    last_valid_frame = frame
                else:
                    print(f"Failed to read frame {video_frame} from video, using last valid frame")
                    if last_valid_frame is None:
                        print(f"No valid frames found in video")
                        break

                # Process the frame (use last_valid_frame)
                processed_frame = process_video_frame(last_valid_frame, resolution, opacity, rotation)

                # Apply transitions if they exist
                if 'transitions' in element and element['transitions']:
                    processed_frame = TransitionManager.process_frame_with_transitions(
                        processed_frame,
                        template_frame,
                        element['transitions']
                    )

                # Save the frame
                frame_path = os.path.join(frames_folder, f"render_{template_frame:04d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

            cap.release()

    print(f"Created frames in layer-specific folders")
    print(f"Resolution: {width}x{height}")

# Usage
# create_frames_from_template_video()
import os
import cv2
import numpy as np
from tqdm import tqdm
import json
import re

@log_execution
@handle_errors
def combine_frames_to_video(deleteframes = True):
    # Fixed paths
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    frames_path = os.path.join(reel_path, 'Frames')
    template_path = os.path.join(reel_path, 'video_template.json')

    # Load the template
    with open(template_path, 'r') as f:
        template = json.load(f)['template']

    # Get video details
    fps = int(template['fps'])
    aspect_ratio = template['aspect_ratio']
    height = template['resolution']
    width = int((height * aspect_ratio[0]) / aspect_ratio[1])
    resolution = (width, height)

    # Get all frame folders and sort them by layer (ascending order)
    frame_folders = [f for f in os.listdir(frames_path) if os.path.isdir(os.path.join(frames_path, f))]
    frame_folders.sort(key=lambda x: int(x.split('_')[0]))

    # Determine the total number of frames
    max_frames = 0
    for folder in frame_folders:
        folder_path = os.path.join(frames_path, folder)
        frames = [f for f in os.listdir(folder_path) if f.startswith('render_') and f.endswith('.png')]
        if frames:
            max_frame_num = max([int(re.search(r'(\d+)', f).group(1)) for f in frames])
            max_frames = max(max_frames, max_frame_num)

    # Create video writer
    output_path = os.path.join(reel_path, 'video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    # Combine frames
    for frame_num in tqdm(range(max_frames + 1), desc="Combining frames"):
        combined_frame = np.zeros((height, width, 4), dtype=np.uint8)
        
        for folder in frame_folders:
            folder_path = os.path.join(frames_path, folder)
            frame_path = os.path.join(folder_path, f'render_{frame_num:04d}.png')
            
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                
                if frame is None:
                    continue
                
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                
                # Alpha blending
                alpha = frame[:, :, 3] / 255.0
                combined_frame = combined_frame * (1 - alpha[:, :, np.newaxis]) + frame * alpha[:, :, np.newaxis]

        # Convert back to BGR for video writing
        combined_frame = cv2.cvtColor(combined_frame.astype(np.uint8), cv2.COLOR_BGRA2BGR)
        out.write(combined_frame)

    out.release()

    # Delete the Frames folder
    if deleteframes:
        try:
            shutil.rmtree(frames_path)
            print(f"Frames folder deleted: {frames_path}")
        except Exception as e:
            print(f"Error deleting Frames folder: {e}")
        
    print(f"Video created and saved as {output_path}")

# Usage
# combine_frames_to_video()
import os
import json
import subprocess

@log_execution
@handle_errors
def add_audio_to_video_and_cleanup(retain_video_audio=False):

    def _load_sequence_data(reel_path):
        """Load sequence data from either sync.json or element_sequence.json (fallback)"""
        # Try sync.json first (new format)
        sync_json_path = os.path.join(reel_path, 'sync.json')
        if os.path.exists(sync_json_path):
            with open(sync_json_path, 'r') as f:
                sync_data = json.load(f)
            return sync_data.get("sync_sequence", [])
        
        # Fallback to element_sequence.json (old format)
        element_sequence_path = os.path.join(reel_path, 'element_sequence.json')
        with open(element_sequence_path, 'r') as f:
            return json.load(f)
    
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    template_path = os.path.join(reel_path, 'video_template.json')

    with open(template_path, 'r') as f:
        template = json.load(f)['template']

    # Load element sequence to check for qualifying videos
    element_sequence = _load_sequence_data(reel_path)

    # element_sequence_path = os.path.join(reel_path, 'element_sequence.json')
    # with open(element_sequence_path, 'r') as f:
    #     element_sequence = json.load(f)

    # Print debug information
    print(f"Processing video audio with {len(element_sequence)} elements")
    print(f"First element: {element_sequence[0]['file_path']}")
    print(f"Last element: {element_sequence[-1]['file_path']}")

    template_audio = Path(template['audio'])
    # spotify_base = Path("Assets/spotify")
    # Use the same logic as your main script
    current_dir = os.getcwd()
    # parent_dir = os.path.dirname(current_dir)
    spotify_base = Path(os.path.join(current_dir, 'Assets', 'spotify'))

    def normalize_for_comparison(name):
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    audio_file = None
    template_name = template_audio.stem

    # Check direct audio folder first
    direct_audio_path = spotify_base / 'audio'
    if direct_audio_path.exists():
        for file in direct_audio_path.glob("*.mp3"):
            if normalize_for_comparison(file.stem) == normalize_for_comparison(template_name):
                audio_file = str(file)
                break

    # audio_file = None
    # template_name = template_audio.stem

    # print(f"Looking for audio file: {template_name}")
    # print(f"Template audio path: {template['audio']}")
    # print(f"Spotify base path: {spotify_base}")

    # # Check direct audio folder first
    # direct_audio_path = spotify_base / 'audio'
    # print(f"Direct audio path: {direct_audio_path}")
    # print(f"Direct audio path exists: {direct_audio_path.exists()}")

    # if direct_audio_path.exists():
    #     print("Files in audio directory:")
    #     mp3_files = list(direct_audio_path.glob("*.mp3"))
    #     for file in mp3_files:
    #         print(f"  Found: {file.name}, stem: {file.stem}")
    #         print(f"  Normalized file: {normalize_for_comparison(file.stem)}")
    #         print(f"  Normalized template: {normalize_for_comparison(template_name)}")
    #         if normalize_for_comparison(file.stem) == normalize_for_comparison(template_name):
    #             audio_file = str(file)
    #             print(f"  MATCH! Using: {audio_file}")
    #             break
        
    #     if not mp3_files:
    #         print("  No .mp3 files found!")
    # else:
    #     print(f"Audio directory does not exist: {direct_audio_path}")

    # If not found, search through subdirectories
    if not audio_file:
        for subfolder in spotify_base.iterdir():
            if subfolder.is_dir():
                audio_path = subfolder / 'audio'
                if audio_path.exists():
                    for file in audio_path.glob("*.mp3"):
                        if normalize_for_comparison(file.stem) == normalize_for_comparison(template_name):
                            audio_file = str(file)
                            break
            if audio_file:
                break

    if not audio_file:
        print(f"Error: Audio file matching '{template_name}' not found in any spotify subdirectories")
        return
    
    current_time = datetime.now().strftime("%m%d%H%M")
    video_with_audio_path = os.path.join(reel_path, f'final_video_{9999999999 - int(current_time):010d}.mp4')
    video_without_audio_path = os.path.join(reel_path, 'video.mp4')

    # Get video duration
    video_duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_without_audio_path]
    video_duration = float(subprocess.check_output(video_duration_cmd).decode('utf-8').strip())

    # Get song duration
    audio_duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_file]
    audio_duration = float(subprocess.check_output(audio_duration_cmd).decode('utf-8').strip())

    # Get overall duration
    duration = min(video_duration, audio_duration)

    # Get audio settings
    audio_settings = template.get('audio_settings', {})
    audio_start_time = audio_settings.get('start_time', 0)

    # Simple logic based on retain_video_audio parameter
    if retain_video_audio:
        # Mix video audio with song audio for ALL videos
        print("Mixing video audio with song audio")
        cmd = [
            'ffmpeg',
            '-y',
            '-i', video_without_audio_path,
            '-ss', f'{audio_start_time}',
            '-i', audio_file,
            '-t', f'{duration}',
            '-c:v', 'copy',
            '-filter_complex', '[0:a]volume=1.0[va];[1:a]volume=0.5[sa];[va][sa]amix=inputs=2:duration=first[a]',
            '-map', '0:v:0',
            '-map', '[a]',
            '-c:a', 'aac',
            video_with_audio_path
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Video with mixed audio created and saved as {video_with_audio_path}")
            os.remove(video_without_audio_path)
            print(f"Removed video without audio: {video_without_audio_path}")
            return
        except subprocess.CalledProcessError as e:
            print(f"Error mixing audio: {e}")
            print(f"ffmpeg stderr output:\n{e.stderr.decode()}")
            print("Falling back to standard audio processing")

    # Default: Replace video audio with song audio only (no video audio retained)
    print("Replacing video audio with song audio only")
    
    # Modify the ffmpeg command to include the audio start time
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_without_audio_path,
        '-ss', f'{audio_start_time}',  # Start time offset for audio
        '-i', audio_file,
        '-t', f'{duration}',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        video_with_audio_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video with audio created and saved as {video_with_audio_path}")
        os.remove(video_without_audio_path)
        print(f"Removed video without audio: {video_without_audio_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error adding audio to video: {e}")
        print(f"ffmpeg stderr output:\n{e.stderr.decode()}")
    except Exception as e:
        print(f"An error occurred: {e}")


from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

@log_execution
@handle_errors
def fill_elements_gpt_o_two():
    # Get the current working directory
    current_dir = Path.cwd()
    
    # Look for .env file in current directory and parent directories
    env_path = None
    for parent in [current_dir] + list(current_dir.parents):
        possible_env = parent / '.env'
        if possible_env.exists():
            env_path = possible_env
            break
    
    if env_path is None:
        raise FileNotFoundError("Could not find .env file in current or parent directories")
    
    # Load environment variables from the specified path
    load_dotenv(dotenv_path=env_path)
    
    # Get the API key from environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Instantiate the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Define paths
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    story_prompt_path = os.path.join(reel_path, 'gpt_story_prompt.json')
    beat_alignment_prompt_path = os.path.join(reel_path, 'gpt_beat_alignment_prompt.json')
    
    # Load the story prompt
    with open(story_prompt_path, 'r') as f:
        story_prompt_data = json.load(f)
    
    # Load the beat alignment prompt
    with open(beat_alignment_prompt_path, 'r') as f:
        beat_alignment_prompt_data = json.load(f)
    
    # Extract data from prompts
    story_instructions = story_prompt_data["instructions"]
    user_prompt = story_prompt_data["user_prompt"]
    images = story_prompt_data["images"]
    beats = beat_alignment_prompt_data["beats"]
    beat_alignment_instructions = beat_alignment_prompt_data["instructions"]
    
    # Construct the messages for the entire conversation
    messages = [
        {"role": "system", "content": story_instructions},
        {
            "role": "user",
            "content": f"""
User Query: {user_prompt}

Images: {json.dumps(images, indent=2)}

Please provide a sequence of selected visuals based on the story creation instructions.
"""
        }
    ]
    
    # Send the story prompt to GPT and get the response
    try:
        story_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )
        
        # Add the assistant's response to the messages
        messages.append({"role": "assistant", "content": story_response.choices[0].message.content})
        
        # Add the beat alignment instructions and request
        messages.append({"role": "system", "content": beat_alignment_instructions})
        messages.append({
            "role": "user",
            "content": f"""
Now, please align the selected visuals with the following beats:

Beats: {json.dumps(beats, indent=2)}

Provide the final sequence with timing information.
"""
        })
        
        # Send the beat alignment prompt in the same conversation
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )
        
        # Get the assistant's final reply
        final_reply = final_response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"An error occurred during the chat completion: {e}")
        return
    
    # Parse the final reply as JSON
    # try:
    #     json_start = final_reply.find('[')
    #     json_end = final_reply.rfind(']') + 1
    #     json_content = final_reply[json_start:json_end]
    #     final_sequence = json.loads(json_content)
    # except (json.JSONDecodeError, ValueError) as e:
    #     print("Failed to parse GPT final response as JSON:", e)
    #     return
    
    try:
        json_start = final_reply.find('[')
        json_end = final_reply.rfind(']') + 1
        json_content = final_reply[json_start:json_end]
        final_sequence = json.loads(json_content)
        
        seen_paths = set()
        crop_index = len(final_sequence)
        
        for i, item in enumerate(final_sequence):
            if item['file_path'] in seen_paths:
                crop_index = i
                break
            seen_paths.add(item['file_path'])
        
        final_sequence = final_sequence[:crop_index]
        
        element_sequence_path = os.path.join(reel_path, 'element_sequence.json')
        with open(element_sequence_path, 'w') as f:
            json.dump(final_sequence, f, indent=2)
            
        print(f"Final element sequence saved to {element_sequence_path}")
    except (json.JSONDecodeError, ValueError) as e:
        print("Failed to parse GPT final response as JSON:", e)
        return
    
    # Save the final element sequence to element_sequence.json
    element_sequence_path = os.path.join(reel_path, 'element_sequence.json')
    with open(element_sequence_path, 'w') as f:
        json.dump(final_sequence, f, indent=2)
    
    print(f"Final element sequence saved to {element_sequence_path}")

# Example usage:
# fill_elements_gpt_o_two()

# initialize_template()
# get_embedding_search_results()
# generate_prompt()
# # fill_elements_ordered()
# fill_elements_gpt_o()
# fill_elements_gpt_o_two()
# # fill_elements_gpt_o1()
# create_video_elements()
# create_frames_from_template_image()
# create_frames_from_template_video()
# combine_frames_to_video()
# add_audio_to_video_and_cleanup()

@log_execution
@handle_errors
def initialize_template_proxy():
    """Creates fixed template without any input parameters"""
    # global config.Reel_ID
    
    template = {
        "template": {
            "user_prompt": "best moments",
            "prompt_id": "best_moments",
            "song_beats": [1.0, 2.0, 5.6, 6.75, 8.25, 9.4, 10.9, 12.1, 13.6, 14.75],
            "updated_prompt": "best moments",
            "aspect_ratio": [16, 9],
            "resolution": 1080,
            "fps": "30",
            "audio": "Assets/Songs/big_dawgs.mp3",
            "srt": "Assets/SRT/big_dawgs.srt"
        }
    }
    
    # Update config.Reel_ID
    prompt_id = template["template"]["prompt_id"]
    config.Reel_ID = f"{config.Reel_ID}_{prompt_id}"
    
    # Create directory and save
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    os.makedirs(reel_path, exist_ok=True)
    
    template_path = os.path.join(reel_path, 'video_template.json')
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=4)
    
    print(f"Template saved as {template_path}")


@log_execution
@handle_errors
def get_embedding_results_proxy():
    # Load cluster data from appropriate cluster file
    base_path = os.path.join(config.User_ID, config.Chat_ID)
    cluster_file = RATING_TO_CLUSTERS.get(config.rating_filter or 'all', 'all_reel_clusters.json')
    
    try:
        with open(os.path.join(base_path, cluster_file), 'r') as f:
            cluster_data = json.load(f)
    except FileNotFoundError:
        print(f"Cluster file not found: {cluster_file}")
        return
    
    # Get current cluster number from config.Reel_ID 
    try:
        cluster_no = config.Reel_ID.split('_')[-1]  # Get last part after splitting
        
        # Verify cluster exists
        if str(cluster_no) not in cluster_data['clusters']:
            print(f"Cluster {cluster_no} not found in {cluster_file}")
            return
            
        # Get images for this cluster
        cluster_images = cluster_data['clusters'][str(cluster_no)]['images']
        
        if not cluster_images:
            print(f"No images found in cluster {cluster_no}")
            return
            
        # Format results
        search_results = []
        for image in cluster_images:
            result = {
                'file_path': image['file_path'],
                'similarity': 1.0,
                'description': image['description'],
                'faces': image.get('faces', [])  # Use get() with default
            }
            search_results.append(result)
        
        # Save results
        reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
        os.makedirs(reel_path, exist_ok=True)  # Ensure directory exists
        results_path = os.path.join(reel_path, 'search_results.json')
        with open(results_path, 'w') as f:
            json.dump(search_results, f, indent=4)
            
        print(f"Saved {len(search_results)} results to {results_path}")
            
    except (IndexError, KeyError) as e:
        print(f"Error processing cluster data: {str(e)}")



@log_execution
@handle_errors
@sync_with_drive(
    inputs=[
        'embeddings_all.pkl',
        'embeddings_average.pkl',
        'embeddings_good.pkl',
        'embeddings_very_good.pkl',
        'embeddings_best.pkl',
        'all_reel_clusters.json',
        'average_reel_clusters.json',
        'good_reel_clusters.json',
        'very_good_reel_clusters.json',
        'best_reel_clusters.json',
        'image_face_assignments.json',
        'video_face_assignments.json',
        os.path.join('Media', 'filename_mapping.json')
    ],
    outputs=[]
)
def process_reel_clusters():
    """Process all clusters without input parameters"""
    # global config.Reel_ID
    #config.User_ID = config.config.User_ID 
    #config.Chat_ID = config.config.Chat_ID
    if not config.User_ID or not config.Chat_ID:
        raise ValueError("config.User_ID and config.Chat_ID must be set")

    def get_checkpoint_status(cluster_no):
        checkpoint_file = os.path.join(config.User_ID, config.Chat_ID, 'checkpoints.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoints = json.load(f)
                filter_key = f"{config.rating_filter or 'all'}_{cluster_no}"
                return checkpoints.get(filter_key, {})
        return {}

    def update_checkpoint(cluster_no, step):
        checkpoint_file = os.path.join(config.User_ID, config.Chat_ID, 'checkpoints.json')
        checkpoints = {}
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoints = json.load(f)
        
        filter_key = f"{config.rating_filter or 'all'}_{cluster_no}"
        if filter_key not in checkpoints:
            checkpoints[filter_key] = {}
        checkpoints[filter_key][step] = True
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoints, f)
    
    # Define processing steps - separated into initial and common steps
    initial_batch_steps = [
        ('embedding', get_embedding_results_proxy),
        ('prompts', generate_two_prompts),
    ]

    initial_user_steps = [
        ('initialize', initialize_template),
        ('embedding', get_embedding_search_results),
        ('prompts', generate_two_prompts),
    ]

    common_steps = [
        ('elements', fill_elements_gpt_o_two),
        ('video_elements', create_video_elements),
        ('transitions', apply_transition_system),
        ('template_frames', create_frames_from_template_image),
        ('video_frames', create_frames_from_template_video),
        ('combine_frames', combine_frames_to_video),
        ('final_audio', add_audio_to_video_and_cleanup)
    ]

    if config.process_type == 'batch':
        # Batch Processing
        base_path = os.path.join(config.User_ID, config.Chat_ID)
        cluster_file = RATING_TO_CLUSTERS.get(config.rating_filter or 'all', 'all_reel_clusters.json')
        
        with open(os.path.join(base_path, cluster_file), 'r') as f:
            cluster_data = json.load(f)
            
        create_cluster_templates()

        # Process each cluster
        for cluster_no in cluster_data['clusters'].keys():
            print(f"\nProcessing cluster {cluster_no}")
            checkpoints = get_checkpoint_status(cluster_no)
            
            original_Reel_ID = config.Reel_ID
            config.Reel_ID = f"Reel{config.rating_filter}_{cluster_no}"
            
            try:
                # Process initial batch steps
                for step_name, step_func in initial_batch_steps:
                    if step_name not in checkpoints:
                        print(f"Running {step_name} for cluster {cluster_no}")
                        step_func()
                        update_checkpoint(cluster_no, step_name)
                    else:
                        print(f"Skipping completed step {step_name} for cluster {cluster_no}")
                
                # Process common steps
                for step_name, step_func in common_steps:
                    if step_name not in checkpoints:
                        print(f"Running {step_name} for cluster {cluster_no}")
                        step_func()
                        update_checkpoint(cluster_no, step_name)
                    else:
                        print(f"Skipping completed step {step_name} for cluster {cluster_no}")
                        
                print(f"Completed processing cluster {cluster_no}")
                
            except Exception as e:
                print(f"Error processing cluster {cluster_no}: {str(e)}")
                raise
                
            finally:
                config.Reel_ID = original_Reel_ID

    else:
        # User Processing
        print("\nProcessing user request")
        checkpoints = get_checkpoint_status('user')
        
        try:
            # Process initial user steps
            for step_name, step_func in initial_user_steps:
                if step_name not in checkpoints:
                    print(f"Running {step_name}")
                    step_func()
                    update_checkpoint('user', step_name)
                else:
                    print(f"Skipping completed step {step_name}")
            
            # Process common steps
            for step_name, step_func in common_steps:
                if step_name not in checkpoints:
                    print(f"Running {step_name}")
                    step_func()
                    update_checkpoint('user', step_name)
                else:
                    print(f"Skipping completed step {step_name}")
                    
            print("Completed processing user request")
            
        except Exception as e:
            print(f"Error processing user request: {str(e)}")
            raise


from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import re
import os


def create_cluster_templates():

    def analyze_cluster_keywords(User_ID, Chat_ID):
        base_path = os.path.join(User_ID, Chat_ID)
        
        custom_stops = {'image', 'video', 'overall', 'showing', 'shows', 'show', 'shown', 'visible', 'seen', 
                    'appears', 'displaying', 'wearing', 'looking'}
        
        person_terms = {
            1: 'person',
            2: 'couple',
            3: 'trio',
            4: 'group',
            5: 'group',
            6: 'group',
            7: 'crowd',
            8: 'crowd',
            9: 'crowd',
            10: 'crowd'
        }

        # Load clusters
        # with open(os.path.join(base_path, 'reel_clusters.json'), 'r') as f:
        #     clusters = json.load(f)['clusters']

        cluster_file = RATING_TO_CLUSTERS.get(config.rating_filter or 'all', 'all_reel_clusters.json')
        with open(os.path.join(base_path, cluster_file), 'r') as f:
            clusters = json.load(f)['clusters']
        
            
        # Load face data and extract names
        face_data_path = os.path.join(base_path, 'image_face_assignments.json')
        with open(face_data_path, 'r') as f:
            face_data = json.load(f)

        # 1. Convert names to lowercase when adding to set AND remove any whitespace:
        names = set()
        for image_data in face_data.values():
            for face in image_data.get('faces', []):
                if isinstance(face, dict) and 'label' in face:
                    names.add(face['label'].lower().strip())
                
        cluster_docs = {}
        cluster_people = {}
        
        for cluster_id, cluster in clusters.items():
            texts = []
            people_counts = []
            
            for img in cluster['images']:
                desc = img.get('description', '')
                if isinstance(desc, dict):
                    texts.extend(str(v) for v in desc.values())
                else:
                    texts.append(str(desc))
                
                faces = img.get('faces', [])
                people_counts.append(len(faces))
            
            median_people = int(np.median(people_counts)) if people_counts else 0
            cluster_people[cluster_id] = person_terms.get(median_people, 'crowd')
            
            text = " ".join(texts)
            text = text.lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            cluster_docs[cluster_id] = text
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            token_pattern=r'(?u)\b[a-z]{4,}\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(list(cluster_docs.values()))
        feature_names = vectorizer.get_feature_names_out()
        
        # Filter out custom stops and names
        valid_indices = [i for i, word in enumerate(feature_names) 
                        if word not in custom_stops and word not in names]
        
        keywords = {}
        cluster_stats = {}
        
        for cluster_id in clusters.keys():
            doc_idx = list(cluster_docs.keys()).index(cluster_id)
            tfidf_scores = tfidf_matrix[doc_idx].toarray()[0]
            
            valid_scores = [(i, tfidf_scores[i]) for i in valid_indices]
            top_indices = sorted(valid_scores, key=lambda x: x[1], reverse=True)[:10]
            
            top_words = [(feature_names[idx], float(score)) for idx, score in top_indices]
            keywords[cluster_id] = top_words
            
            person_term = cluster_people[cluster_id]
            cluster_stats[cluster_id] = {
                'image_count': len(clusters[cluster_id]['images']),
                'top_words': [(person_term, 1.0)] + top_words,
                'avg_word_score': np.mean([score for _, score in top_words])
            }
        
        return keywords, cluster_stats
    def match_clusters_with_songs(User_ID, Chat_ID):
        # Get cluster keywords
        keywords, stats = analyze_cluster_keywords(User_ID, Chat_ID)
        
        # Load song data
        # base_path = Path("Assets/spotify")
        # with open(base_path / "song_preprocessed_data.pkl", 'rb') as f:
        #     data = pickle.load(f)

        spotify_base = Path("Assets/spotify/embeddings") 
        with open(spotify_base / config.embeding_path, 'rb') as f:
            data = pickle.load(f)
        
        model = data['model']
        index = data['index']
        song_names = data['song_names']
        used_songs = set()
        
        matches = {}
        for cluster_id, cluster_info in stats.items():
            # Create search query from top words
            top_words = [word for word, _ in cluster_info['top_words']]
            search_query = " ".join(top_words)
            
            # Generate embedding
            query_embedding = model.encode([search_query])
            
            # Search for songs
            distances, indices = index.search(query_embedding.astype('float32'), k=20)
            
            # Find first unused song
            for idx, dist in zip(indices[0], distances[0]):
                song_name = song_names[idx]
                if song_name not in used_songs:
                    used_songs.add(song_name)
                    matches[cluster_id] = {
                        'keywords': top_words,
                        'song': song_name,
                        'score': 1 / (1 + dist)
                    }
                    break
        
        # Print results
        for cluster_id, match in matches.items():
            print(f"\nCluster {cluster_id}:")
            print(f"Keywords: {', '.join(match['keywords'])}")
            print(f"Assigned Song: {match['song']} (score: {match['score']:.2f})")
        
        return matches
    """Creates templates with matched songs for each cluster"""

    # global config.Reel_ID
    
    base_path = os.path.join(config.User_ID, config.Chat_ID)
    matches = match_clusters_with_songs(config.User_ID, config.Chat_ID)

    # Load cluster data
    # with open(os.path.join(base_path, 'reel_clusters.json'), 'r') as f:
    #     cluster_data = json.load(f)
    
    cluster_file = RATING_TO_CLUSTERS.get(config.rating_filter or 'all', 'all_reel_clusters.json')
    with open(os.path.join(base_path, cluster_file), 'r') as f:
        cluster_data = json.load(f)
    
    # Load spotify data
    spotify_data = load_spotify_data()
    
    # Correct path handling for audio analysis data
    spotify_base = Path("Assets/spotify")
    beats_dir = spotify_base / "beats"
    analysis_dir = spotify_base / "analysis"
    
    beats_data = {}
    analysis_data = {}
    
    # Load beats data
    for beat_file in beats_dir.glob("*_bars.json"):
        song_name = beat_file.stem.replace("_bars", "")
        try:
            with open(beat_file, 'r') as f:
                beats_data[song_name] = [bar['time'] for bar in json.load(f)]
        except json.JSONDecodeError:
            print(f"Error reading beats file for {song_name}")
            beats_data[song_name] = []
    
    # Load analysis data
    for analysis_file in analysis_dir.glob("*_analysis.json"):
        song_name = analysis_file.stem.replace("_analysis", "")
        try:
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
                # Get start time from third section (index 2) if available
                sections = analysis.get('sections', [])
                if len(sections) > 2:
                    analysis_data[song_name] = round(sections[2]['start'], 2)
                else:
                    analysis_data[song_name] = 0.0
        except json.JSONDecodeError:
            print(f"Error reading analysis file for {song_name}")
            analysis_data[song_name] = 0.0

    for cluster_id, match in matches.items():
        try:
            song_name = match['song']
            sanitized_song = song_name.lower().replace(' ', '_')

            cluster_images = cluster_data['clusters'][str(cluster_id)]['images']
            image_count = len(cluster_images)
            
            # Get analysis data
            analysis = spotify_data['analysis'].get(sanitized_song, {})
            audio_start_time = analysis.get('start_time', 0.0)
            section_times = analysis.get('sections', [])

            # Also keep the old method as fallback
            if audio_start_time == 0.0:
                audio_start_time = analysis_data.get(sanitized_song, 0.0)
            
            # Get beats data with fallback
            beats = spotify_data['beats'].get(sanitized_song, [])
            if not beats:
                beats = beats_data.get(sanitized_song, [])
            
            # Process beats - filter and adjust by start time
            adjusted_beats = [
                round(beat - audio_start_time, 2)
                for beat in beats
                if beat >= audio_start_time
            ]
            
            max_beats = image_count if image_count <= 20 else int(min(20, image_count / 1.75))
            beats = adjusted_beats[:max_beats]  # Limit beats
            
            # If no section times from spotify_data, try the old method
            if not section_times:
                try:
                    analysis_file = find_spotify_file(song_name, 'analysis')
                    if analysis_file:
                        with open(analysis_file, 'r') as f:
                            full_analysis = json.load(f)
                            sections = full_analysis.get('sections', [])
                            section_times = [
                                round(section['start'] - audio_start_time, 2)
                                for section in sections 
                                if section['start'] >= audio_start_time
                            ]
                except (json.JSONDecodeError, FileNotFoundError):
                    section_times = []
            
            # Find audio and lyrics files
            audio_file = find_spotify_file(song_name, 'audio')
            lyrics_file = find_spotify_file(song_name, 'lyrics')
            
            # If not found, try the old paths
            if not audio_file:
                audio_path = spotify_base / "audio" / f"{song_name}.mp3"
                if audio_path.exists():
                    audio_file = audio_path
                    
            if not lyrics_file:
                lyrics_path = spotify_base / "lyrics" / f"{sanitized_song}_lyrics.srt"
                if lyrics_path.exists():
                    lyrics_file = lyrics_path
            
            template = {
                "template": {
                    "user_prompt": " ".join(match['keywords']),
                    "prompt_id": f"cluster_{cluster_id}",
                    "song_beats": beats,
                    "section_times": section_times,
                    "audio_start_time": audio_start_time,
                    "updated_prompt": " ".join(match['keywords']),
                    "aspect_ratio": [16, 9],
                    "resolution": 1080,
                    "fps": "30",
                    "audio": str(audio_file) if audio_file else "",
                    "srt": str(lyrics_file) if lyrics_file else ""
                }
            }

            # Create config.Reel_ID and save template
            config.Reel_ID = f"Reel{config.rating_filter}_{cluster_id}"
            reel_path = os.path.join(base_path, config.Reel_ID)
            os.makedirs(reel_path, exist_ok=True)
            
            template_path = os.path.join(reel_path, 'video_template.json')
            with open(template_path, 'w') as f:
                json.dump(template, f, indent=4)
                
            print(f"Created template for cluster {cluster_id} with song {song_name}")
            print(f"Found {len(beats)} beats and {len(section_times)} sections")
            
        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {str(e)}")
            continue

    return True


def find_spotify_file(base_name, file_type):
    """
    Find a file recursively in the spotify directory structure with fuzzy matching
    
    Args:
        base_name (str): Base name of the file (without extension)
        file_type (str): Type of file to find ('beats', 'analysis', 'audio', 'lyrics')
    
    Returns:
        Path or None: Path to the found file, or None if not found
    """
    spotify_base = Path("Assets/spotify")
    
    def normalize_for_comparison(name):
        # Remove all special chars and spaces, convert to lowercase
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()
    
    normalized_base = normalize_for_comparison(base_name)
    
    # Define file patterns based on type
    patterns = {
        'beats': "*.json",
        'analysis': "*.json",
        'audio': "*.mp3",
        'lyrics': "*.srt"
    }
    
    if file_type not in patterns:
        raise ValueError(f"Unknown file type: {file_type}")
        
    pattern = patterns[file_type]
    
    # For beats and analysis, look in respective subdirectories
    if file_type in ['beats', 'analysis']:
        for dir_path in spotify_base.rglob(file_type):
            if dir_path.is_dir():
                for file_path in dir_path.glob(pattern):
                    # Remove _bars or _analysis from filename before comparison
                    clean_name = file_path.stem.replace("_bars", "").replace("_analysis", "")
                    if normalize_for_comparison(clean_name) == normalized_base:
                        return file_path
    else:
        # For audio and lyrics, search everywhere
        for file_path in spotify_base.rglob(pattern):
            # For audio files, compare without extension
            clean_name = file_path.stem
            if normalize_for_comparison(clean_name) == normalized_base:
                return file_path
            
    # If no exact match found for audio, try finding closest match
    if file_type == 'audio':
        closest_match = None
        closest_similarity = 0
        
        for file_path in spotify_base.rglob("*.mp3"):
            clean_name = normalize_for_comparison(file_path.stem)
            normalized_base_no_spaces = normalized_base.replace(" ", "")
            
            # Check if one is substring of other
            if clean_name in normalized_base_no_spaces or normalized_base_no_spaces in clean_name:
                similarity = len(set(clean_name) & set(normalized_base_no_spaces)) / len(set(clean_name) | set(normalized_base_no_spaces))
                if similarity > closest_similarity:
                    closest_similarity = similarity
                    closest_match = file_path
        
        if closest_similarity > 0.8:  # Threshold for accepting a close match
            print(f"Found close match for '{base_name}': '{closest_match.name}'")
            return closest_match
            
    return None

def load_spotify_data():
    """Load all beats and analysis data from all directories"""
    spotify_base = Path("Assets/spotify")
    data = {
        'beats': {},
        'analysis': {},
        'audio_files': {},
        'lyrics_files': {}
    }
    
    # Find and load all beats files
    for beats_dir in spotify_base.rglob('beats'):
        if beats_dir.is_dir():
            for beat_file in beats_dir.glob('*_bars.json'):
                song_name = beat_file.stem.replace("_bars", "")
                try:
                    with open(beat_file, 'r') as f:
                        data['beats'][song_name] = [bar['time'] for bar in json.load(f)]
                except json.JSONDecodeError:
                    print(f"Error reading beats file for {song_name}")
                    data['beats'][song_name] = []
    
    # Find and load all analysis files
    for analysis_dir in spotify_base.rglob('analysis'):
        if analysis_dir.is_dir():
            for analysis_file in analysis_dir.glob('*_analysis.json'):
                song_name = analysis_file.stem.replace("_analysis", "")
                try:
                    with open(analysis_file, 'r') as f:
                        analysis = json.load(f)
                        sections = analysis.get('sections', [])
                        data['analysis'][song_name] = {
                            'start_time': round(sections[2]['start'], 2) if len(sections) > 2 else 0.0,
                            'sections': [round(section['start'], 2) for section in sections]
                        }
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"Error reading analysis file for {song_name}: {e}")
                    data['analysis'][song_name] = {'start_time': 0.0, 'sections': []}

    # Load from combined JSON files for missing songs
    combined_dir = Path("Assets/jsons/combined")
    if combined_dir.exists():
        # Find most recent combined song embeddings file
        combined_files = sorted(combined_dir.glob("combined_song_embeddings_*.json"), reverse=True)
        if combined_files:
            try:
                with open(combined_files[0], 'r') as f:
                    combined_data = json.load(f)

                # Add missing songs from combined file
                for song_name, song_data in combined_data.items():
                    if song_name not in data['analysis'] and 'analysis' in song_data:
                        sections = song_data['analysis'].get('sections', [])
                        if sections:
                            data['analysis'][song_name] = {
                                'start_time': round(sections[2]['start'], 2) if len(sections) > 2 else 0.0,
                                'sections': [round(section['start'], 2) for section in sections]
                            }
            except Exception as e:
                print(f"Error reading combined embeddings: {e}")

    return data



@log_execution
@handle_errors
@sync_with_drive(
    inputs=[
        'embeddings_all.pkl',
        'embeddings_average.pkl',
        'embeddings_good.pkl',
        'embeddings_very_good.pkl',
        'embeddings_best.pkl',
        'all_reel_clusters.json',
        'average_reel_clusters.json',
        'good_reel_clusters.json',
        'very_good_reel_clusters.json',
        'best_reel_clusters.json',
        'image_face_assignments.json',
        'video_face_assignments.json',
        'cluster_checkpoint.json',
        os.path.join('Media', 'filename_mapping.json')
    ],
    outputs=['cluster_checkpoint.json']
)
def select_and_run_pipeline():
    """
    Detect content type and select the appropriate pipeline
    """
    if not config.User_ID or not config.Chat_ID:
        raise ValueError("config.User_ID and config.Chat_ID must be set")
        
    print(f"=== Starting Pipeline Selection ===")
    print(f"User ID: {config.User_ID}")
    print(f"Chat ID: {config.Chat_ID}")
    print(f"Process type: {config.process_type}")
    print(f"Rating filter: {config.rating_filter}")
    
    # Determine clusters to process based on process_type
    if config.process_type == 'batch':
        base_path = os.path.join(config.User_ID, config.Chat_ID)
        cluster_file = RATING_TO_CLUSTERS.get(config.rating_filter or 'all', 'all_reel_clusters.json')
        
        try:
            with open(os.path.join(base_path, cluster_file), 'r') as f:
                cluster_data = json.load(f)
                
            # Create templates
            create_cluster_templates()
            
            # Analyze each cluster for content type
            for cluster_no in cluster_data['clusters'].keys():
                # Store original Reel_ID
                original_Reel_ID = config.Reel_ID if hasattr(config, 'Reel_ID') else None
                config.Reel_ID = f"Reel{config.rating_filter}_{cluster_no}"
                
                try:
                    # Check content type
                    content_type = check_media_content_type(config.User_ID, config.Chat_ID, cluster_no)
                    print(f"\nCluster {cluster_no} contains: {content_type}")
                    
                    # Run appropriate pipeline
                    if content_type == 'images_only':
                        run_images_pipeline()
                    else:
                        run_videos_pipeline()
                        
                except Exception as e:
                    print(f"Error processing cluster {cluster_no}: {str(e)}")
                    
                finally:
                    # Restore original Reel_ID
                    config.Reel_ID = original_Reel_ID
                    
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            
    else:  # User processing
        try:
            # Run initialization steps
            initialize_template()
            get_embedding_search_results()
            
            # Check content type
            content_type = check_media_content_type(config.User_ID, config.Chat_ID)
            print(f"User request contains: {content_type}")
            
            # Run appropriate pipeline
            if content_type == 'images_only':
                run_images_pipeline()
            else:
                run_videos_pipeline()
                
        except Exception as e:
            print(f"Error in user processing: {str(e)}")
            
    print(f"=== Pipeline Selection Completed ===")
    return True

def check_media_content_type(User_ID, Chat_ID, cluster_no=None):
    """
    Determine if content contains only images or mixed media
    
    Args:
        User_ID (str): User ID
        Chat_ID (str): Chat ID
        cluster_no (str, optional): Cluster number for batch processing
        
    Returns:
        str: Either 'images_only' or 'mixed'
    """
    base_path = os.path.join(User_ID, Chat_ID)
    image_dir = os.path.join(base_path, 'Media', 'Images')
    video_dir = os.path.join(base_path, 'Media', 'Videos')
    
    def is_video_file(file_path):
        """Check if file is a video based on existence in video directory"""
        # Extract file ID (first part before any dash or dot)
        file_id = file_path.split('-')[0].split('.')[0]
        
        # Check if this file exists in the video directory
        for file in os.listdir(video_dir):
            if file.startswith(file_id):
                return True
                
        # If not found in video directory, it's not a video
        return False
    
    # For batch processing
    if cluster_no is not None:
        cluster_file = RATING_TO_CLUSTERS.get(config.rating_filter or 'all', 'all_reel_clusters.json')
        
        try:
            with open(os.path.join(base_path, cluster_file), 'r') as f:
                cluster_data = json.load(f)
                
            if str(cluster_no) in cluster_data['clusters']:
                # Check each file in the cluster
                for img in cluster_data['clusters'][str(cluster_no)]['images']:
                    if is_video_file(img['file_path']):
                        return 'mixed'
                
                return 'images_only'
            else:
                print(f"Warning: Cluster {cluster_no} not found in {cluster_file}")
                return 'mixed'  # Default to mixed if cluster not found
                
        except Exception as e:
            print(f"Error checking media content type for cluster {cluster_no}: {str(e)}")
            return 'mixed'  # Default to mixed on error
    
    # For user processing - check search_results.json if available
    else:
        reel_path = os.path.join(base_path, config.Reel_ID)
        search_results_path = os.path.join(reel_path, 'search_results.json')
        
        if os.path.exists(search_results_path):
            try:
                with open(search_results_path, 'r') as f:
                    search_results = json.load(f)
                
                # Check each file in the search results
                for item in search_results:
                    if is_video_file(item['file_path']):
                        return 'mixed'
                
                return 'images_only'
            
            except Exception as e:
                print(f"Error checking media content type for user request: {str(e)}")
                return 'mixed'  # Default to mixed on error
        else:
            print(f"Warning: search_results.json not found, cannot determine content type")
            return 'mixed'  # Default to mixed if search results not found

@log_execution
@handle_errors
def run_images_pipeline():
    """
    Execute the image-only pipeline for the current config.Reel_ID
    """
    # Get checkpoint info
    def get_checkpoint_status():
        cluster_no = 'user'
        if config.process_type == 'batch':
            # Extract cluster number from Reel_ID
            parts = config.Reel_ID.split('_')
            if len(parts) > 1:
                cluster_no = parts[-1]
                
        checkpoint_file = os.path.join(config.User_ID, config.Chat_ID, 'checkpoints.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoints = json.load(f)
                filter_key = f"{config.rating_filter or 'all'}_{cluster_no}_images"
                return checkpoints.get(filter_key, {})
        return {}

    def update_checkpoint(step):
        cluster_no = 'user'
        if config.process_type == 'batch':
            # Extract cluster number from Reel_ID
            parts = config.Reel_ID.split('_')
            if len(parts) > 1:
                cluster_no = parts[-1]
                
        checkpoint_file = os.path.join(config.User_ID, config.Chat_ID, 'checkpoints.json')
        checkpoints = {}
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoints = json.load(f)
        
        filter_key = f"{config.rating_filter or 'all'}_{cluster_no}_images"
        if filter_key not in checkpoints:
            checkpoints[filter_key] = {}
        checkpoints[filter_key][step] = True
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoints, f)
    
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    print(f"Running image-only pipeline for {config.Reel_ID}")
    
    checkpoints = get_checkpoint_status()
    
    # Define steps for image pipeline
    if config.process_type == 'batch':
        # For batch processing
        steps = [
            ('embedding', get_embedding_results_proxy),
            ('prompts', generate_two_prompts),
            ('apply_lut', apply_lut_with_mapping),
            ('cluster_story', cluster_story_pipeline),
            ('upload_to_drive', upload_images_to_drive),
            ('create_instagram_draft', create_insta_images),
            ('cleanup_folder', delete_folder)
        ]
    else:
        # For user processing
        steps = [
            ('initialize', initialize_template),
            ('embedding', get_embedding_search_results),
            ('prompts', generate_two_prompts),
            ('cluster_story', cluster_story_pipeline),
            ('apply_filter', instagram_filter.apply_filter),
            ('upload_to_drive', upload_images_to_drive),
            ('create_instagram_draft', create_insta_images),
            ('cleanup_folder', delete_folder)
        ]
    
    # Execute each step
    for step_name, step_func in steps:
        if step_name not in checkpoints:
            print(f"Running {step_name}")
            try:
                step_func()
                update_checkpoint(step_name)
            except Exception as e:
                print(f"Error in {step_name}: {str(e)}")
                raise
        else:
            print(f"Skipping completed step {step_name}")
    
    print(f"Completed image-only pipeline for {config.Reel_ID}")
    return True


@log_execution
@handle_errors
def run_videos_pipeline():
    """
    Execute the full video pipeline for the current config.Reel_ID
    """
    # Get checkpoint info
    def get_checkpoint_status():
        cluster_no = 'user'
        if config.process_type == 'batch':
            # Extract cluster number from Reel_ID
            parts = config.Reel_ID.split('_')
            if len(parts) > 1:
                cluster_no = parts[-1]
                
        checkpoint_file = os.path.join(config.User_ID, config.Chat_ID, 'checkpoints.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoints = json.load(f)
                filter_key = f"{config.rating_filter or 'all'}_{cluster_no}_videos"
                return checkpoints.get(filter_key, {})
        return {}

    def update_checkpoint(step):
        cluster_no = 'user'
        if config.process_type == 'batch':
            # Extract cluster number from Reel_ID
            parts = config.Reel_ID.split('_')
            if len(parts) > 1:
                cluster_no = parts[-1]
                
        checkpoint_file = os.path.join(config.User_ID, config.Chat_ID, 'checkpoints.json')
        checkpoints = {}
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoints = json.load(f)
        
        filter_key = f"{config.rating_filter or 'all'}_{cluster_no}_videos"
        if filter_key not in checkpoints:
            checkpoints[filter_key] = {}
        checkpoints[filter_key][step] = True
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoints, f)
    
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    print(f"Running video pipeline for {config.Reel_ID}")
    
    checkpoints = get_checkpoint_status()
    
    # Define steps for video pipeline
    if config.process_type == 'batch':
        # For batch processing
        steps = [
            ('embedding', get_embedding_results_proxy),
            ('prompts', generate_two_prompts),
            ('elements', fill_elements_gpt_o_two),
            ('video_elements', create_video_elements),
            ('transitions', apply_transition_system),
            ('template_frames', create_frames_from_template_image),
            ('video_frames', create_frames_from_template_video),
            ('combine_frames', combine_frames_to_video),
            ('final_audio', add_audio_to_video_and_cleanup),
            ('create_insta_video', create_insta_video)

        ]
    else:
        # For user processing
        steps = [
            ('initialize', initialize_template),
            ('embedding', get_embedding_search_results),
            ('prompts', generate_two_prompts),
            ('elements', fill_elements_gpt_o_two),
            ('video_elements', create_video_elements),
            ('transitions', apply_transition_system),
            ('template_frames', create_frames_from_template_image),
            ('video_frames', create_frames_from_template_video),
            ('combine_frames', combine_frames_to_video),
            ('final_audio', add_audio_to_video_and_cleanup),
            ('create_insta_video', create_insta_video)
        ]
    
    # Execute each step
    for step_name, step_func in steps:
        if step_name not in checkpoints:
            print(f"Running {step_name}")
            try:
                step_func()
                update_checkpoint(step_name)
            except Exception as e:
                print(f"Error in {step_name}: {str(e)}")
                raise
        else:
            print(f"Skipping completed step {step_name}")
    
    print(f"Completed video pipeline for {config.Reel_ID}")
    return True


@log_execution
@handle_errors
def get_search_results_standalone():
    """
    Standalone search function that works independently of video template.
    Uses config.user_prompt directly and returns fixed number of results.
    """
    base_path = os.path.join(config.User_ID, config.Chat_ID)

    # Define mapping for different process filters to their respective pkl files
    RATING_TO_PKL = {
        'all': 'embeddings_all.pkl',
        'average': 'embeddings_average.pkl',
        'good': 'embeddings_good.pkl',
        'very_good': 'embeddings_very_good.pkl',
        'best': 'embeddings_best.pkl'
    }
    
    # Get the correct pkl file based on process filter
    pkl_file = RATING_TO_PKL.get(config.rating_filter or 'all', 'embeddings_all.pkl')
    preprocessed_data_path = os.path.join(base_path, pkl_file)
    
    images_folder = os.path.join(base_path, 'Media', 'Images')

    # Use config.user_prompt directly and set fixed result limit
    query = config.user_prompt
    top_k = 20  # Fixed limit for search results
    
    # Load preprocessed data
    with open(preprocessed_data_path, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    model = preprocessed_data['model']
    index = preprocessed_data['index']
    file_paths = preprocessed_data['file_paths']
    embedding_data = preprocessed_data['embedding_data']
    
    # Generate query embedding
    query_embedding = model.encode(query)
    
    # Perform initial search to get top 100 candidates
    initial_top_k = 100
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), initial_top_k)
    
    # Prepare candidates for cross-encoder
    candidates = []
    for i, idx in enumerate(indices[0]):
        file_path = file_paths[idx]
        candidates.append({
            'file_path': file_path,
            'initial_similarity': 1 - distances[0][i],
            'description': embedding_data[file_path]['description'],
            'faces': embedding_data[file_path]['faces']
        })
    
    import torch
    
    device = 'cpu'  # Force CPU usage to avoid MPS device issues
    # Initialize cross-encoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
    
    # Prepare input pairs for cross-encoder
    input_pairs = [[query, f"This image contains the following people: {' '.join(candidate['faces'])} and Description: {candidate['description']}"] for candidate in candidates]
    
    # Compute cross-encoder scores
    cross_encoder_scores = cross_encoder.predict(input_pairs)
    
    # Combine candidates with cross-encoder scores and sort
    results = sorted(zip(candidates, cross_encoder_scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Prepare final results
    search_results = []
    for candidate, score in results:
        search_results.append({
            'file_path': candidate['file_path'],
            'similarity': float(score),  # Use cross-encoder score as final similarity
            'description': candidate['description'],
            'faces': candidate['faces']
        })
    
    # Save the results in the reel directory
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    os.makedirs(reel_path, exist_ok=True)  # Ensure directory exists
    results_path = os.path.join(reel_path, 'search_results.json')
    with open(results_path, 'w') as f:
        json.dump(search_results, f, indent=4)

    print(f"Saved {len(search_results)} search results to {results_path}")


@log_execution
@handle_errors
def assemble_video_template():
    """
    Create complete video template by assembling components from song.json and sync.json.
    This replaces both template initialization and create_video_elements functionality.
    """
    # Fixed paths
    base_path = os.path.join(config.User_ID, config.Chat_ID)
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    
    # Helper functions copied from create_video_elements
    def find_file(directory, truncated_filename):
        truncated_name = truncated_filename.split('.')[0]  # Remove extension if present
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith(truncated_name):
                    return os.path.join(root, file)
        return None

    def find_closest_file(directory, truncated_filename):
        from difflib import SequenceMatcher
        truncated_name = truncated_filename.split('.')[0]
        highest_similarity = 0
        closest_file = None
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_name = file.split('.')[0]
                similarity = SequenceMatcher(None, truncated_name, file_name).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    closest_file = os.path.join(root, file)
        
        if closest_file:
            logging.info(f"Using closest matching file {os.path.basename(closest_file)} instead of {truncated_filename}")
        return closest_file

    # Load song.json
    song_json_path = os.path.join(reel_path, 'song.json')
    try:
        with open(song_json_path, 'r', encoding='utf-8') as f:
            song_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Error loading song.json: {str(e)}")
        return

    # Load sync.json  
    sync_json_path = os.path.join(reel_path, 'sync.json')
    try:
        with open(sync_json_path, 'r', encoding='utf-8') as f:
            sync_file_data = json.load(f)
        sync_sequence = sync_file_data.get("sync_sequence", [])
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Error loading sync.json: {str(e)}")
        return
    
    if not sync_sequence:
        print("No sync sequence found in sync.json")
        return

    # Create audio path from song name
    # song_name = song_data.get("song", "")
    # audio_path = f"Assets/spotify/audio/{song_name}.mp3"

    # Create audio path from song name (remove .mp3 if already present)
    song_name = song_data.get("song", "")
    if song_name.endswith(".mp3"):
        song_name = song_name[:-4]  # Remove .mp3 extension
    audio_path = f"Assets/spotify/audio/{song_name}.mp3"
    
    # Create template structure with audio_settings from song.json
    template = {
        "template": {
            "aspect_ratio": [9, 16],
            "resolution": 1080,
            "fps": "30",
            "audio": audio_path,
            "audio_settings": {
                "volume": song_data.get("volume", 1.0),
                "speed": song_data.get("speed", 1.0),
                "pitch": song_data.get("pitch", 0),
                "start_time": song_data.get("start_time", 0),
                "end_time": song_data.get("end_time", 30)
            },
            "elements": []
        }
    }

    fps = 30
    elements = []

    # Initialize cumulative timeline tracker for NEW format
    cumulative_time = 0.0

    # Process each item in sync sequence (logic from create_video_elements)
    for index, item in enumerate(sync_sequence):
        # Calculate frame_start and frame_end
        if 'offset' in item:
            # NEW FORMAT - offset/duration exists
            offset = item['offset']
            duration = item['duration']
            frame_start = int(cumulative_time * fps)
            frame_end = int((cumulative_time + duration) * fps) - 1
            cumulative_time += duration
        elif 'start_time' in item:
            # LEGACY FORMAT - start_time/end_time (for backward compatibility)
            duration = item['end_time'] - item['start_time']
            offset = item.get('start_time', 0.0)  # Use start_time as offset for legacy
            frame_start = int(cumulative_time * fps)
            frame_end = int((cumulative_time + duration) * fps) - 1
            cumulative_time += duration
        else:
            # OLD FORMAT - end_time is cumulative position
            offset = 0.0
            if index == 0:
                frame_start = 0
            else:
                frame_start = int(sync_sequence[index - 1]['end_time'] * fps)
            frame_end = int(item['end_time'] * fps) - 1  # Subtract 1 to avoid overlap
            duration = (frame_end - frame_start + 1) / fps
        
        # Search for the file without extension
        image_dir = os.path.join(config.User_ID, config.Chat_ID, 'Media', 'Images')
        video_dir = os.path.join(config.User_ID, config.Chat_ID, 'Media', 'Videos')
        
        image_path = find_file(image_dir, item['file_path'])
        video_path = find_file(video_dir, item['file_path'])

        # Handle missing files with closest match
        if image_path is None and video_path is None:
            image_path = find_closest_file(image_dir, item['file_path'])
            if image_path is not None:
                logging.info(f"File not found: {item['file_path']}, replaced with: {os.path.basename(image_path)}")
            if image_path is None:  # If no close image match
                video_path = find_closest_file(video_dir, item['file_path'])
                if video_path is not None:
                    logging.info(f"File not found: {item['file_path']}, replaced with: {os.path.basename(video_path)}")

        # Determine file type based on extension and actual file found
        image_extensions = ['.jpg', '.jpeg', '.heic', '.heif']
        has_image_extension = any(item['file_path'].lower().endswith(ext) for ext in image_extensions)
        has_no_extension = '.' not in item['file_path']
        
        # Final determination: actual file found takes precedence
        is_video = video_path is not None or (has_no_extension and image_path is None)
        
        if is_video:
            full_filename = os.path.basename(video_path) if video_path else item['file_path']

            element = {
                "element_type": "video",
                "description": "Main video clip with effects",
                "frame_start": frame_start,
                "frame_end": frame_end,
                "media_path": full_filename,
                "clip_start_time": offset,  # Seconds to skip from source start
                "clip_end_time": offset + duration,  # End position in source
                "clip_start_frame": 0,
                "clip_end_frame": 1000,
                "layer_order": 2,
                "opacity": 1.0,
                "rotation": 0,
                "effects": "blur",
                "transitions": [],
                "audio_settings": {
                    "volume": 1.0,
                    "mute": False
                }
            }
        else:
            full_filename = os.path.basename(image_path) if image_path else item['file_path']

            element = {
                "element_type": "image",
                "description": "Overlay logo with effects",
                "frame_start": frame_start,
                "frame_end": frame_end,
                "media_path": full_filename,
                "layer_order": 2,
                "opacity": 1,
                "rotation": 0,
                "transitions": [],
                "effects": "glow"
            }
        
        elements.append(element)
    
    # Add the elements to the template
    template['template']['elements'] = elements
    
    # Save the video template
    template_path = os.path.join(reel_path, 'video_template.json')
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Video template with {len(elements)} elements created at {template_path}")


@log_execution
@handle_errors
def validate_and_adjust_template_timing():
    """
    Validates video durations and adjusts template timing to prevent stuck frames.
    When videos are shorter than intended, extends subsequent elements to maintain beat sync.

    Logic:
    - If video ends early, let it end naturally (no stuck frames)
    - Next element starts immediately after early-ending video
    - Next element extends to reach original end time (maintains beat sync)
    - If next element is also short video, process iteratively
    """
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    template_path = os.path.join(reel_path, 'video_template.json')
    
    # Load template
    with open(template_path, 'r') as f:
        template_data = json.load(f)
    
    elements = template_data['template']['elements']
    fps = int(template_data['template']['fps'])
    
    # def get_video_duration_frames(media_path):
    #     """Get actual video duration in frames using OpenCV"""
    #     video_dir = os.path.join(config.User_ID, config.Chat_ID, 'Media', 'Videos')
    #     video_path = os.path.join(video_dir, media_path)
        
    #     if not os.path.exists(video_path):
    #         print(f"Warning: Video file not found: {video_path}")
    #         return 0
            
    #     try:
    #         cap = cv2.VideoCapture(video_path)
    #         if not cap.isOpened():
    #             print(f"Warning: Could not open video: {video_path}")
    #             return 0
                
    #         video_fps = cap.get(cv2.CAP_PROP_FPS)
    #         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         cap.release()
            
    #         if video_fps <= 0:
    #             print(f"Warning: Invalid FPS for video: {video_path}")
    #             return 0
                
    #         # Convert to our template fps
    #         actual_duration_seconds = frame_count / video_fps
    #         return int(actual_duration_seconds * fps)
            
    #     except Exception as e:
    #         print(f"Error reading video {video_path}: {str(e)}")
    #         return 0
    
    def get_video_duration_frames(media_path):
        """Get actual video duration in frames using HelperFunctions"""
        video_dir = os.path.join(config.User_ID, config.Chat_ID, 'Media', 'Videos')
        video_path = os.path.join(video_dir, media_path)
        
        try:
            cap = HelperFunctions.load_video(video_path)
            if cap is None:
                print(f"Warning: Could not load video: {video_path}")
                return 0
                
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if video_fps <= 0:
                print(f"Warning: Invalid FPS for video: {video_path}")
                return 0
                
            # Convert to our template fps
            actual_duration_seconds = frame_count / video_fps
            return int(actual_duration_seconds * fps)
            
        except Exception as e:
            print(f"Error reading video {video_path}: {str(e)}")
            return 0
    
    # Track accumulated timing gap that needs to be absorbed
    accumulated_gap_frames = 0
    adjustments_made = 0
    
    for i in range(len(elements)):
        element = elements[i]
        
        # Apply any accumulated gap adjustment
        if accumulated_gap_frames > 0:
            # Move element start time back by accumulated gap
            element['frame_start'] -= accumulated_gap_frames
            # Keep original end time - this extends the element duration
            
            if element['element_type'] == 'image':
                # Images can always absorb any gap
                print(f"Image element {i} extended by {accumulated_gap_frames/fps:.2f}s to absorb gap")
                accumulated_gap_frames = 0
                adjustments_made += 1
                
            elif element['element_type'] == 'video':
                # Check if video can handle the extension
                intended_duration_frames = element['frame_end'] - element['frame_start']
                actual_duration_frames = get_video_duration_frames(element['media_path'])
                
                required_duration_frames = intended_duration_frames + accumulated_gap_frames
                
                if actual_duration_frames >= required_duration_frames:
                    # Video is long enough to absorb the gap
                    print(f"Video element {i} extended by {accumulated_gap_frames/fps:.2f}s to absorb gap")
                    accumulated_gap_frames = 0
                    adjustments_made += 1
                else:
                    # Video can only partially absorb gap
                    max_absorption = max(0, actual_duration_frames - intended_duration_frames)
                    accumulated_gap_frames -= max_absorption
                    
                    if max_absorption > 0:
                        print(f"Video element {i} partially absorbed {max_absorption/fps:.2f}s gap")
                        adjustments_made += 1
        
        # Check if current video element ends early
        if element['element_type'] == 'video':
            intended_duration_frames = element['frame_end'] - element['frame_start']
            actual_duration_frames = get_video_duration_frames(element['media_path'])
            
            if actual_duration_frames > 0 and actual_duration_frames < intended_duration_frames:
                # Video ends early - calculate new gap
                gap_frames = intended_duration_frames - actual_duration_frames
                accumulated_gap_frames += gap_frames
                
                # Adjust current video to end when it actually ends
                element['frame_end'] = element['frame_start'] + actual_duration_frames
                
                print(f"Video element {i} ends {gap_frames/fps:.2f}s early - gap added to queue")
                adjustments_made += 1
    
    # Handle any remaining gap at the end
    if accumulated_gap_frames > 0:
        print(f"Warning: {accumulated_gap_frames/fps:.2f}s gap remaining at end of sequence")
    
    # Save adjusted template
    with open(template_path, 'w') as f:
        json.dump(template_data, f, indent=2)
    
    if adjustments_made > 0:
        print(f"Template timing validated - {adjustments_made} adjustments made")
    else:
        print("Template timing validated - no adjustments needed")


@log_execution
@handle_errors
def apply_transition_system():
    """
    Apply transition system using transition.json if available, otherwise use default transitions
    """
    reel_path = os.path.join(config.User_ID, config.Chat_ID, config.Reel_ID)
    transition_json_path = os.path.join(reel_path, 'transition.json')
    sync_json_path = os.path.join(reel_path, 'sync.json') 
    # We integrate transitions into template-based frame rendering instead of
    # producing a separate transitions-only video here.
    
    try:
        # Load template to inject transitions (single path)
        template_path = os.path.join(reel_path, 'video_template.json')
        with open(template_path, 'r') as f:
            template = json.load(f)

        elements = template.get('template', {}).get('elements', [])
        if not elements:
            print("❌ No elements found in template to apply transitions")
            return False

        # Optional inputs (use when present)
        sync_sequence = []
        transition_sequence = []
        if os.path.exists(sync_json_path):
            try:
                with open(sync_json_path, 'r') as f:
                    sync_data = json.load(f)
                sync_sequence = sync_data.get('sync_sequence', [])
            except Exception:
                sync_sequence = []

        if os.path.exists(transition_json_path):
            try:
                with open(transition_json_path, 'r') as f:
                    transition_data = json.load(f)
                transition_sequence = transition_data.get('transition_sequence', [])
            except Exception:
                transition_sequence = []

        # Use default fade types if transition data is missing
        has_custom_transitions = len(transition_sequence) > 0

        # Transition duration consistent with default system: 1/6 second
        fps = int(template['template'].get('fps', '30'))
        transition_duration = max(1, int(fps / 6))

        # Apply entry/exit transitions to each element in a single unified path
        n = len(elements)
        for i in range(n):
            elem = elements[i]
            frame_start = elem.get('frame_start', 0)
            frame_end = elem.get('frame_end', frame_start)

            # Determine types
            if has_custom_transitions and i < len(transition_sequence):
                # Entry ALWAYS fade_in (consistent, clean)
                entry_type = 'fade_in'
                # Exit uses current element's transition from JSON
                exit_type = transition_sequence[i]['transition_type']
            else:
                entry_type = 'fade_in'
                exit_type = 'fade_out'

            # Guard frame ranges
            entry_end = min(frame_end, frame_start + transition_duration)
            exit_start = max(frame_start, frame_end - transition_duration)

            elem['transitions'] = [
                {
                    'type': entry_type,
                    'frame_start': frame_start,
                    'frame_end': entry_end
                },
                {
                    'type': exit_type,
                    'frame_start': exit_start,
                    'frame_end': frame_end
                }
            ]

        # Persist updated template
        template['template']['elements'] = elements
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)

        print(f"✅ Applied transitions to {n} elements (custom: {has_custom_transitions})")
        return True

    except Exception as e:
        print(f"❌ Error applying transitions to template: {str(e)}")
        return False