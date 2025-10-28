import os
import json
import requests
from typing import Dict, Optional, Tuple
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tqdm.auto import tqdm
from decorators import *
import librosa
import requests
from bs4 import BeautifulSoup

import re
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List


def get_lyrics_from_genius(url):
    """Extract lyrics from Genius webpage."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all divs whose class starts with 'Lyrics__Container-'
        # lyrics_containers = soup.find_all('div', class_=lambda x: x and x.startswith('Lyrics__Container-'))
        lyrics_containers = soup.find_all('div', {'data-lyrics-container': 'true'})

        
        if not lyrics_containers:
            return None
            
        full_lyrics = []
        for container in lyrics_containers:
            # Get text and preserve line breaks
            for br in container.find_all('br'):
                br.replace_with('\n')
            lyrics = container.get_text()
            full_lyrics.append(lyrics)
        
        # Join all parts and clean up
        lyrics_text = '\n'.join(full_lyrics)
        # Remove extra whitespace while preserving line breaks
        lyrics_text = '\n'.join(line.strip() for line in lyrics_text.split('\n'))
        # Remove multiple consecutive newlines
        lyrics_text = '\n'.join(filter(None, lyrics_text.split('\n')))
        
        return lyrics_text
        
    except Exception as e:
        print(f"Error extracting lyrics: {str(e)}")
        return None

def format_timestamp(seconds):
    """Convert seconds to MM:SS.ms format"""
    return f"{int(seconds//60):02d}:{seconds%60:06.3f}"

# Use the function
# url = "https://genius.com/Hanumankind-and-kalmi-big-dawgs-lyrics"
# lyrics = get_lyrics_from_genius(url)


class SpotifyHandler:
    def __init__(self):
        """
        Initialize the SpotifyHandler with your Spotify API credentials.
        """
        # Try to find .env file in current directory first
        current_dir = Path.cwd()
        env_path = current_dir / '.env'
        
        # If not in current directory, try one level up
        if not env_path.exists():
            env_path = current_dir.parent / '.env'
        
        # If still not found, raise an error
        if not env_path.exists():
            raise FileNotFoundError(
                "Could not find .env file. Please create one in the current "
                "directory or parent directory with your Spotify credentials."
            )
        
        # Load environment variables from the specified path
        load_dotenv(dotenv_path=env_path)
        
        # Check if credentials are loaded
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            raise ValueError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in your .env file. "
                "Format should be:\n"
                "SPOTIFY_CLIENT_ID=your_client_id_here\n"
                "SPOTIFY_CLIENT_SECRET=your_client_secret_here"
            )
        
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))
        
        # Create necessary directories
        self.base_path = Path("Assets/spotify")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # # Create subdirectories
        # self.audio_path = self.base_path / "audio"
        # self.analysis_path = self.base_path / "analysis"
        # self.lyrics_path = self.base_path / "lyrics"
        # self.beats_path = self.base_path / "beats"
        # self.beats_path.mkdir(exist_ok=True)
        
        # for path in [self.audio_path, self.analysis_path, self.lyrics_path]:
        #     path.mkdir(exist_ok=True)

    def search_track(self, track_name: str) -> Optional[Dict]:
        """
        Search for a track and return its information.
        
        Args:
            track_name (str): Name of the track to search for
            
        Returns:
            Dict: Track information or None if not found
        """
        results = self.sp.search(q=track_name, type='track', limit=1)
        
        if not results['tracks']['items']:
            return None
            
        return results['tracks']['items'][0]

    def get_track_info(self, track_id: str) -> Dict:
        """
        Get detailed track information.
        
        Args:
            track_id (str): Spotify track ID
            
        Returns:
            Dict: Detailed track information
        """
        return self.sp.track(track_id)

    def get_audio_analysis(self, track_id: str) -> Tuple[Dict, str]:
        """
        Get audio analysis for a track and save it to a file.
        
        Args:
            track_id (str): Spotify track ID
            
        Returns:
            Tuple[Dict, str]: Audio analysis data and the path where it was saved
        """
        # Get the analysis
        analysis = self.sp.audio_analysis(track_id)
        
        # Save to file
        file_path = self.analysis_path / f"{track_id}_analysis.json"
        with open(file_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis, str(file_path)

    def get_track_lyrics(self, track_id: str) -> Optional[str]:
        """
        Get track lyrics and save as SRT format.
        Note: This is a placeholder as Spotify API doesn't provide lyrics.
        You would need to integrate with a lyrics service like Genius or Musixmatch.
        
        Args:
            track_id (str): Spotify track ID
            
        Returns:
            Optional[str]: Path to the saved lyrics file
        """
        # This is where you'd implement lyrics fetching from a third-party service
        # For now, we'll create a placeholder file
        file_path = self.lyrics_path / f"{track_id}_lyrics.srt"
        with open(file_path, 'w') as f:
            f.write("# Lyrics would go here\n")
        
        return str(file_path)
    
    def prepare_for_visualization(self, track_name: str) -> Tuple[str, str]:
        """Prepare track data for visualization."""
        result = self.process_track(track_name)
        
        # Create consistent filenames - ensure lowercase with underscores
        safe_name = self.sanitize_filename(track_name)
        
        # Move analysis file to correct location with lowercase underscore format
        analysis_file = self.analysis_path / f"{safe_name}_analysis.json"
        beats_file = self.beats_path / f"{safe_name}_beats.json"
        audio_file = self.audio_path / f"{safe_name}.mp3"
        
        # Ensure the analysis is saved with correct name
        if not analysis_file.exists():
            with open(result['analysis_path'], 'r') as f:
                analysis_data = json.load(f)
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
        
        return str(analysis_file), str(audio_file)
    
    
    def sanitize_filename(self, filename: str) -> str:
        """Convert song name to a safe filename with lowercase and underscores"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '')
        # Convert to lowercase and replace spaces with underscores
        filename = filename.strip().lower().replace(" ", "_")
        return filename

    def generate_internal_beats(self, audio_path: str) -> Dict:
        """Generate beats data in Spotify format"""
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Enhanced onset detection
        hop_length = 512
        onset_env = librosa.onset.onset_strength(
            y=y, 
            sr=sr,
            hop_length=hop_length,
            aggregate=np.median
        )
        
        # Get tempo
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Adjust tempo to be in a reasonable range (80-180 BPM)
        if tempo < 80:
            tempo *= 2
        elif tempo > 180:
            tempo /= 2
            
        # Calculate time for one bar (4 beats)
        seconds_per_bar = 4 * 60.0 / tempo
        
        # Initialize output format matching reference
        output = []
        
        current_time = 0
        duration = librosa.get_duration(y=y, sr=sr)
        
        while current_time < duration:
            frame_idx = int(current_time * sr / hop_length)
            
            if frame_idx < len(onset_env):
                # Get local window of onset strength
                window = 4
                start_idx = max(0, frame_idx - window)
                end_idx = min(len(onset_env), frame_idx + window)
                local_onset = onset_env[start_idx:end_idx]
                
                if len(local_onset) > 0:
                    # Calculate confidence
                    confidence = np.mean(local_onset) / np.max(onset_env)
                    confidence = 0.3 + (0.7 * confidence)
                    confidence = min(max(confidence, 0.2), 0.95)
                else:
                    confidence = 0.5
                    
                # output['time'].append(float(current_time))
                # output['confidence'].append(float(confidence))
                # output['duration'].append(float(seconds_per_bar))

                output.append({
                    'time': float(current_time),
                    'confidence': float(confidence),
                    'duration': float(seconds_per_bar)
                })
            
            current_time += seconds_per_bar
        
        return output

    def generate_internal_sections(self, audio_path: str) -> List[Dict]:
        """Generate sections data in Spotify format"""
        y, sr = librosa.load(audio_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Compute mel spectrogram with explicit parameters
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=128, 
            hop_length=512
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Compute novelty curve from mel spectrogram
        novelty = librosa.onset.onset_strength(
            S=mel_db, 
            sr=sr,
            hop_length=512,
            aggregate=np.mean
        )
        
        # Find section boundaries
        kernel_size = 500
        peaks, properties = find_peaks(
            novelty,
            distance=kernel_size,
            prominence=0.15,
            height=None
        )
        
        # Convert frame indices to times
        section_times = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
        
        # Add start time if not present
        if len(section_times) == 0 or section_times[0] > 0:
            section_times = np.insert(section_times, 0, 0)
            peaks = np.insert(peaks, 0, 0)
            properties = {'prominences': np.insert(properties.get('prominences', []), 0, 1.0)}
        
        # Format sections to match reference format
        formatted_sections = []
        for i in range(len(section_times)):
            start_time = section_times[i]
            end_time = duration if i == len(section_times) - 1 else section_times[i + 1]
            
            # Get section audio
            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)
            section_audio = y[start_frame:end_frame]
            
            if len(section_audio) > sr * 5:  # Only process sections > 5 seconds
                tempo = librosa.feature.rhythm.tempo(y=section_audio, sr=sr)[0]
                
                # Calculate confidence based on peak prominence
                if i == 0:
                    confidence = 0.8
                else:
                    prominence = properties['prominences'][i-1]
                    confidence = min(max(prominence / np.max(properties['prominences']), 0.2), 0.95)
                
                # Calculate loudness
                rms = librosa.feature.rms(y=section_audio)[0]
                loudness = float(np.mean(rms))
                
                formatted_sections.append({
                    "start": float(start_time),
                    "duration": float(end_time - start_time),
                    "confidence": float(confidence),
                    "loudness": float(loudness * -30),  # Scale to approximate Spotify range
                    "tempo": float(tempo),
                    "tempo_confidence": 0.5,
                    "key": 0,
                    "key_confidence": 0.5,
                    "mode": 1,
                    "mode_confidence": 0.5,
                    "time_signature": 4,
                    "time_signature_confidence": 1.0
                })
        
        return formatted_sections

        

    
    def get_track_lyrics(self, track_id: str, song_name: str, folder_name: str) -> Optional[str]:
        """Get track lyrics using Genius API and webpage scraping."""
        safe_name = self.sanitize_filename(song_name)
        lyrics_path = Path(f"Assets/spotify/{folder_name}/lyrics")
        lyrics_path.mkdir(exist_ok=True, parents=True)
        # file_path = self.lyrics_path / f"{safe_name}_lyrics.srt"
        file_path = lyrics_path / f"{safe_name}_lyrics.srt"  # Changed this line

        
        try:
            # Get track info to get artist name
            track_info = self.get_track_info(track_id)
            artist_name = track_info['artists'][0]['name']
            
            # Get Genius API token from env
            genius_token = os.getenv('GENIUS_ACCESS_TOKEN')
            if not genius_token:
                raise ValueError("GENIUS_ACCESS_TOKEN not found in .env file")
            
            # Setup headers for Genius API
            headers = {
                'Authorization': f'Bearer {genius_token}'
            }
            
            # Search for the song
            search_url = f'https://api.genius.com/search?q={song_name} {artist_name}'
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            
            search_results = response.json()
            
            if search_results['response']['hits']:
                # Get the first hit
                song_info = search_results['response']['hits'][0]['result']
                lyrics_url = song_info['url']
                
                # Get lyrics from webpage
                lyrics_text = get_lyrics_from_genius(lyrics_url)
                
                if lyrics_text:
                    # Save as SRT with timestamps
                    lines = lyrics_text.split('\n')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for i, line in enumerate(lines, 1):
                            if line.strip():
                                start_time = (i - 1) * 3  # 3 seconds per line
                                end_time = i * 3
                                
                                f.write(f"{i}\n")
                                f.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                                f.write(f"{line}\n\n")
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"# Could not extract lyrics for {song_name} by {artist_name}\n")
            else:
                with open(file_path, 'w') as f:
                    f.write(f"# Song not found on Genius: {song_name} by {artist_name}\n")
        
        except Exception as e:
            print(f"Error fetching lyrics: {str(e)}")
            with open(file_path, 'w') as f:
                f.write(f"# Error fetching lyrics: {str(e)}\n")
        
        return str(file_path)

    def process_track(self, track_name: str) -> Dict:
        """Process a track: search, get info, analysis, and lyrics."""
        track_info = self.search_track(track_name)
        if not track_info:
            raise ValueError(f"Track '{track_name}' not found")
        
        track_id = track_info['id']
        
        # Get detailed track information
        detailed_info = self.get_track_info(track_id)
        
        # Get audio analysis with song name
        analysis, analysis_path = self.get_audio_analysis(track_id, track_name)
        
        # Get lyrics with song name
        lyrics_path = self.get_track_lyrics(track_id, track_name)
        
        return {
            'track_info': detailed_info,
            'audio_analysis': analysis,
            'analysis_path': analysis_path,
            'lyrics_path': lyrics_path
        }

def delete_song_files(song_name: str) -> bool:
    """
    Delete all files associated with a given song.
    
    Args:
        song_name (str): Name of the song to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        spotify = initialize_spotify()
        base_path = Path("Assets/spotify")
        sanitized_name = spotify.sanitize_filename(song_name)
        
        # Define all possible file paths
        files_to_delete = [
            base_path / "audio" / f"{sanitized_name}.mp3",
            base_path / "lyrics" / f"{sanitized_name}_lyrics.srt",
            base_path / "beats" / f"{sanitized_name}_bars.json",
            base_path / "analysis" / f"{sanitized_name}_analysis.json",
            base_path / "song_info" / f"{sanitized_name}_info.txt"
        ]
        
        deleted_files = []
        for file_path in files_to_delete:
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_files.append(file_path.name)
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
        
        if deleted_files:
            print(f"Successfully deleted the following files for {song_name}:")
            for file in deleted_files:
                print(f"- {file}")
            return True
        else:
            print(f"No files found for song: {song_name}")
            return False
            
    except Exception as e:
        print(f"Error in delete_song_files: {str(e)}")
        return False

from duckduckgo_search import DDGS
from natwar import GPT_web_scraper, GPT_search_summarize
from pathlib import Path
import time
# from time import time



def research_song_info(song_name, folder_name, search_engine='brave'):
    """Research a song and save findings without using sheets"""
    try:
        save_dir = Path(f"Assets/spotify/{folder_name}/song_info")
        save_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = save_dir / f"{song_name.lower().replace(' ','_')}_info.txt"
        

        if file_path.exists():
            return  # Skip processing if file already exists
            
        # Get song metadata from Spotify
        spotify_handler = SpotifyHandler()
        track_info = spotify_handler.get_track_info(spotify_handler.search_track(song_name)['id'])
        
        # Form search query
        artists = [artist['name'] for artist in track_info['artists']]
        album = track_info['album']['name']
        search_term = f"{song_name} by {', '.join(artists)} from album {album}"

        query = (
            "Respond with a structured analysis of the song, organized into the following sections:\n\n"
            "1. Introduction: Provide a brief overview of the song’s general character and musical style.\n\n"
            "2. Emotional Landscape: Describe in detail the emotions, moods, and psychological atmospheres the song evokes. "
            "Explain how these feelings might shift throughout the track.\n\n"
            "3. Musical Elements: Analyze the instrumentation, melody, harmony, rhythm, and any notable vocal characteristics. "
            "Discuss how these elements work together to shape the overall tone and experience.\n\n"
            "4. Thematic Resonance: Explore any thematic concepts, cultural influences, or narrative threads that the song "
            "might represent or suggest. Consider the historical context, genre conventions, or production values.\n\n"
            "5. Visual Pairings: Recommend types of visual content or storytelling themes that would naturally complement "
            "the music. Consider various video formats—travel vlogs, cinematic montages, nature footage, documentary style, "
            "inspirational narratives, art films, personal journals—and explain why the song’s qualities suit these themes.\n\n"
            "6. Key Moments for Editing: Identify specific moments in the track (such as crescendos, breakdowns, solos, "
            "transitions, or dynamic shifts) that could align well with particular editing techniques or narrative beats "
            "in a video.\n\n"
            "7. Conclusion: Summarize the core attributes of the song and reinforce how its emotional and musical qualities "
            "lend themselves to a wide range of visual storytelling possibilities.\n\n"
            "Do not include any links or references."
        )

        # --- Step 7 and 8 modifications for Brave vs DuckDuckGo ---

        if search_engine.lower() == 'brave':
            current_dir = Path.cwd()
            env_path = current_dir / '.env'
            
            # If not in current directory, try one level up
            if not env_path.exists():
                env_path = current_dir.parent / '.env'
            
            # If still not found, raise an error
            if not env_path.exists():
                raise FileNotFoundError(
                    "Could not find .env file. Please create one in the current "
                    "directory or parent directory with your Spotify and Brave credentials."
                )
            
            # Load environment variables from the specified path
            load_dotenv(dotenv_path=env_path)
            brave_api_key = os.getenv('BRAVE_API_KEY')
            # Add these lines to call Brave API
            import requests
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": "BSAOwfDk3DgXTqzxqyAmPKZMu7IokWJ"
            }
            params = {
                "q": search_term,
                "count": 7,
                "safesearch": "off"
            }
            response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            raw_results = data.get("web", {}).get("results", [])
            formatted_results = []
            for result in raw_results:
                formatted_results.append({
                    'search_term': search_term,
                    'title': result.get('title', ''),
                    'url': result.get('url', '')
                })
        else:
            # Fallback to DuckDuckGo
            ddgs = DDGS()
            raw_results = list(ddgs.text(
                search_term,
                region='wt-wt',
                safesearch='off',
                max_results=7
            ))

        
            # Format results
            formatted_results = []
            for result in raw_results:
                formatted_results.append({
                    'search_term': search_term,
                    'title': result.get('title', ''),
                    'url': result.get('link', result.get('href', ''))
                })
        
        # Process URLs concurrently
        detailed_content = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for result in formatted_results:
                if result['url']:  # Make sure we have a valid URL
                    future = executor.submit(
                        process_single_url,
                        result['url'],
                        result['title'],
                        search_term,
                        query
                    )
                    futures.append(future)
            
            for future in futures:
                try:
                    process_result = future.result(timeout=15)
                    if process_result:
                        detailed_content.append(process_result['processed_content'])
                except TimeoutError:
                    print("Timeout processing URL")
                    continue
        
        # Only proceed if we have content
        if detailed_content:
            summary = GPT_search_summarize(
                search_term=search_term,
                context=query,
                detailed_text="\n\n".join(detailed_content)
            )
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Song: {song_name}\n")
                f.write(f"Artists: {', '.join(artists)}\n")
                f.write(f"Album: {album}\n\n") 
                f.write("Research Summary:\n")
                f.write(summary)
        else:
            print(f"No content could be processed for song: {song_name}")
            
    except Exception as e:
        print(f"Error in research_song_info: {str(e)}")



def process_single_url(url: str, title: str, search_term: str, query: str) -> Optional[Dict]:
    """Process a single URL and return its processed content"""
    try:
        # print(f"\nProcessing URL: {url}")
        response = requests.get(url)
        # print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)[:20000]
            # print(f"Extracted text length: {len(text)} characters")
            
            # Process text through GPT
            # print("Sending to GPT for processing...")
            processed_content = GPT_web_scraper(
                search_term=search_term,
                context=query,
                text=text
            )
            # print(f"GPT processed content length: {len(processed_content)} characters")
            
            return {
                'url': url,
                'title': title,
                'search_term': search_term,
                'processed_content': processed_content
            }
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return None
# Test
# research_song_info("Adore You")

def find_audio_file(self, base_name: str) -> Optional[Path]:
    """Find audio file matching the base name, ignoring case and special characters"""
    audio_files = list(self.audio_path.glob("*.mp3"))
    # Print files for debugging
    print(f"Looking for '{base_name}' among files:")
    for file in audio_files:
        print(f"  - {file.stem}")
        if file.stem.lower() == base_name.lower():
            return file
    return None


@log_execution
@handle_errors
def initialize_spotify():
    """Create SpotifyHandler instance if not exists"""
    global spotify_handler
    if not hasattr(globals(), 'spotify_handler'):
        spotify_handler = SpotifyHandler()
    return spotify_handler


# @log_execution
# @handle_errors
# def process_lyrics():
#     """Process missing lyrics for all songs in audio folder"""
#     spotify = initialize_spotify()
#     audio_path = Path("Assets/spotify/audio")
#     lyrics_path = Path("Assets/spotify/lyrics")
    
#     audio_files = list(audio_path.glob("*.mp3"))
#     for audio_file in tqdm(audio_files, desc="Processing lyrics", unit="song"):
#         song_name = spotify.sanitize_filename(audio_file.stem)
#         lyrics_file = lyrics_path / f"{song_name}_lyrics.srt"
        
#         if not lyrics_file.exists():
#             track_info = spotify.search_track(audio_file.stem)  # Use original name for search
#             if track_info:
#                 spotify.get_track_lyrics(track_info['id'], song_name)
# @log_execution
# @handle_errors
# def process_info():
#     """Process missing song info files with improved error handling"""
#     try:
#         spotify = initialize_spotify()
#         audio_path = Path("Assets/spotify/audio")
#         info_path = Path("Assets/spotify/song_info")
#         info_path.mkdir(exist_ok=True, parents=True)
        
#         audio_files = list(audio_path.glob("*.mp3"))
        
#         if not audio_files:
#             print(f"No MP3 files found in {audio_path}")
#             return
            
#         for audio_file in tqdm(audio_files, desc="Processing song info", unit="song"):
            
#             song_name = spotify.sanitize_filename(audio_file.stem)
#             info_file = info_path / f"{song_name}_info.txt"
            
#             if not info_file.exists():
#                 try:
#                     research_song_info(audio_file.stem)  # Use original name for research
#                 except Exception as e:
#                     print(f"Error processing {audio_file.stem}: {str(e)}")
                    
#     except Exception as e:
#         print(f"Process info failed with error: {str(e)}")
#         raise




# @log_execution
# @handle_errors
# def process_analysis():
#     """Process missing analysis files"""
#     spotify = initialize_spotify()
#     audio_path = Path("Assets/spotify/audio") 
#     analysis_path = Path("Assets/spotify/analysis")
    
#     # Print all files for debugging
#     audio_files = list(audio_path.glob("*.mp3"))
#     print(f"Found {len(audio_files)} MP3 files in {audio_path}:")
#     for file in audio_files:
#         print(f"- {file.name}")
    
#     for audio_file in tqdm(audio_files, desc="Processing audio analysis", unit="song"):
#         try:
#             # Use original filename as is
#             song_name = audio_file.stem
#             safe_name = spotify.sanitize_filename(song_name)
#             analysis_file = analysis_path / f"{safe_name}_analysis.json"
            
#             # Only process if analysis doesn't exist
#             if not analysis_file.exists():
#                 beats = spotify.generate_internal_beats(str(audio_file))
#                 sections = spotify.generate_internal_sections(str(audio_file))
                
#                 analysis = {
#                     "meta": {
#                         "analyzer_version": "internal_v1.0",
#                         "platform": "internal",
#                         "detailed_status": "OK",
#                         "status_code": 0,
#                         "timestamp": int(time.time()),
#                         "analysis_time": 0,
#                         "input_process": "internal"
#                     },
#                     "track": {
#                         "duration": librosa.get_duration(path=str(audio_file)),
#                         "sample_rate": 22050,
#                         "tempo": float(np.mean([s["tempo"] for s in sections])),
#                     },
#                     "bars": beats,
#                     "sections": sections
#                 }
                
#                 with open(analysis_file, 'w') as f:
#                     json.dump(analysis, f, indent=2)
                
#         except Exception as e:
#             print(f"Error processing {audio_file.name}: {str(e)}")
#             continue

# @log_execution
# @handle_errors
# def process_bars():
#     """Process missing bars data"""
#     spotify = initialize_spotify()
#     audio_path = Path("Assets/spotify/audio")
#     bars_path = Path("Assets/spotify/beats")
    
#     # Print all files for debugging
#     audio_files = list(audio_path.glob("*.mp3"))
#     print(f"Found {len(audio_files)} MP3 files in {audio_path}:")
#     for file in audio_files:
#         print(f"- {file.name}")
    
#     for audio_file in tqdm(audio_files, desc="Processing beats data", unit="song"):
#         try:
#             # Use original filename as is
#             song_name = audio_file.stem
#             safe_name = spotify.sanitize_filename(song_name)
#             bars_file = bars_path / f"{safe_name}_bars.json"
            
#             # Only process if bars don't exist
#             if not bars_file.exists():
#                 beats = spotify.generate_internal_beats(str(audio_file))
                
#                 with open(bars_file, 'w') as f:
#                     json.dump(beats, f, indent=2)
                
#         except Exception as e:
#             print(f"Error processing {audio_file.name}: {str(e)}")
#             continue

from GoogleServiceAPI import GoogleDriveServiceOperations_two 
import config
from typing import List

def get_folder_names_from_links(drive_ops, links: List[str]) -> List[str]:
    if not links:
        print("No drive links provided")
        return []
    
    folder_names = []
    for url in links:
        try:
            folder_id = url.split('folders/')[-1].split('?')[0]
            folder_info = drive_ops.service.files().get(
                fileId=folder_id,
                fields='name',
                supportsAllDrives=True
            ).execute()
            folder_names.append(folder_info['name'])
        except Exception as e:
            print(f"Error getting folder name for {url}: {e}")
            continue
    return folder_names

@log_execution
@handle_errors
def process_analysis():
   """Process missing analysis files for each folder"""
   spotify = initialize_spotify()
   drive_ops = GoogleDriveServiceOperations_two()
   folder_names = get_folder_names_from_links(drive_ops, config.links)
   
   for folder_name in folder_names:
       base_path = Path(f"Assets/spotify/{folder_name}")
       audio_path = base_path / "audio"
       analysis_path = base_path / "analysis"
       analysis_path.mkdir(parents=True, exist_ok=True)
       
       audio_files = list(audio_path.glob("*.mp3"))
       for audio_file in tqdm(audio_files, desc=f"Processing audio analysis for {folder_name}"):
           try:
               song_name = audio_file.stem
               safe_name = spotify.sanitize_filename(song_name)
               analysis_file = analysis_path / f"{safe_name}_analysis.json"
               
               if not analysis_file.exists():
                   beats = spotify.generate_internal_beats(str(audio_file))
                   sections = spotify.generate_internal_sections(str(audio_file))
                   
                   analysis = {
                       "meta": {
                           "analyzer_version": "internal_v1.0",
                           "platform": "internal",
                           "detailed_status": "OK",
                           "status_code": 0,
                           "timestamp": int(time.time()),
                           "analysis_time": 0,
                           "input_process": "internal"
                       },
                       "track": {
                           "duration": librosa.get_duration(path=str(audio_file)),
                           "sample_rate": 22050,
                           "tempo": float(np.mean([s["tempo"] for s in sections])),
                       },
                       "bars": beats,
                       "sections": sections
                   }
                   
                   with open(analysis_file, 'w') as f:
                       json.dump(analysis, f, indent=2)
                   
           except Exception as e:
               print(f"Error processing {audio_file.name}: {str(e)}")
               continue

@log_execution
@handle_errors
def process_bars():
   """Process missing bars data for each folder"""
   spotify = initialize_spotify()
   drive_ops = GoogleDriveServiceOperations_two()
   folder_names = get_folder_names_from_links(drive_ops, config.links)
   
   for folder_name in folder_names:
       base_path = Path(f"Assets/spotify/{folder_name}")
       audio_path = base_path / "audio"
       bars_path = base_path / "beats"
       bars_path.mkdir(parents=True, exist_ok=True)
       
       audio_files = list(audio_path.glob("*.mp3"))
       for audio_file in tqdm(audio_files, desc=f"Processing beats for {folder_name}"):
           try:
               song_name = audio_file.stem
               safe_name = spotify.sanitize_filename(song_name)
               bars_file = bars_path / f"{safe_name}_bars.json"
               
               if not bars_file.exists():
                   beats = spotify.generate_internal_beats(str(audio_file))
                   with open(bars_file, 'w') as f:
                       json.dump(beats, f, indent=2)
                   
           except Exception as e:
               print(f"Error processing {audio_file.name}: {str(e)}")
               continue

@log_execution
@handle_errors
def process_info():
    spotify = initialize_spotify()
    drive_ops = GoogleDriveServiceOperations_two()
    folder_names = get_folder_names_from_links(drive_ops, config.links)
    
    for folder_name in folder_names:
        base_path = Path(f"Assets/spotify/{folder_name}")
        audio_path = base_path / "audio"
        info_path = base_path / "song_info"
        info_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(audio_path.glob("*.mp3"))
        for audio_file in tqdm(audio_files, desc=f"Processing info for {folder_name}"):
            song_name = spotify.sanitize_filename(audio_file.stem)
            info_file = info_path / f"{song_name}_info.txt"
            
            if not info_file.exists():
                try:
                    research_song_info(audio_file.stem, folder_name)
                except Exception as e:
                    print(f"Error processing {audio_file.stem}: {str(e)}")

@log_execution
@handle_errors
def process_lyrics():
    spotify = initialize_spotify()
    drive_ops = GoogleDriveServiceOperations_two()
    folder_names = get_folder_names_from_links(drive_ops, config.links)
    
    for folder_name in folder_names:
        base_path = Path(f"Assets/spotify/{folder_name}")
        audio_path = base_path / "audio"
        lyrics_path = base_path / "lyrics"
        lyrics_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(audio_path.glob("*.mp3"))
        for audio_file in tqdm(audio_files, desc=f"Processing lyrics for {folder_name}"):
            song_name = spotify.sanitize_filename(audio_file.stem)
            lyrics_file = lyrics_path / f"{song_name}_lyrics.srt"
            
            if not lyrics_file.exists():
                track_info = spotify.search_track(audio_file.stem)
                if track_info:
                    spotify.get_track_lyrics(track_info['id'], audio_file.stem, folder_name)



# @log_execution
# @handle_errors
# def prepare_song_embedding_data():
#     """Creates a JSON file with embeddings for each song, including info and lyrics."""
#     try:
#         base_path = Path("Assets/spotify")
        
#         # Input paths
#         song_info_path = base_path / "song_info"
#         lyrics_path = base_path / "lyrics"
        
#         # Output path
#         song_embedding_data_path = base_path / "song_embedding_data.json"

#         embedding_data = {}

#         # Process song info files
#         for info_file in song_info_path.glob("*_info.txt"):
#             song_name = info_file.stem[:-5]  # Remove '_info' suffix
#             lyrics_file = lyrics_path / f"{song_name}_lyrics.srt"
            
#             if lyrics_file.exists():
#                 try:
#                     # Read song info with error handling for encoding
#                     try:
#                         with open(info_file, 'r', encoding='utf-8') as f:
#                             info_content = f.read()
#                     except UnicodeDecodeError:
#                         # Try different encoding if utf-8 fails
#                         with open(info_file, 'r', encoding='cp1252') as f:
#                             info_content = f.read()
                            
#                     # Extract summary section with error handling
#                     summary_parts = info_content.split("Research Summary:\n")
#                     summary_section = summary_parts[-1].strip() if len(summary_parts) > 1 else ""
                    
#                     # Read and process lyrics with error handling
#                     try:
#                         with open(lyrics_file, 'r', encoding='utf-8') as f:
#                             lyrics_content = f.read()
#                     except UnicodeDecodeError:
#                         with open(lyrics_file, 'r', encoding='cp1252') as f:
#                             lyrics_content = f.read()
                            
#                     # Extract only the lyrics text, skipping timestamps
#                     lyrics_lines = []
#                     for line in lyrics_content.split('\n'):
#                         if not (line.strip().isdigit() or '-->' in line or not line.strip()):
#                             lyrics_lines.append(line.strip())
#                     lyrics_text = ' '.join(lyrics_lines)
                    
#                     # Store in embedding data
#                     embedding_data[song_name] = {
#                         'song_info_summary': summary_section,
#                         'song_lyrics': lyrics_text
#                     }
#                 except Exception as e:
#                     print(f"Error processing {song_name}: {str(e)}")
#                     continue

#         print(f"Processed {len(embedding_data)} songs")

#         if embedding_data:  # Only save if we have data
#             # Save the embedding data to JSON file
#             with open(song_embedding_data_path, 'w', encoding='utf-8') as f:
#                 json.dump(embedding_data, f, indent=4, ensure_ascii=False)
#             print(f"Song embedding data saved to {song_embedding_data_path}")
#         else:
#             print("No data to save - check if input files exist and are readable")

#     except Exception as e:
#         print(f"Error in prepare_song_embedding_data: {str(e)}")
#         raise

# @log_execution
# @handle_errors
# def process_song_embeddings():
#     """
#     Preprocesses the song embedding data, generates embeddings, and creates an index.
#     """
#     base_path = Path("Assets/spotify")
#     song_embedding_data_path = base_path / "song_embedding_data.json"
    
#     # Check if embedding data exists
#     if not song_embedding_data_path.exists():
#         print(f"Embedding data file not found at {song_embedding_data_path}")
#         print("Please run prepare_song_embedding_data first")
#         return
        
#     model_name = 'all-MiniLM-L12-v2'
    
#     # Helper function to clean text
#     def clean_text(text):
#         if not isinstance(text, str):
#             text = str(text)
#         # More robust text cleaning
#         text = re.sub(r'[\*"\'\\\[\]{}]', '', text)
#         try:
#             text = text.encode('utf-8').decode('unicode-escape')
#         except:
#             pass  # If decode fails, use original text
#         text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
#         return ' '.join(text.split())  # Normalize whitespace
    
#     try:
#         # Load embedding data with explicit encoding
#         with open(song_embedding_data_path, 'r', encoding='utf-8') as f:
#             song_embedding_data = json.load(f)
        
#         if not song_embedding_data:
#             print("No data found in embedding file")
#             return
            
#         print(f"Loaded embedding data for {len(song_embedding_data)} songs")
        
#         # Initialize the model
#         model = SentenceTransformer(model_name)
#         print("Model initialized")
        
#         # Generate embeddings
#         embeddings = []
#         song_names = []
        
#         for song_name, data in tqdm(song_embedding_data.items(), desc="Generating embeddings"):
#             try:
#                 info_summary = clean_text(data.get('song_info_summary', ''))
#                 lyrics = clean_text(data.get('song_lyrics', ''))
#                 combined_text = f"Song information: {info_summary} Song lyrics: {lyrics}"
#                 embedding = model.encode(combined_text)
#                 embeddings.append(embedding)
#                 song_names.append(song_name)
#             except Exception as e:
#                 print(f"Error processing {song_name}: {str(e)}")
#                 continue
        
#         if not embeddings:
#             print("No embeddings generated")
#             return
            
#         print(f"Generated embeddings for {len(embeddings)} songs")
        
#         # Create FAISS index
#         embeddings = np.array(embeddings)
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatL2(dimension)
#         index.add(embeddings.astype('float32'))
        
#         # Save preprocessed data
#         song_preprocessed_data = {
#             'model': model,
#             'index': index,
#             'song_names': song_names,
#             'song_embedding_data': song_embedding_data
#         }
        
#         with open(base_path / 'song_preprocessed_data.pkl', 'wb') as f:
#             pickle.dump(song_preprocessed_data, f)
        
#         print("Song embedding preprocessing completed and data saved")
        
#     except Exception as e:
#         print(f"Error in process_song_embeddings: {str(e)}")
#         raise


# # # Usage
# # if __name__ == "__main__":
# #     initialize_spotify()
# #     process_lyrics()
# #     process_info()
# #     process_analysis()  
# #     process_bars()



@log_execution
@handle_errors
def prepare_song_embedding_data():
    """Creates embeddings for songs in specified folder(s)"""
    try:
        drive_ops = GoogleDriveServiceOperations_two()
        folder_names = get_folder_names_from_links(drive_ops, config.links)
        base_path = Path("Assets/spotify")
        embedding_data = {}
        processing_log = {}

        for folder_name in folder_names:
            info_path = base_path / folder_name / "song_info"
            lyrics_path = base_path / folder_name / "lyrics"
            
            info_files = list(info_path.glob("*_info.txt"))
            for info_file in info_files:
                song_name = info_file.stem[:-5]
                lyrics_file = lyrics_path / f"{song_name}_lyrics.srt"
                
                if lyrics_file.exists():
                    try:
                        with open(info_file, 'r', encoding='utf-8') as f:
                            info_content = f.read()
                        with open(lyrics_file, 'r', encoding='utf-8') as f:
                            lyrics_content = f.read()
                            
                        summary_parts = info_content.split("Research Summary:\n")
                        summary_section = summary_parts[-1].strip() if len(summary_parts) > 1 else ""
                        
                        lyrics_lines = [line.strip() for line in lyrics_content.split('\n')
                                      if not (line.strip().isdigit() or '-->' in line or not line.strip())]
                        lyrics_text = ' '.join(lyrics_lines)
                        
                        embedding_data[song_name] = {
                            'song_info_summary': summary_section,
                            'song_lyrics': lyrics_text,
                            'playlist': folder_name
                        }
                        
                        processing_log[song_name] = {
                            'playlist': folder_name,
                            'info_path': str(info_file),
                            'lyrics_path': str(lyrics_file)
                        }
                    except Exception as e:
                        print(f"Error processing {song_name}: {str(e)}")

        if embedding_data:
            embedding_path = base_path / "embeddings"
            embedding_path.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with open(embedding_path / f"song_embedding_data_{timestamp}.json", 'w') as f:
                json.dump(embedding_data, f, indent=4)
            with open(embedding_path / f"embedding_log_{timestamp}.json", 'w') as f:
                json.dump(processing_log, f, indent=4)
            
            return embedding_data
    except Exception as e:
        print(f"Error in prepare_song_embedding_data: {str(e)}")
        raise

@log_execution
@handle_errors
def process_song_embeddings():
    """Generate embeddings for most recent data"""
    try:
        base_path = Path("Assets/spotify/embeddings")
        embedding_files = sorted(base_path.glob("song_embedding_data_*.json"))
        
        if not embedding_files:
            print("No embedding data found")
            return
            
        latest_file = embedding_files[-1]
        with open(latest_file, 'r') as f:
            song_embedding_data = json.load(f)

        device = 'cpu'    
        model = SentenceTransformer('all-MiniLM-L12-v2', device = device)
        embeddings = []
        song_names = []
        
        for song_name, data in song_embedding_data.items():
            try:
                combined_text = f"Song information: {data['song_info_summary']} Song lyrics: {data['song_lyrics']}"
                embedding = model.encode(combined_text)
                embeddings.append(embedding)
                song_names.append(song_name)
            except Exception as e:
                print(f"Error embedding {song_name}: {str(e)}")
                
        index = faiss.IndexFlatL2(embeddings[0].shape[0])
        index.add(np.array(embeddings).astype('float32'))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preprocessed_data = {
            'model': model,
            'index': index,
            'song_names': song_names,
            'song_embedding_data': song_embedding_data,
            'created_at': timestamp
        }
        
        with open(base_path / f"song_preprocessed_data_{timestamp}.pkl", 'wb') as f:
            pickle.dump(preprocessed_data, f)
            
    except Exception as e:
        print(f"Error in process_song_embeddings: {str(e)}")
        raise


@log_execution
@handle_errors
def update_json_with_analysis_data():
    """Update the most recent embedding data JSON with analysis and beats data"""
    try:
        # Find the most recent embedding data JSON
        base_path = Path("Assets/spotify")
        embedding_path = base_path / "embeddings"
        
        if not embedding_path.exists():
            print("Embedding path not found")
            return False
            
        # Find most recent embedding data file
        json_files = sorted(embedding_path.glob("song_embedding_data_*.json"))
        if not json_files:
            print("No embedding data files found")
            return False
            
        latest_json = json_files[-1]
        timestamp = latest_json.stem.split("_")[-1]
        print(f"Using embedding data from {latest_json}")
        
        # Load the JSON
        with open(latest_json, 'r', encoding='utf-8') as f:
            embedding_data = json.load(f)
            
        # Find the folder for each song
        spotify = initialize_spotify()
        
        # Get all folder names
        drive_ops = GoogleDriveServiceOperations_two()
        folder_names = get_folder_names_from_links(drive_ops, config.links)
        
        # Update songs with analysis and beats data
        updated_count = 0
        
        for song_name, song_data in tqdm(embedding_data.items(), desc="Adding analysis data"):
            # Skip if already has analysis
            if 'analysis' in song_data:
                continue
                
            safe_name = spotify.sanitize_filename(song_name)
            
            # Get the folder for this song
            folder_name = song_data.get('playlist')
            if not folder_name:
                continue
                
            # Check for analysis file
            analysis_file = base_path / folder_name / "analysis" / f"{safe_name}_analysis.json"
            beats_file = base_path / folder_name / "beats" / f"{safe_name}_bars.json"
            
            # Add analysis data if it exists
            if analysis_file.exists():
                try:
                    with open(analysis_file, 'r') as f:
                        analysis_data = json.load(f)
                    embedding_data[song_name]['analysis'] = analysis_data
                    updated_count += 1
                except Exception as e:
                    print(f"Error reading analysis for {song_name}: {e}")
            
            # Add beats data if analysis doesn't exist but beats do
            elif beats_file.exists():
                try:
                    with open(beats_file, 'r') as f:
                        beats_data = json.load(f)
                    embedding_data[song_name]['analysis'] = {'bars': beats_data}
                    updated_count += 1
                except Exception as e:
                    print(f"Error reading beats for {song_name}: {e}")
        
        print(f"Updated {updated_count} songs with analysis data")
        
        # Save the updated JSON file with the SAME timestamp
        # This ensures it will be picked up correctly by subsequent functions
        with open(latest_json, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, indent=4, ensure_ascii=False)
            
        print(f"Updated original JSON file with analysis data: {latest_json}")
        
        return True
        
    except Exception as e:
        print(f"Error updating JSON: {e}")
        return False
    
# @log_execution
# @handle_errors
# def sync_pkl_timestamp():
#     """Sync the PKL file timestamp with the JSON file timestamp"""
#     try:
#         base_path = Path("Assets/spotify/embeddings")
        
#         # Find latest JSON and PKL files
#         json_files = sorted(base_path.glob("song_embedding_data_*.json"))
#         pkl_files = sorted(base_path.glob("song_preprocessed_data_*.pkl"))
        
#         if not json_files or not pkl_files:
#             print("Missing required files")
#             return False
            
#         latest_json = json_files[-1]
#         latest_pkl = pkl_files[-1]
        
#         # Get JSON timestamp
#         json_timestamp = latest_json.stem.split("_")[-1]
        
#         # Rename PKL file to match JSON timestamp
#         new_pkl_path = base_path / f"song_preprocessed_data_{json_timestamp}.pkl"
#         latest_pkl.rename(new_pkl_path)
        
#         print(f"Renamed PKL file to match JSON timestamp: {json_timestamp}")
#         return True
        
#     except Exception as e:
#         print(f"Error syncing timestamps: {e}")
#         return False
    
@log_execution
@handle_errors
def sync_pkl_timestamp():
    """Sync the PKL file timestamp with the JSON file timestamp"""
    try:
        base_path = Path("Assets/spotify/embeddings")
        
        # Find latest JSON and PKL files
        json_files = sorted(base_path.glob("song_embedding_data_*.json"))
        pkl_files = sorted(base_path.glob("song_preprocessed_data_*.pkl"))
        
        if not json_files or not pkl_files:
            print("Missing required files")
            return False
            
        latest_json = json_files[-1]
        latest_pkl = pkl_files[-1]
        
        # Extract the full timestamp from the JSON filename
        # Instead of just the last part, get everything after "song_embedding_data_"
        json_filename = latest_json.stem  # e.g., song_embedding_data_20250131_010804
        prefix = "song_embedding_data_"
        json_timestamp = json_filename[len(prefix):]  # e.g., 20250131_010804
        
        # Rename PKL file to match JSON timestamp
        new_pkl_path = base_path / f"song_preprocessed_data_{json_timestamp}.pkl"
        latest_pkl.rename(new_pkl_path)
        
        print(f"Renamed PKL file to match JSON timestamp: {json_timestamp}")
        return True
        
    except Exception as e:
        print(f"Error syncing timestamps: {e}")
        return False
    


