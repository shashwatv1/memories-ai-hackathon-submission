import os
import csv
import shutil
from pathlib import Path
import uuid
import json

import cv2
import logging  # Added for logging
# Adjust the imports below based on your project structure
# from GoogleAPI import GoogleDriveOperations
from GoogleServiceAPI import GoogleDriveServiceOperations
from CacheManager import CacheManager
import numpy as np
import subprocess
from difflib import SequenceMatcher
# Add this near the top of your script
logging.getLogger().setLevel(logging.WARNING)

# helper_functions.py (add near your other imports)
import os, json, subprocess
from functools import wraps
from pathlib import Path
from typing import Optional

def _ffprobe_ok(path: str) -> bool:
    try:
        p = subprocess.run(
            ["ffprobe","-v","error","-show_format","-of","json", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        j = json.loads(p.stdout or "{}")
        fmt = (j.get("format") or {}).get("format_name","")
        return any(x in fmt for x in ["mp4","mov","mov,mp4,m4a,3gp,3g2,mj2"])
    except Exception:
        return False

def _probe_pixfmt(path: str) -> Optional[str]:
    try:
        p = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=pix_fmt","-of","json", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        j = json.loads(p.stdout or "{}")
        s = (j.get("streams") or [{}])[0]
        return s.get("pix_fmt")
    except Exception:
        return None

def _profile_pixfmt(pix_fmt: Optional[str]):
    pf = (pix_fmt or "").lower()
    if "yuv444" in pf: return "high444","yuv444p"
    if "yuv422" in pf: return "high422","yuv422p"
    return "main","yuv420p"

def robustify_load_video(fn):
    """Decorator to validate/repair MP4 before the actual load_video runs."""
    @wraps(fn)
    def wrapper(video_path: str, *args, **kwargs):
        # 1) HTML stub / tiny-file guard
        try:
            with open(video_path, "rb") as f:
                head = f.read(2048)
            if head.lstrip().startswith(b"<!DOCTYPE html") or b"Google Drive" in head:
                raise ValueError("Downloaded HTML stub instead of video")
        except Exception:
            pass
        if os.path.getsize(video_path) < 16 * 1024:
            raise ValueError("Downloaded file too small to be a valid MP4")

        # 2) ffprobe → faststart remux → minimal re-encode
        if not _ffprobe_ok(video_path):
            faststart = str(Path(video_path).with_suffix(".faststart.mp4"))
            try:
                subprocess.run(
                    ["ffmpeg","-y","-v","error","-i",video_path,"-c","copy","-movflags","faststart",faststart],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
                )
                if _ffprobe_ok(faststart):
                    video_path = faststart
                else:
                    prof, px = _profile_pixfmt(_probe_pixfmt(video_path))
                    reenc = str(Path(video_path).with_suffix(".reencoded.mp4"))
                    subprocess.run(
                        ["ffmpeg","-y","-v","error","-i",video_path,
                         "-c:v","libx264","-profile:v",prof,"-pix_fmt",px,
                         "-preset","veryfast","-crf","22","-c:a","aac","-b:a","128k", reenc],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
                    )
                    if _ffprobe_ok(reenc):
                        video_path = reenc
                    else:
                        raise ValueError("Unrecoverable container error (moov/format)")
            except subprocess.CalledProcessError as e:
                raise ValueError(f"FFmpeg repair failed: {e.stderr or e}") from e

        return fn(video_path, *args, **kwargs)
    return wrapper

# apply it either as a decorator:
# @robustify_load_video
# def load_video(...): ...
# or after definition:
# load_video = robustify_load_video(load_video)

class HelperFunctions:
    @staticmethod
    def write_to_csv(filepath, data):
        file_exists = os.path.isfile(filepath)
        with open(filepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()  # Write header only if file doesn't exist
            writer.writerow(data)    # Append row to the file
    
    @staticmethod
    def get_cache_directory():
        """Get the standard cache directory path in the project root"""
        # Get current file's directory (where HelperFunctions class is defined)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Create cache directory next to the current file
        cache_dir = os.path.join(current_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def _get_mapping(file_path):
        """Get mapping.json from parent directory"""
        try:
            # Handle Windows paths by replacing backslashes and fixing escapes
            clean_path = str(file_path).replace('\r', '').replace('\n', '').replace('\t', '')
            clean_path = clean_path.replace('\\', '/')
            
            # Split path components
            parts = clean_path.split('/')
            
            # Find 'Media' directory index
            try:
                media_index = parts.index('Media')
                # Reconstruct path up to Media directory
                base_path = '/'.join(parts[:media_index+1])
                mapping_file = f"{base_path}/filename_mapping.json"
                
                if os.path.exists(mapping_file):
                    logging.info(f"Found mapping file at: {mapping_file}")
                    with open(mapping_file) as f:
                        return json.load(f)
                else:
                    logging.error(f"Mapping file not found at: {mapping_file}")
                    
            except ValueError:
                logging.error(f"No 'Media' directory found in path: {clean_path}")
                
        except Exception as e:
            logging.error(f"Error loading mapping file for {file_path}: {str(e)}")
        
        return None
    


    
    @staticmethod
    def calculate_total_video_duration(directory):
        """
        Recursively searches for video files in the given directory and its subfolders,
        and calculates the total duration of all videos found.

        :param directory: Path to the directory to search for videos
        :return: Total duration in seconds, number of videos found
        """
        total_duration = 0
        video_count = 0
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')  # Add more if needed

        print(f"Searching for videos in {directory} and its subfolders...")
        
        # Walk through directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_path = os.path.join(root, file)
                    video_count += 1
                    
                    try:
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            print(f"Error opening video file: {video_path}")
                            continue
                        
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps
                        total_duration += duration
                        
                        cap.release()
                    except Exception as e:
                        print(f"Error processing video {video_path}: {str(e)}")

        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\nTotal number of videos found: {video_count}")
        print(f"Total duration: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
        print(f"Total duration in seconds: {total_duration:.2f}")

        return total_duration, video_count

    @staticmethod
    def copy_files(input_directory, output_directory):
        """Copies all files from the input directory to the output directory."""
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # List all files in the input directory
        files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

        # Copy each file to the output directory
        for file in files:
            src = os.path.join(input_directory, file)
            dst = os.path.join(output_directory, file)
            shutil.copy(src, dst)


    @staticmethod
    def create_media_symlinks(source_directories, output_directory):
        """
        Creates symlinks for various types of media files.
        Organizes symlinks by media type within a 'Media' folder in the output directory.
        Assigns a random unique key to each symlink.
        
        :param source_directories: List of source directory paths to search for media files
        :param output_directory: Directory to use for the destination folder
        """
        # Convert all paths to absolute paths
        current_dir = Path.cwd()
        source_directories = [current_dir / Path(src) for src in source_directories]
        base_dest_dir = current_dir / Path(output_directory) / "Media"
        
        # Define media types and their corresponding file extensions
        media_types = {
            "Images": (".jpg", ".jpeg", ".png", ".gif", ".bmp",".heic", ".tiff"),
            "Videos": (".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"),
            "Audio": (".mp3", ".wav", ".ogg", ".flac", ".aac"),
            "Documents": (".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"),
        }
        
        # Create destination directories for each media type
        for media_type in media_types:
            os.makedirs(base_dest_dir / media_type, exist_ok=True)
        
        # Create a directory for other file types
        os.makedirs(base_dest_dir / "Other", exist_ok=True)

        def rename_uppercase_extension(file_path):
            if file_path.suffix.isupper():
                new_name = file_path.stem + file_path.suffix.lower()
                new_path = file_path.with_name(new_name)
                os.rename(file_path, new_path)
                print(f"Renamed file: {file_path} -> {new_path}")
                return new_path
            return file_path
        
        def get_media_type(file_path):
            ext = file_path.suffix.lower()
            for media_type, extensions in media_types.items():
                if ext in extensions:
                    return media_type
            return "Other"
        
        filename_mapping = {}
        
        for source_dir in source_directories:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    file_path = rename_uppercase_extension(file_path)
                    media_type = get_media_type(file_path)
                    
                    unique_key = str(uuid.uuid4())
                    new_filename = f"{unique_key}{file_path.suffix}"
                    
                    dest_dir = base_dest_dir / media_type
                    link_path = dest_dir / new_filename
                    
                    try:
                        os.symlink(file_path, link_path)
                        # print(f"Created symlink: {link_path} -> {file_path}")
                        
                        filename_mapping[file] = {
                            "unique_key": unique_key,
                            "original_path": str(file_path),
                            "symlink_path": str(link_path),
                            "media_type": media_type
                        }
                    except OSError as e:
                        print(f"Error creating symlink for {file_path}: {e}")
                        raise  # Re-raise the exception to stop execution

        mapping_file = base_dest_dir / "filename_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(filename_mapping, f, indent=4)
        
        print(f"Filename mapping saved to {mapping_file}")

    def load_image(image_path, max_retries=3):
        """
        Loads an image while handling broken symlinks, cache corruption, and download issues.
        
        Args:
            image_path: Path to the image file
            max_retries: Maximum number of retry attempts for loading/downloading
            
        Returns:
            numpy.ndarray: Loaded image or None if loading fails
        """
        import tempfile
        import hashlib
        import shutil
        import os
        import tempfile
        import hashlib
        import shutil
        import os
        import pillow_heif
        from PIL import Image

        # Use a valid temp directory for Windows
        # cache_dir = os.path.join(tempfile.gettempdir(), "gdrive_cache")
        cache_dir = HelperFunctions.get_cache_directory()


        # cache_manager = CacheManager(cache_dir=cache_dir)
        cache_manager = CacheManager(cache_dir=HelperFunctions.get_cache_directory())
        # drive_ops = GoogleDriveOperations()
        drive_ops = GoogleDriveServiceOperations(service_account_file='service-account.json')


        
        def verify_image_integrity(file_path):
            """Verify if the image file is valid and not corrupted."""
            try:
                if file_path.lower().endswith('.heic'):
                    # Convert HEIC to temp JPG for verification
                    temp_jpg = os.path.join(os.path.dirname(file_path), f"temp_{os.path.basename(file_path)}.jpg")
                    convert_heic_to_jpg(file_path, temp_jpg)
                    img = cv2.imread(temp_jpg)
                    os.remove(temp_jpg)
                else:
                    img = cv2.imread(file_path)
                # img = cv2.imread(file_path)
                if img is None:
                    return False
                # Check if image has valid dimensions and channels
                if len(img.shape) < 2 or (len(img.shape) == 3 and img.shape[2] not in [1, 3, 4]):
                    return False
                # Check if image has non-zero size
                if img.size == 0 or 0 in img.shape:
                    return False
                return True
            except Exception:
                return False
        
        import os

        def convert_heic_to_jpg(input_path, output_path):
            """
            Convert HEIC to JPG format with better error handling and robust conversion.
            
            Args:
                input_path: Path to the HEIC input file
                output_path: Path where the JPG should be saved
                
            Returns:
                str: Path to the converted JPG file
            """
            import os
            import shutil
            import logging
            import time
            import pillow_heif
            from PIL import Image
            
            logger = logging.getLogger(__name__)
            logger.debug(f"Converting HEIC to JPG: {input_path} -> {output_path}")
            
            # Ensure input file exists and has content
            if not os.path.exists(input_path):
                logger.debug(f"Input HEIC file doesn't exist: {input_path}")
                raise FileNotFoundError(f"Input file not found: {input_path}")
                
            file_size = os.path.getsize(input_path)
            if file_size == 0:
                logger.debug(f"Input HEIC file is empty: {input_path}")
                raise ValueError(f"Input file is empty: {input_path}")
                
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use absolute paths to avoid any relative path issues
            input_path = os.path.abspath(input_path)
            output_path = os.path.abspath(output_path)
            
            # Read file header for format detection
            with open(input_path, 'rb') as f:
                header = f.read(16)
                header_hex = ' '.join([f"{b:02x}" for b in header[:16]])
                logger.debug(f"File header: {header_hex}")
            
            # Check if this is actually a JPEG file with wrong extension
            if header.startswith(b'\xff\xd8\xff'):
                logger.debug(f"File is actually JPEG with HEIC extension: {input_path}")
                shutil.copy2(input_path, output_path)
                return output_path
            
            # Try conversion with pillow-heif
            try:
                pillow_version = getattr(pillow_heif, '__version__', 'unknown')
                logger.debug(f"Using pillow-heif version: {pillow_version}")
                
                # First attempt - standard pillow-heif approach
                heif_file = pillow_heif.open_heif(input_path)
                logger.debug(f"HEIC file opened with dimensions: {heif_file.size}")
                
                # Create PIL image
                image = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )
                
                # Save as JPEG with high quality
                image.save(output_path, "JPEG", quality=95)
                
                # Verify the output file was created and has content
                if not os.path.exists(output_path):
                    raise Exception("Output file was not created")
                    
                if os.path.getsize(output_path) == 0:
                    raise Exception("Output file is empty")
                    
                logger.debug(f"Successfully converted to JPEG: {output_path}")
                return output_path
                
            except Exception as first_error:
                logger.warning(f"First conversion attempt failed: {str(first_error)}")
                
                # Second attempt - alternative approach using pillow_heif's higher-level API
                try:
                    logger.debug("Trying alternative conversion method")
                    
                    # Force pillow_heif to decode the HEIC file to a PIL Image directly
                    pil_image = pillow_heif.read_heif(input_path)
                    
                    # Save directly to JPEG
                    pil_image.save(output_path, "JPEG", quality=95)
                    
                    # Verify the output
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        logger.debug(f"Successfully converted using alternative method: {output_path}")
                        return output_path
                        
                    raise Exception("Alternative method failed to create valid output")
                    
                except Exception as second_error:
                    # Log both errors for diagnostic purposes
                    logger.error(f"Both conversion methods failed.")
                    logger.error(f"First error: {str(first_error)}")
                    logger.error(f"Second error: {str(second_error)}")
                    
                    # Raise a combined error message
                    raise Exception(f"Failed to convert HEIC to JPEG: {str(first_error)} | {str(second_error)}")

        def calculate_file_hash(file_path):
            """Calculate SHA-256 hash of file to verify integrity."""
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()

        mapping_json = None
        last_error = None

        def create_fallback_image(width=100, height=100):
            """Create a small black image as fallback"""
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        def remove_from_mapping(image_path):
            """Remove entry from filename_mapping.json and delete the file"""
            try:
                # Get parent directory and mapping file path
                parent_dir = os.path.dirname(os.path.dirname(image_path))
                mapping_file = os.path.join(parent_dir, 'filename_mapping.json')
                
                if os.path.exists(mapping_file):
                    with open(mapping_file, 'r') as f:
                        mapping_data = json.load(f)
                    
                    # Find and remove the entry
                    filename = os.path.basename(image_path)
                    if filename in mapping_data:
                        del mapping_data[filename]
                        
                        # Write updated mapping back to file
                        with open(mapping_file, 'w') as f:
                            json.dump(mapping_data, f, indent=4)
                        
                        # Delete the actual file
                        if os.path.exists(image_path):
                            os.remove(image_path)
                        
                        logging.info(f"Removed {filename} from mapping and deleted file")
            except Exception as e:
                logging.error(f"Error removing mapping for {image_path}: {str(e)}")
        
        for attempt in range(max_retries):
            try:
                # logging.info(f"Attempt {attempt + 1} to load image: {image_path}")
                
                # 1. Handle symlink case
                if os.path.islink(image_path):
                    target = os.readlink(image_path)
                    


                if os.path.islink(image_path):
                    target = os.readlink(image_path)
                    
                    # Check in the cache directory
                    cache_file = os.path.join(cache_dir, os.path.basename(target))
                    if os.path.exists(cache_file) and verify_image_integrity(cache_file):
                        if cache_file.lower().endswith('.heic'):
                            temp_jpg = os.path.join(os.path.dirname(cache_file), f"temp_{os.path.basename(cache_file)}.jpg")
                            convert_heic_to_jpg(cache_file, temp_jpg)
                            image = cv2.imread(temp_jpg)
                            os.remove(temp_jpg)
                        else:
                            image = cv2.imread(cache_file)
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            return image
                    else:
                        # Try direct target path
                        if os.path.exists(target) and verify_image_integrity(target):
                            if target.lower().endswith('.heic'):
                                temp_jpg = os.path.join(os.path.dirname(target), f"temp_{os.path.basename(target)}.jpg")
                                convert_heic_to_jpg(target, temp_jpg)
                                image = cv2.imread(temp_jpg)
                                os.remove(temp_jpg)
                            else:
                                image = cv2.imread(target)
                            if image is not None:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                return image

                # 2. Try direct file path
                if os.path.exists(image_path) and verify_image_integrity(image_path):
                    if image_path.lower().endswith('.heic'):
                        temp_jpg = os.path.join(os.path.dirname(image_path), f"temp_{os.path.basename(image_path)}.jpg")
                        convert_heic_to_jpg(image_path, temp_jpg)
                        image = cv2.imread(temp_jpg)
                        os.remove(temp_jpg)
                    else:
                        image = cv2.imread(image_path)
                    if image is not None:
                        # logging.info(f"Loaded image directly from {image_path}")
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        return image
                
                # 3. Resolve Google Drive ID
                gdrive_id = None
                if os.path.islink(image_path):
                    target = os.readlink(image_path)
                    gdrive_id = os.path.basename(target)
                else:
                    # Load mapping if not already loaded
                    if mapping_json is None:
                        mapping_json = HelperFunctions._get_mapping(image_path)
                    if not mapping_json:
                        raise FileNotFoundError(f"No mapping file found for {image_path}")

                    # Get gdrive_id from mapping
                    norm_path = str(Path(image_path))
                    for _, info in mapping_json.items():
                        if info['symlink_path'] == norm_path:
                            gdrive_id = os.path.basename(info['original_path'])
                            break
                    


                if not gdrive_id:
                    raise FileNotFoundError(f"No mapping found for {image_path}")

                # 4. Handle cache and download
                cache_path = None
                if cache_manager.is_cached(gdrive_id):
                    cache_path = cache_manager.get_cache_path(gdrive_id)
                    if not verify_image_integrity(cache_path):
                        logging.warning(f"Corrupted cache file detected for {gdrive_id}")
                        cache_manager.remove_file(gdrive_id)
                        cache_path = None

                if cache_path is None:
                    # Download file with temporary name
                    file_ext = os.path.splitext(image_path)[1].lower()
                    temp_download_path = os.path.join(cache_dir, f"temp_{gdrive_id}_{attempt}{file_ext}")
                    drive_ops.download_file(gdrive_id, temp_download_path)
                    
                    # If it's HEIC, convert to JPG before caching
                    if temp_download_path.lower().endswith('.heic'):
                        if not os.path.exists(temp_download_path) or os.path.getsize(temp_download_path) == 0:
                            os.remove(temp_download_path)
                            raise Exception(f"Downloaded file is corrupted: {gdrive_id}")
                            
                        jpg_path = os.path.splitext(temp_download_path)[0] + '.jpg'
                        try:
                            convert_heic_to_jpg(temp_download_path, jpg_path)
                            os.remove(temp_download_path)  # Remove original HEIC
                            temp_download_path = jpg_path  # Use JPG for caching
                        except Exception as e:
                            if os.path.exists(temp_download_path):
                                os.remove(temp_download_path)
                            raise Exception(f"HEIC conversion failed: {str(e)}")
                    else:
                        # For non-HEIC files, verify integrity
                        if not verify_image_integrity(temp_download_path):
                            os.remove(temp_download_path)
                            raise Exception(f"Downloaded file is corrupted: {gdrive_id}")
                    
                    # Store file hash before caching
                    original_hash = calculate_file_hash(temp_download_path)
                    
                    # Add to cache (now either original file or converted JPG)
                    cache_manager.add_file(gdrive_id, temp_download_path)
                    cache_path = cache_manager.get_cache_path(gdrive_id)
                    
                    # Verify cached file integrity
                    if not os.path.exists(cache_path) or calculate_file_hash(cache_path) != original_hash:
                        raise Exception(f"Cache verification failed for {gdrive_id}")

                # 5. Update symlink
                if os.path.exists(image_path) or os.path.islink(image_path):
                    os.unlink(image_path)
                
                # Use absolute path with Windows extended-length path syntax
                # abs_cache_path = f"\\\\?\\{os.path.abspath(cache_path)}"
                abs_cache_path = os.path.abspath(cache_path)


                os.symlink(abs_cache_path, image_path)
                # logging.info(f"Created symlink {image_path} -> {abs_cache_path}")

                # 6. Final load attempt
                image = cv2.imread(abs_cache_path)
                if image is not None and verify_image_integrity(abs_cache_path):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Add this line
                    return image
                
                raise Exception(f"Failed to load image after caching: {gdrive_id}")



            except Exception as e:
                last_error = str(e)
                logging.error(f"Error loading image {image_path} (attempt {attempt + 1}): {last_error}")
                
                # Clean up any temporary files
                for temp_file in os.listdir(cache_dir):
                    if temp_file.startswith(f"temp_{gdrive_id}_"):
                        try:
                            os.remove(os.path.join(cache_dir, temp_file))
                        except Exception:
                            pass
                
                if attempt == max_retries - 1:
                    logging.warning(f"Max retries exceeded for {image_path}. Removing file and mapping.")
                    remove_from_mapping(image_path)
                    return create_fallback_image()

        # raise Exception(f"Failed to load image {image_path} after {max_retries} attempts. Last error: {last_error}")
        # logging.warning(f"Failed to load image {image_path} after {max_retries} attempts. Returning fallback image.")
        logging.warning(f"Failed to load image {image_path} after {max_retries} attempts. Removing file and mapping.")
        remove_from_mapping(image_path)
        return create_fallback_image()

    @staticmethod
    # @robustify_load_video
    def load_video(video_path, max_retries=3):
        """
        Loads a video while handling broken symlinks, cache corruption, and download issues.
        
        Args:
            video_path: Path to the video file
            max_retries: Maximum number of retry attempts for loading/downloading
            
        Returns:
            cv2.VideoCapture: Loaded video capture object or None if loading fails
        """
        import tempfile
        import hashlib
        import shutil

        # Use a valid temp directory for Windows
        # cache_dir = os.path.join(tempfile.gettempdir(), "gdrive_cache")
        cache_dir = HelperFunctions.get_cache_directory()


        # cache_manager = CacheManager(cache_dir=cache_dir)
        cache_manager = CacheManager(cache_dir=HelperFunctions.get_cache_directory())
        # drive_ops = GoogleDriveOperations()
        drive_ops = GoogleDriveServiceOperations(service_account_file='service-account.json')

        def check_moov_atom(file_path):
            """
            Check if file has moov atom.
            Checks both beginning and end of file since moov can be at either location.
            Videos with moov at the end are valid (just not optimized for streaming).
            """
            try:
                file_size = os.path.getsize(file_path)

                with open(file_path, 'rb') as f:
                    # Check first 10KB
                    first_chunk = f.read(10000)

                    # For files larger than 20KB, also check last 10KB
                    last_chunk = b''
                    if file_size > 20000:
                        f.seek(-10000, 2)  # Seek to 10KB before end
                        last_chunk = f.read(10000)

                has_ftyp = b'ftyp' in first_chunk
                has_moov_start = b'moov' in first_chunk
                has_moov_end = b'moov' in last_chunk if last_chunk else False
                has_moov = has_moov_start or has_moov_end

                if has_ftyp and not has_moov:
                    logging.warning(f"⚠️ {os.path.basename(file_path)} is missing moov atom")
                    return False
                elif has_moov:
                    location = "start" if has_moov_start else "end"
                    logging.info(f"✓ {os.path.basename(file_path)} has moov atom at {location}")
                    return True
                else:
                    logging.info(f"? {os.path.basename(file_path)} is not an MP4/MOV file")
                    return None
            except Exception as e:
                logging.error(f"Error checking moov atom: {str(e)}")
                return None
            
        def repair_video(file_path, output_path):
            """Repair video using FFmpeg."""
            try:
                logging.info(f"Repairing: {os.path.basename(file_path)} -> {os.path.basename(output_path)}")
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', file_path,
                    '-c', 'copy',
                    '-movflags', 'faststart',
                    output_path
                ]
                
                process = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=60
                )
                
                if process.returncode == 0 and os.path.exists(output_path):
                    logging.info(f"✓ Successfully repaired: {os.path.basename(file_path)}")
                    return True
                else:
                    logging.error(f"FFmpeg error: {process.stderr.decode('utf-8')[:100]}")
                    return False
            except Exception as e:
                logging.error(f"Error repairing video: {str(e)}")
                return False
        
        def verify_video_integrity(file_path):
            """Verify if the video file is valid and not corrupted."""
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    return False
                    
                # Check basic video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Basic validation
                if fps <= 0 or frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
                    cap.release()
                    return False
                    
                # Try reading first frame
                ret, frame = cap.read()
                cap.release()
                
                if not ret or frame is None:
                    return False
                    
                return True
            except Exception:
                return False

        def calculate_file_hash(file_path):
            """Calculate SHA-256 hash of file to verify integrity."""
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()

        mapping_json = None
        last_error = None
        
        for attempt in range(max_retries):
            try:

                if os.path.islink(video_path):
                    target = os.readlink(video_path)
                    
                    # # Check in the cache directory
                    # cache_file = os.path.join(cache_dir, os.path.basename(target))
                    # if os.path.exists(cache_file) and verify_video_integrity(cache_file):
                    #     cap = cv2.VideoCapture(cache_file)
                    #     if cap.isOpened():
                    #         return cap
                    # Check in the cache directory
                    cache_file = os.path.join(cache_dir, os.path.basename(target))
                    if os.path.exists(cache_file):
                        # Check for moov atom
                        has_moov = check_moov_atom(cache_file)
                        if has_moov is False:  # Missing moov atom
                            # Create repaired copy in videos directory
                            videos_dir = os.path.dirname(video_path)
                            repaired_path = os.path.join(videos_dir, f"repaired_{os.path.basename(video_path)}")
                            
                            if repair_video(cache_file, repaired_path):
                                # Try the repaired version first
                                cap = cv2.VideoCapture(repaired_path)
                                if cap.isOpened():
                                    if os.path.exists(repaired_path):
                                        try:
                                            os.remove(repaired_path)
                                        except:
                                            pass
                                    return cap
                        
                        # Try the original if repair wasn't needed or failed
                        if verify_video_integrity(cache_file):
                            cap = cv2.VideoCapture(cache_file)
                            if cap.isOpened():
                                return cap
                    else:
                        # Try direct target path
                        if os.path.exists(target) and verify_video_integrity(target):
                            cap = cv2.VideoCapture(target)
                            if cap.isOpened():
                                return cap
                                
                    # If we get here, symlink is broken or points to invalid file
                    # logging.warning(f"Invalid symlink or target file: {video_path} -> {target}")
                    
                # # 2. Try direct file path
                # if os.path.exists(video_path) and verify_video_integrity(video_path):
                #     cap = cv2.VideoCapture(video_path)
                #     if cap.isOpened():
                #         # logging.info(f"Loaded video directly from {video_path}")
                #         return cap

                # 2. Try direct file path
                if os.path.exists(video_path):
                    # Check for moov atom
                    has_moov = check_moov_atom(video_path)
                    if has_moov is False:  # Missing moov atom
                        # Create repaired copy
                        videos_dir = os.path.dirname(video_path)
                        repaired_path = os.path.join(videos_dir, f"repaired_{os.path.basename(video_path)}")
                        
                        if repair_video(video_path, repaired_path):
                            # Try the repaired version first
                            cap = cv2.VideoCapture(repaired_path)
                            if cap.isOpened():
                                if os.path.exists(repaired_path):
                                    try:
                                        os.remove(repaired_path)
                                    except:
                                        pass
                                return cap
                    
                    # Try the original if repair wasn't needed or failed
                    if verify_video_integrity(video_path):
                        cap = cv2.VideoCapture(video_path)
                        if cap.isOpened():
                            return cap
                
                # 3. Resolve Google Drive ID
                gdrive_id = None
                if os.path.islink(video_path):
                    target = os.readlink(video_path)
                    gdrive_id = os.path.basename(target)
                else:
                    # Load mapping if not already loaded
                    if mapping_json is None:
                        mapping_json = HelperFunctions._get_mapping(video_path)
                    if not mapping_json:
                        raise FileNotFoundError(f"No mapping file found for {video_path}")

                    # Get gdrive_id from mapping
                    norm_path = str(Path(video_path))
                    for _, info in mapping_json.items():
                        if info['symlink_path'] == norm_path:
                            gdrive_id = os.path.basename(info['original_path'])
                            break

                if not gdrive_id:
                    raise FileNotFoundError(f"No mapping found for {video_path}")

                # 4. Handle cache and download
                cache_path = None
                if cache_manager.is_cached(gdrive_id):
                    cache_path = cache_manager.get_cache_path(gdrive_id)
                    if not verify_video_integrity(cache_path):
                        # logging.warning(f"Corrupted cache file detected for {gdrive_id}")
                        cache_manager.remove_file(gdrive_id)
                        cache_path = None

                if cache_path is None:
                    # Download file with temporary name
                    temp_download_path = os.path.join(cache_dir, f"temp_{gdrive_id}_{attempt}")
                    drive_ops.download_file(gdrive_id, temp_download_path)
                    
                    # Verify downloaded file
                    if not verify_video_integrity(temp_download_path):
                        os.remove(temp_download_path)
                        raise Exception(f"Downloaded file is corrupted: {gdrive_id}")
                    
                    # Store file hash before caching
                    original_hash = calculate_file_hash(temp_download_path)
                    file_size = os.path.getsize(temp_download_path)
                    
                    # Add to cache
                    cache_manager.add_file(gdrive_id, temp_download_path)
                    cache_path = cache_manager.get_cache_path(gdrive_id)
                    
                    # Verify cached file integrity
                    if not os.path.exists(cache_path) or \
                    calculate_file_hash(cache_path) != original_hash:
                        raise Exception(f"Cache verification failed for {gdrive_id}")
                    
                    # logging.info(f"Successfully downloaded and cached file {gdrive_id}")

                # 5. Update symlink
                if os.path.exists(video_path) or os.path.islink(video_path):
                    os.unlink(video_path)
                
                # Use absolute path with Windows extended-length path syntax
                # abs_cache_path = f"\\\\?\\{os.path.abspath(cache_path)}"
                abs_cache_path = os.path.abspath(cache_path)


                os.symlink(abs_cache_path, video_path)
                # logging.info(f"Created symlink {video_path} -> {abs_cache_path}")

                # 6. Final load attempt
                # cap = cv2.VideoCapture(abs_cache_path)
                # if cap.isOpened() and verify_video_integrity(abs_cache_path):
                #     return cap
                
                # 6. Final load attempt
                has_moov = check_moov_atom(abs_cache_path)
                if has_moov is False:  # Missing moov atom
                    # Create repaired copy
                    videos_dir = os.path.dirname(video_path)
                    repaired_path = os.path.join(videos_dir, f"repaired_{os.path.basename(video_path)}")
                    
                    if repair_video(abs_cache_path, repaired_path):
                        # Try the repaired version first
                        cap = cv2.VideoCapture(repaired_path)
                        if cap.isOpened():
                            if os.path.exists(repaired_path):
                                try:
                                    os.remove(repaired_path)
                                except:
                                    pass
                            return cap

                # Try the original if repair wasn't needed or failed
                cap = cv2.VideoCapture(abs_cache_path)
                if cap.isOpened() and verify_video_integrity(abs_cache_path):
                    return cap
                
                raise Exception(f"Failed to load video after caching: {gdrive_id}")

            except Exception as e:
                last_error = str(e)
                logging.error(f"Error loading video {video_path} (attempt {attempt + 1}): {last_error}")
                
                # Clean up any temporary files
                for temp_file in os.listdir(cache_dir):
                    if temp_file.startswith(f"temp_{gdrive_id}_"):
                        try:
                            os.remove(os.path.join(cache_dir, temp_file))
                        except Exception:
                            pass
                
                if attempt == max_retries - 1:
                    raise Exception(f"Max retries exceeded for loading video {video_path}. Last error: {last_error}")

        raise Exception(f"Failed to load video {video_path} after {max_retries} attempts. Last error: {last_error}")