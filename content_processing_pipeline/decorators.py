import time
import logging
import os
import csv
from datetime import datetime
from functools import wraps
import traceback

import config
from supabase_logger import get_supabase_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Get Supabase logger instance
supabase_logger = get_supabase_logger()

def write_to_csv(filepath, data):
    """Legacy CSV writer - kept for backward compatibility"""
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        User_ID = config.User_ID
        Chat_ID = config.Chat_ID
        Reel_ID = config.Reel_ID
        start_time = time.time()
        result = None
        
        try:
            result = func(*args, **kwargs)
            status = 'success'
        except Exception as e:
            status = 'failed'
            result = str(e)
            raise  # Re-raise the exception after logging
        finally:
            end_time = time.time()
            execution_time = end_time - start_time

            # Format arguments for logging
            args_str = ', '.join([str(arg) for arg in args])
            kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            all_args = f"Args: {args_str} | Kwargs: {kwargs_str}"

            # Log to Supabase
            supabase_data = {
                'user_id': User_ID,
                'chat_id': Chat_ID,
                'reel_id': Reel_ID,
                'function_name': func.__name__,
                'arguments': all_args,
                'execution_time': execution_time,
                'status': status,
                'result': str(result)[:1000] if result else None,  # Limit result size
                'metadata': {
                    'module': func.__module__,
                    'qualname': func.__qualname__ if hasattr(func, '__qualname__') else None
                }
            }
            
            # Try to log to Supabase
            if not supabase_logger.log_execution(supabase_data):
                # Fallback to CSV if Supabase fails
                log_directory = os.path.join("Assets", "Log_Files")
                os.makedirs(log_directory, exist_ok=True)
                
                log_filename = f"{User_ID}_log_{datetime.today().strftime('%Y-%m-%d')}.csv"
                log_filepath = os.path.join(log_directory, log_filename)

                log_data = {
                    "Function Name": func.__name__,
                    "Chat ID": Chat_ID,
                    "Reel ID": Reel_ID,
                    "Arguments": all_args,
                    "Execution Time (s)": execution_time,
                    "Date": datetime.today().strftime('%Y-%m-%d'),
                    "Time": datetime.now().strftime('%H:%M:%S'),
                    "User ID": User_ID,
                    "Status": status
                }

                write_to_csv(log_filepath, log_data)

            logging.info(f"Executed {func.__name__} with {all_args} in {execution_time:.4f} seconds")

        return result
    return wrapper

def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        User_ID = config.User_ID
        Chat_ID = config.Chat_ID
        Reel_ID = config.Reel_ID
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Format arguments for logging
            args_str = ', '.join([str(arg) for arg in args])
            kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            all_args = f"Args: {args_str} | Kwargs: {kwargs_str}"

            # Get stack trace
            stack_trace = traceback.format_exc()

            # Log to Supabase
            error_data = {
                'user_id': User_ID,
                'chat_id': Chat_ID,
                'reel_id': Reel_ID,
                'function_name': func.__name__,
                'arguments': all_args,
                'error_message': str(e),
                'error_type': type(e).__name__,
                'stack_trace': stack_trace,
                'severity': 'ERROR',
                'metadata': {
                    'module': func.__module__,
                    'qualname': func.__qualname__ if hasattr(func, '__qualname__') else None
                }
            }
            
            # Try to log to Supabase
            if not supabase_logger.log_error(error_data):
                # Fallback to CSV if Supabase fails
                error_directory = os.path.join("Assets", "Error_Files")
                os.makedirs(error_directory, exist_ok=True)
                
                error_filename = f"{User_ID}_error_{datetime.today().strftime('%Y-%m-%d')}.csv"
                error_filepath = os.path.join(error_directory, error_filename)

                csv_error_data = {
                    "Function Name": func.__name__,
                    "Chat ID": Chat_ID,
                    "Reel ID": Reel_ID,
                    "Arguments": all_args,
                    "Error Message": str(e),
                    "Error Type": type(e).__name__,
                    "Date": datetime.today().strftime('%Y-%m-%d'),
                    "Time": datetime.now().strftime('%H:%M:%S'),
                    "User ID": User_ID
                }

                write_to_csv(error_filepath, csv_error_data)

            logging.error(f"Error occurred in {func.__name__} with {all_args}: {e}")
            return None

    return wrapper

import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import logging
import json
from pathlib import Path
from functools import wraps

class DriveSync:
    def __init__(self, service_account_file='service-account.json'):
        self.credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        self.service = build('drive', 'v3', credentials=self.credentials)
        self._folder_cache = {}  # Cache for folder IDs

    def get_or_create_chat_folder(self, user_id, chat_id, base_folder_id):
        """Get or create unique folder for User_ID/Chat_ID combination"""
        cache_key = f"{user_id}_{chat_id}"
        
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        user_query = f"name='{user_id}' and mimeType='application/vnd.google-apps.folder' and '{base_folder_id}' in parents"
        user_results = self.service.files().list(
            q=user_query,
            fields='files(id)',
            supportsAllDrives=True
        ).execute()

        if user_results['files']:
            user_folder_id = user_results['files'][0]['id']
            chat_query = f"name='{chat_id}' and mimeType='application/vnd.google-apps.folder' and '{user_folder_id}' in parents"
            chat_results = self.service.files().list(
                q=chat_query,
                fields='files(id)',
                supportsAllDrives=True
            ).execute()

            if chat_results['files']:
                folder_id = chat_results['files'][0]['id']
                self._folder_cache[cache_key] = folder_id
                return folder_id

        user_folder_id = self.find_or_create_folder(user_id, base_folder_id)
        chat_folder_id = self.find_or_create_folder(chat_id, user_folder_id)
        
        self._folder_cache[cache_key] = chat_folder_id
        return chat_folder_id

    # def find_or_create_folder(self, folder_name, parent_id=None):
    #     """Find folder by name and parent, create if doesn't exist"""
    #     query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    #     if parent_id:
    #         query += f" and '{parent_id}' in parents"
            
    #     results = self.service.files().list(
    #         q=query,
    #         fields='files(id, name)',
    #         supportsAllDrives=True,
    #         includeItemsFromAllDrives=True
    #     ).execute()

    #     if results['files']:
    #         return results['files'][0]['id']
        
    #     file_metadata = {
    #         'name': folder_name,
    #         'mimeType': 'application/vnd.google-apps.folder'
    #     }
    #     if parent_id:
    #         file_metadata['parents'] = [parent_id]
            
    #     folder = self.service.files().create(
    #         body=file_metadata,
    #         fields='id',
    #         supportsAllDrives=True
    #     ).execute()
        
    #     return folder['id']

    def upload_file(self, local_path, filename, parent_id):
        """Upload or update file in Drive"""
        # Skip symlinks
        if os.path.islink(local_path):
            logging.info(f"Skipping symlink: {filename}")
            return None

        query = f"name='{filename}' and '{parent_id}' in parents"
        results = self.service.files().list(
            q=query,
            fields='files(id)',
            supportsAllDrives=True
        ).execute()

        file_metadata = {
            'name': filename,
            'parents': [parent_id]
        }
        media = MediaFileUpload(local_path, resumable=True)

        if results['files']:
            file_id = results['files'][0]['id']
            self.service.files().update(
                fileId=file_id,
                media_body=media,
                supportsAllDrives=True
            ).execute()
            return file_id
        else:
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()
            return file['id']

    # def download_file(self, file_id, local_path):
    #     """Download file from Drive"""
    #     if os.path.exists(local_path):
    #         logging.info(f"File exists locally, skipping download: {local_path}")
    #         return
            
    #     os.makedirs(os.path.dirname(local_path), exist_ok=True)
    #     request = self.service.files().get_media(fileId=file_id)
    #     with open(local_path, 'wb') as f:
    #         downloader = MediaIoBaseDownload(f, request)
    #         done = False
    #         while not done:
    #             _, done = downloader.next_chunk()
    
    # Update the download_file method to handle shared drives
    def download_file(self, file_id, local_path):
        """Download file from Drive"""
        if os.path.exists(local_path):
            logging.info(f"File exists locally, skipping download: {local_path}")
            return
            
        try:    
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            request = self.service.files().get_media(
                fileId=file_id,
                supportsAllDrives=True  # Add this parameter
            )
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            logging.info(f"Successfully downloaded: {local_path}")
        except Exception as e:
            logging.error(f"Error downloading file {file_id}: {str(e)}")
            raise
    
    def sync_from_drive_to_directory(self, folder_id, local_dir):
        """Download contents from Drive folder to local directory"""
        os.makedirs(local_dir, exist_ok=True)
        
        # List all items in the folder
        query = f"'{folder_id}' in parents"
        results = self.service.files().list(
            q=query,
            fields='files(id, name, mimeType)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        # Process each item
        for item in results.get('files', []):
            local_path = os.path.join(local_dir, item['name'])
            
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                # Recursively sync subfolder
                self.sync_from_drive_to_directory(item['id'], local_path)
            else:
                # Download file if it doesn't exist locally
                if not os.path.exists(local_path):
                    self.download_file(item['id'], local_path)

    # def sync_directory_to_drive(self, local_dir, parent_id):
    #     """Recursively sync a directory to Drive"""
    #     if not os.path.isdir(local_dir):
    #         return

    #     folder_name = os.path.basename(local_dir)
    #     folder_id = self.find_or_create_folder(folder_name, parent_id)

    #     for item in os.listdir(local_dir):
    #         item_path = os.path.join(local_dir, item)
    #         if os.path.isfile(item_path) and not os.path.islink(item_path):
    #             self.upload_file(item_path, item, folder_id)
    #         elif os.path.isdir(item_path):
    #             self.sync_directory_to_drive(item_path, folder_id)

    def sync_directory_to_drive(self, local_dir, parent_id):
        """Recursively sync a directory to Drive ensuring exact folder structure"""
        if not os.path.isdir(local_dir):
            return

        # Get the full local directory structure
        local_structure = {}
        for root, dirs, files in os.walk(local_dir):
            rel_path = os.path.relpath(root, local_dir)
            if rel_path == '.':
                rel_path = ''
            local_structure[rel_path] = {
                'dirs': dirs,
                'files': files
            }

        # First, create all directories to match structure
        folder_ids = {''} # Root folder
        for rel_path in sorted(local_structure.keys()):
            if rel_path == '':
                folder_ids[rel_path] = parent_id
                continue
                
            parent_path = os.path.dirname(rel_path) if rel_path else ''
            folder_name = os.path.basename(rel_path) if rel_path else os.path.basename(local_dir)
            
            # Create folder with exact name and under exact parent
            folder_id = self.find_or_create_folder(folder_name, folder_ids[parent_path])
            folder_ids[rel_path] = folder_id

        # Then sync all files maintaining structure
        for rel_path, content in local_structure.items():
            current_folder_id = folder_ids[rel_path]
            
            for file_name in content['files']:
                if not os.path.islink(os.path.join(local_dir, rel_path, file_name)):
                    file_path = os.path.join(local_dir, rel_path, file_name)
                    self.upload_file(file_path, file_name, current_folder_id)

    def find_or_create_folder(self, folder_name, parent_id):
        """Find folder by exact name and parent, create if doesn't exist"""
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents"
                
        results = self.service.files().list(
            q=query,
            fields='files(id, name)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()

        if results['files']:
            # If multiple folders somehow exist with same name under same parent, use the first one
            if len(results['files']) > 1:
                logging.warning(f"Multiple folders named '{folder_name}' found under same parent. Using first one.")
            return results['files'][0]['id']
        
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
                
        folder = self.service.files().create(
            body=file_metadata,
            fields='id',
            supportsAllDrives=True
        ).execute()
        
        return folder['id']

def sync_with_drive(inputs=None, outputs=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import config
            User_ID = config.User_ID
            Chat_ID = config.Chat_ID
            BASE_DRIVE_FOLDER = "1be5p41JtvBbSxKpBaxrcotet0RZCzt5Y"

            drive_sync = DriveSync()

            # def recreate_structure_from_mapping(mapping_file_path, base_path):
            #     """Recreate directory structure and symlinks from mapping file"""
            #     try:
            #         with open(mapping_file_path, 'r') as f:
            #             mapping_data = json.load(f)

            #         # Create cache directory
            #         cache_dir = Path("/tmp/gdrive_cache")
            #         cache_dir.mkdir(parents=True, exist_ok=True)

            #         # Create media directories
            #         media_dir = os.path.join(base_path, 'Media')
            #         for media_type in ["Images", "Videos", "Audio", "Documents", "Other"]:
            #             os.makedirs(os.path.join(media_dir, media_type), exist_ok=True)

            #         # Create symlinks and cache files
            #         for filename, file_info in mapping_data.items():
            #             if 'cache_path' in file_info:
            #                 cache_file = Path(file_info['cache_path'])
            #                 cache_file.parent.mkdir(parents=True, exist_ok=True)
            #                 if not cache_file.exists():
            #                     cache_file.touch()

            #             if 'symlink_path' in file_info:
            #                 symlink_path = Path(file_info['symlink_path'])
            #                 symlink_path.parent.mkdir(parents=True, exist_ok=True)
            #                 if not symlink_path.exists() and not os.path.islink(str(symlink_path)):
            #                     os.symlink(str(cache_file), str(symlink_path))

            #     except Exception as e:
            #         logging.error(f"Error recreating structure from mapping: {str(e)}")
            #         raise
            def recreate_structure_from_mapping(mapping_file_path, base_path):
                """Recreate directory structure and symlinks from mapping file"""
                try:
                    # Get User_ID and Chat_ID directly from config
                    import config
                    User_ID = config.User_ID
                    Chat_ID = config.Chat_ID
                    
                    with open(mapping_file_path, 'r') as f:
                        mapping_data = json.load(f)

                    # Determine appropriate cache directory based on environment
                    if os.path.exists("/tmp") and os.access("/tmp", os.W_OK):
                        # Docker or Linux environment
                        cache_dir = Path("/tmp/gdrive_cache")
                    else:
                        # Windows or other environment - use local cache
                        cache_dir = Path(os.path.join(os.getcwd(), "cache"))
                    
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    logging.info(f"Using cache directory: {cache_dir}")

                    # Create consistent base path using config values
                    cwd = os.getcwd()
                    local_base = os.path.join(User_ID, Chat_ID)
                    media_dir = os.path.join(local_base, 'Media')
                    
                    # Ensure we have write permission at the base_path
                    try:
                        # Test if we can write to the directory
                        test_file = os.path.join(local_base, ".write_test")
                        with open(test_file, 'w') as f:
                            f.write("")
                        os.remove(test_file)
                        
                        # Create media directories
                        for media_type in ["Images", "Videos", "Audio", "Documents", "Other"]:
                            os.makedirs(os.path.join(media_dir, media_type), exist_ok=True)
                        
                        logging.info(f"Created media directories in: {media_dir}")
                    except Exception as e:
                        logging.warning(f"Cannot write to {local_base}, using alternative path: {e}")
                        # Use a writable directory but maintain User_ID/Chat_ID structure
                        alt_base = os.path.join(cwd, User_ID, Chat_ID)
                        media_dir = os.path.join(alt_base, 'Media')
                        
                        for media_type in ["Images", "Videos", "Audio", "Documents", "Other"]:
                            os.makedirs(os.path.join(media_dir, media_type), exist_ok=True)
                        
                        logging.info(f"Created media directories in alternative location: {media_dir}")

                    # Create symlinks and cache files
                    for filename, file_info in mapping_data.items():
                        if 'cache_path' in file_info:
                            # Make sure cache path is in our controlled cache directory
                            orig_cache_path = file_info['cache_path']
                            if not str(orig_cache_path).startswith(str(cache_dir)):
                                # Redirect to our controlled cache directory
                                cache_path = os.path.join(str(cache_dir), os.path.basename(orig_cache_path))
                            else:
                                cache_path = orig_cache_path
                            
                            cache_file = Path(cache_path)
                            cache_file.parent.mkdir(parents=True, exist_ok=True)
                            if not cache_file.exists():
                                cache_file.touch()
                                logging.info(f"Created cache file: {cache_file}")

                                # Check if this is a default file
                                if file_info.get('is_default', False):
                                    default_files = {"Images": "default_blank.png", "Videos": "default_video.mp4"}
                                    media_type = file_info.get('media_type')
                                    if media_type in default_files:
                                        default_src = Path("Assets/defaults") / default_files[media_type]
                                        if default_src.exists():
                                            import shutil
                                            shutil.copy2(default_src, cache_file)
                                            logging.info(f"Copied default {media_type} content to cache file")

                        if 'symlink_path' in file_info:
                            # Adjust symlink path to use config-based paths
                            orig_symlink_path = file_info['symlink_path']
                            
                            # For absolute or app-based paths, redirect to our local structure
                            if os.path.isabs(orig_symlink_path) or '/app' in orig_symlink_path:
                                # Extract just the filename and use proper media type folder
                                filename = os.path.basename(orig_symlink_path)
                                
                                # Determine media type from original path
                                media_type = "Other"
                                for folder in ["Images", "Videos", "Audio", "Documents"]:
                                    if f"/Media/{folder}/" in orig_symlink_path:
                                        media_type = folder
                                        break
                                
                                # Use consistent path with config values
                                symlink_path = Path(os.path.join(media_dir, media_type, filename))
                            else:
                                symlink_path = Path(orig_symlink_path)
                            
                            symlink_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            if not symlink_path.exists() and not os.path.islink(str(symlink_path)):
                                try:
                                    os.symlink(str(cache_file), str(symlink_path))
                                    logging.info(f"Created symlink: {symlink_path} -> {cache_file}")
                                except OSError as se:
                                    logging.error(f"Failed to create symlink: {se}")

                except Exception as e:
                    logging.error(f"Error recreating structure from mapping: {str(e)}")
                    raise
            
            def sync_path(path, drive_folder_id, download=True):
                """Sync path while maintaining exact folder structure"""
                local_base = os.path.join(User_ID, Chat_ID)
                full_path = os.path.join(local_base, path)
                
                # Split the path into components
                path_parts = Path(path).parts
                current_folder_id = drive_folder_id
                
                # Build folder structure one level at a time
                for i, folder in enumerate(path_parts[:-1]):  # Skip the last part if it's a file
                    current_folder_id = drive_sync.find_or_create_folder(folder, current_folder_id)
                
                if download:
                    if os.path.isdir(full_path):
                        drive_sync.sync_from_drive_to_directory(current_folder_id, full_path)
                    else:
                        file_name = path_parts[-1]
                        query = f"name='{file_name}' and '{current_folder_id}' in parents"
                        results = drive_sync.service.files().list(
                            q=query,
                            fields='files(id)',
                            supportsAllDrives=True,
                            includeItemsFromAllDrives=True  # Add this parameter
                        ).execute()
                        
                        if results['files']:
                            os.makedirs(os.path.dirname(full_path), exist_ok=True)
                            drive_sync.download_file(results['files'][0]['id'], full_path)
                            
                            # Handle mapping file special case
                            if file_name == 'filename_mapping.json':
                                recreate_structure_from_mapping(full_path, local_base)
                else:
                    if os.path.isdir(full_path):
                        drive_sync.sync_directory_to_drive(full_path, current_folder_id)
                    elif os.path.isfile(full_path) and not os.path.islink(full_path):
                        drive_sync.upload_file(full_path, os.path.basename(full_path), current_folder_id)
                        
            # def sync_path(path, drive_folder_id, download=True):
            #     local_base = os.path.join(User_ID, Chat_ID)
            #     full_path = os.path.join(local_base, path)

            #     if download:
            #         mapping_file_exists = os.path.exists(
            #             os.path.join(local_base, 'Media', 'filename_mapping.json')
            #         )

            #         if os.path.isdir(full_path):
            #             drive_sync.sync_from_drive_to_directory(drive_folder_id, full_path)
            #         else:
            #             query = f"name='{os.path.basename(path)}' and '{drive_folder_id}' in parents"
            #             results = drive_sync.service.files().list(
            #                 q=query,
            #                 fields='files(id)',
            #                 supportsAllDrives=True
            #             ).execute()
            #             if results['files']:
            #                 os.makedirs(os.path.dirname(full_path), exist_ok=True)
            #                 drive_sync.download_file(results['files'][0]['id'], full_path)

            #                 # Handle mapping file special case
            #                 if os.path.basename(path) == 'filename_mapping.json' and not mapping_file_exists:
            #                     recreate_structure_from_mapping(full_path, local_base)
            #     else:
            #         if os.path.isdir(full_path):
            #             drive_sync.sync_directory_to_drive(full_path, drive_folder_id)
            #         elif os.path.isfile(full_path) and not os.path.islink(full_path):
            #             drive_sync.upload_file(full_path, os.path.basename(full_path), drive_folder_id)

            try:
                # Use cached folder lookup
                chat_folder_id = drive_sync.get_or_create_chat_folder(
                    User_ID, 
                    Chat_ID, 
                    BASE_DRIVE_FOLDER
                )

                if inputs:
                    for path in inputs:
                        sync_path(path, chat_folder_id, download=True)

                result = func(*args, **kwargs)

                if outputs:
                    for path in outputs:
                        sync_path(path, chat_folder_id, download=False)

                return result

            except Exception as e:
                logging.error(f"Drive sync error: {str(e)}")
                raise

        return wrapper
    return decorator


from functools import wraps
import json
from datetime import datetime
import logging
import config
from typing import Optional, Dict, Any
import traceback

logger = logging.getLogger(__name__)

# Global supabase instance
_supabase_wrapper = None

def get_supabase_instance():
    """Get or create global Supabase instance"""
    global _supabase_wrapper
    if _supabase_wrapper is None:
        from SupabaseWrapper import SupabaseWrapper
        _supabase_wrapper = SupabaseWrapper()
    return _supabase_wrapper.supabase

def log_whatsapp_activity(func):
    """
    Decorator to log WhatsApp message activities to Supabase.
    Uses the global SupabaseLogger instance.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.utcnow()
        execution_start = time.time()
        
        try:
            # Determine message type with safe attribute checking
            message_type = "text"
            if hasattr(config, 'whatsapp_template_name') and config.whatsapp_template_name:
                message_type = "template"
            
            # Get recipient with safe fallback
            recipient = getattr(config, 'whatsapp_recipient', None)
            if not recipient:
                logger.error("No recipient found in config")
                raise ValueError("No recipient specified")
            
            # Prepare base log entry with safe defaults
            log_entry = {
                'user_id': getattr(config, 'User_ID', 'unknown'),
                'chat_id': getattr(config, 'Chat_ID', 'unknown'),
                'reel_id': getattr(config, 'Reel_ID', None),
                'function_name': func.__name__,
                'recipient': recipient,
                'message_type': message_type,
                'message_content': getattr(config, 'whatsapp_message', '') if message_type == "text" else None,
                'template_name': getattr(config, 'whatsapp_template_name', '') if message_type == "template" else None,
                'template_params': config.whatsapp_template_params if hasattr(config, 'whatsapp_template_params') else None,
                'media_url': getattr(config, 'whatsapp_video_url', None),
                'status': 'pending'
            }

            try:
                # Execute the WhatsApp send function
                response = func(*args, **kwargs)
                
                # Safely get WhatsApp message ID
                messages = response.get("messages", []) if isinstance(response, dict) else []
                whatsapp_id = messages[0].get("id") if messages else None
                
                # Update log entry with success details
                log_entry['status'] = 'success'
                log_entry['response'] = response
                log_entry['metadata'] = {
                    'whatsapp_message_id': whatsapp_id,
                    'execution_time': time.time() - execution_start,
                    'completed_at': datetime.utcnow().isoformat()
                }
                
                logger.info(f"Successfully sent WhatsApp {message_type} message")
                
            except Exception as e:
                # Get full traceback for error logging
                error_traceback = traceback.format_exc()
                
                # Update log entry with error details
                log_entry['status'] = 'failed'
                log_entry['response'] = {'error': str(e), 'traceback': error_traceback}
                log_entry['metadata'] = {
                    'execution_time': time.time() - execution_start,
                    'completed_at': datetime.utcnow().isoformat()
                }
                
                logger.error(f"Failed to send WhatsApp message: {str(e)}\n{error_traceback}")
                raise
                
            finally:
                try:
                    # Try to log to Supabase using the new logger
                    if not supabase_logger.log_whatsapp_activity(log_entry):
                        # Fallback to old Supabase instance if new logger fails
                        try:
                            supabase = get_supabase_instance()
                            result = supabase.table("whatsapp_logs").insert(log_entry).execute()
                            if result and result.data:
                                logger.info("Successfully logged WhatsApp activity to Supabase (fallback)")
                        except:
                            # Final fallback to CSV
                            log_directory = os.path.join("Assets", "WhatsApp_Logs")
                            os.makedirs(log_directory, exist_ok=True)
                            
                            log_filename = f"{log_entry['user_id']}_whatsapp_{datetime.today().strftime('%Y-%m-%d')}.csv"
                            log_filepath = os.path.join(log_directory, log_filename)
                            
                            csv_data = {
                                "Function Name": func.__name__,
                                "User ID": log_entry['user_id'],
                                "Chat ID": log_entry['chat_id'],
                                "Reel ID": log_entry['reel_id'],
                                "Recipient": log_entry['recipient'],
                                "Message Type": log_entry['message_type'],
                                "Status": log_entry['status'],
                                "Date": datetime.today().strftime('%Y-%m-%d'),
                                "Time": datetime.now().strftime('%H:%M:%S')
                            }
                            
                            write_to_csv(log_filepath, csv_data)
                            logger.info("Logged WhatsApp activity to CSV (final fallback)")
                    else:
                        logger.info("Successfully logged WhatsApp activity to Supabase")
                        
                except Exception as e:
                    error_trace = traceback.format_exc()
                    logger.error(f"Failed to log WhatsApp activity: {str(e)}\n{error_trace}")
                
                # Return the original response if it exists
                if 'response' in locals():
                    return response
                    
        except Exception as outer_e:
            logger.error(f"Critical error in decorator: {str(outer_e)}\n{traceback.format_exc()}")
            # Still try to execute the original function if everything else fails
            return func(*args, **kwargs)

    return wrapper