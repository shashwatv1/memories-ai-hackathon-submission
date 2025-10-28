
import sys
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import json
import functools
from tqdm import tqdm
from SupabaseWrapper import SupabaseWrapper
import config
import jushn
import GoogleServiceAPI
import pipeline
import video_processor
import video_processor_optimized
import image_processor
import clustering
import Spoty
import whatsapp 
from docker_status_reporter import DockerStatusReporter
import whatsapp_event_processor
from whatsapp_event_processor import WhatsAppEventProcessor
from lead_status_manager import LeadStatusManager
from GdriveSync import startup_sync , recreate_media_structure_from_drive_mapping , set_config_links  # Add this import
from GoogleAPI import send_processing_complete_email, send_reel_creation_complete_email, send_reel_creation_complete_email_new
import auto_json_generator
import media_catalog

import tensorflow as tf
import keras
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
# os.environ["TF_USE_LEGACY_KERAS"] = "1"


print(f"Is GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

def safe_execute_function(func):
    """Wrapper to safely execute any function and continue on error"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"❌ Error in {func.__name__}: {str(e)}")
            print("✳️ Continuing to next task...")
            return False
    return wrapper

def with_progress(func):
    """Wrapper to add progress bar to functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            with tqdm(total=100, desc=f"Running {func.__name__}") as pbar:
                result = func(*args, **kwargs)
                pbar.update(100)
                return result
        except Exception as e:
            print(f"❌ Progress bar error in {func.__name__}: {str(e)}")
            return func(*args, **kwargs)
    return wrapper

@safe_execute_function
def print_service_completion():
    """Print completion message for the service"""
    service_name = os.environ.get('SERVICE_NAME', 'unknown')
    print("\n" + "="*60)
    print(f"✅ All functions for {service_name} completed successfully!")
    print(f"You can now proceed to the next service in the pipeline.")
    print("="*60 + "\n")

@safe_execute_function
def safe_path_join(*args):
    """Safely join path components, replacing None with empty string"""
    try:
        valid_parts = [str(arg) for arg in args if arg is not None]
        return os.path.join(*valid_parts) if valid_parts else ""
    except Exception as e:
        print(f"❌ Path join error: {str(e)}")
        return ""

@safe_execute_function
def print_config():
    """Print current configuration state"""
    print("\n=== Config State ===")
    config_fields = [
        'User_ID', 'Chat_ID', 'Reel_ID', 'Container_ID', 'user_prompt',
        'rating_filter', 'process_type', 'embeding_path', 'links', 'songname',
        'user_query', 'whatsapp_recipient', 'whatsapp_template_name',
        'whatsapp_template_params', 'whatsapp_message', 'whatsapp_video_url'
    ]
    
    for field in config_fields:
        try:
            value = getattr(config, field, None)
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except:
                    pass
            print(f"{field}: {value}")
        except Exception as e:
            print(f"❌ Error printing {field}: {str(e)}")
    print("=" * 50)

# @log_execution
# @handle_errors
@safe_execute_function
def wait_five_minutes():
    """Wait 5 minutes before starting reel processing"""
    print("⏳ Waiting 5 minutes before starting  processing...")
    time.sleep(150)
    print("✅ Wait complete, proceeding with processing")

# Combined service functions dictionary
SERVICE_FUNCTIONS = {

    'preprocessing_service': {
         # setup_service functions
        'create_directories': safe_execute_function(jushn.create_directories),
        'wait_five_minutes': wait_five_minutes,
        'gdrive_sync': safe_execute_function(startup_sync),
        'set_config_links': safe_execute_function(set_config_links),
        'process_videos_with_segmentation': safe_execute_function(video_processor.process_videos_with_segmentation),
        'process_segments_with_audio_check': safe_execute_function(video_processor.process_segments_with_audio_check),
        'create_enhanced_media_mapping': safe_execute_function(GoogleServiceAPI.create_enhanced_media_mapping),
        # 'create_media_symlinks': safe_execute_function(GoogleServiceAPI.create_media_symlinks_service),
        'create_clusters': safe_execute_function(jushn.create_clusters),
        'process_clusters_analysis': safe_execute_function(jushn.process_clusters_with_detailed_analysis),
        'auto_label_clusters': safe_execute_function(jushn.auto_label_clusters),
    
        
        # video_processing_service functions
        'process_videos_katna': safe_execute_function(jushn.process_videos_with_katna),
        'assign_video_clusters': safe_execute_function(jushn.assign_video_to_clusters),
        'process_videos_parallel': safe_execute_function(jushn.process_videos_parallel),
        
        # parallel_processing_service functions
        'assign_images_clusters': safe_execute_function(jushn.assign_images_to_clusters),
        'process_images_parallel': safe_execute_function(jushn.process_images_parallel),
        
        # cluster_analysis_service functions
        'find_similar_clusters': safe_execute_function(jushn.find_similar_image_clusters),
        'add_ratings_clusters': safe_execute_function(jushn.add_ratings_to_clusters),
        'create_tiered_embeddings': safe_execute_function(jushn.create_tiered_embeddings),
        'process_reel_clusters': safe_execute_function(jushn.process_all_reel_clusters),
        'adjust_clusters': safe_execute_function(clustering.adjust_clusters),
        'rename_cluster_files': safe_execute_function(clustering.rename_cluster_files),

        # Media catalog functions
        'create_media_catalog': safe_execute_function(media_catalog.create_media_catalog),
        'create_shortened_descriptions': safe_execute_function(media_catalog.create_shortened_descriptions),

        'send_processing_complete_email': safe_execute_function(send_processing_complete_email)

    },

    'create_reel_service': {
        'create_directories': safe_execute_function(jushn.create_directories),
        'wait_five_minutes': wait_five_minutes,
        'delete_all_json_files': safe_execute_function(auto_json_generator.delete_all_json_files),
        'gdrive_sync': safe_execute_function(startup_sync),
        'recreate_media_structure_from_drive_mapping': safe_execute_function(recreate_media_structure_from_drive_mapping),
        'generate_all_missing_json_files': safe_execute_function(auto_json_generator.generate_all_missing_json_files),
        'assemble_video_template': safe_execute_function(pipeline.assemble_video_template),
        'validate_and_adjust_template_timing': safe_execute_function(pipeline.validate_and_adjust_template_timing),
        'apply_transition_system': safe_execute_function(pipeline.apply_transition_system),
        'create_frames_from_template_image': safe_execute_function(pipeline.create_frames_from_template_image),
        'create_frames_from_template_video': safe_execute_function(pipeline.create_frames_from_template_video),
        'combine_frames_to_video': safe_execute_function(pipeline.combine_frames_to_video),
        'add_audio_to_video_and_cleanup': safe_execute_function(pipeline.add_audio_to_video_and_cleanup),
        'upload_videos_to_shared_drive': safe_execute_function(GoogleServiceAPI.upload_videos_to_shared_drive),
        # 'set_config_links': safe_execute_function(set_config_links),
        # 'move_to_target_GDrive_service': safe_execute_function(GoogleServiceAPI.move_to_target_GDrive_service),
        'send_reel_creation_complete_email': safe_execute_function(send_reel_creation_complete_email_new)
    },

    'create_collage_service': {
        'create_directories': safe_execute_function(jushn.create_directories)
    },

    'all_service': {
        # setup_service functions
        'create_directories': safe_execute_function(jushn.create_directories),
        'process_videos_with_segmentation': safe_execute_function(video_processor.process_videos_with_segmentation),
        'process_segments_with_audio_check': safe_execute_function(video_processor.process_segments_with_audio_check),
        'create_enhanced_media_mapping': safe_execute_function(GoogleServiceAPI.create_enhanced_media_mapping),
        # 'create_media_symlinks': safe_execute_function(GoogleServiceAPI.create_media_symlinks_service),
        'create_clusters': safe_execute_function(jushn.create_clusters),
        'process_clusters_analysis': safe_execute_function(jushn.process_clusters_with_detailed_analysis),
        'auto_label_clusters': safe_execute_function(jushn.auto_label_clusters),
        
        
        # video_processing_service functions
        'process_videos_katna': safe_execute_function(jushn.process_videos_with_katna),
        'assign_video_clusters': safe_execute_function(jushn.assign_video_to_clusters),
        'process_videos_parallel': safe_execute_function(jushn.process_videos_parallel),
        
        # parallel_processing_service functions
        'assign_images_clusters': safe_execute_function(jushn.assign_images_to_clusters),
        'process_images_parallel': safe_execute_function(jushn.process_images_parallel),
        
        # cluster_analysis_service functions
        'find_similar_clusters': safe_execute_function(jushn.find_similar_image_clusters),
        'add_ratings_clusters': safe_execute_function(jushn.add_ratings_to_clusters),
        'create_tiered_embeddings': safe_execute_function(jushn.create_tiered_embeddings),
        'process_reel_clusters': safe_execute_function(jushn.process_all_reel_clusters),
        'adjust_clusters': safe_execute_function(clustering.adjust_clusters),
        'rename_cluster_files': safe_execute_function(clustering.rename_cluster_files),
        
        # results_service functions
        'download_results': safe_execute_function(GoogleServiceAPI.download_spoty_results),
        # 'process_reel_clusters_results': safe_execute_function(pipeline.process_reel_clusters),
        # 'move_to_gdrive': safe_execute_function(GoogleServiceAPI.move_to_target_GDrive_service)
        'initialize_spotify_environment': safe_execute_function(GoogleServiceAPI.initialize_spotify_environment),
        'insta_login': safe_execute_function(image_processor.insta_login),
        'select_and_run_pipeline': safe_execute_function(pipeline.select_and_run_pipeline),
        'upload_videos_to_shared_drive': safe_execute_function(GoogleServiceAPI.upload_videos_to_shared_drive)

    },
    'setup_service': {
        'create_directories': safe_execute_function(jushn.create_directories),
        'process_videos_with_segmentation': safe_execute_function(video_processor.process_videos_with_segmentation),
        'process_segments_with_audio_check': safe_execute_function(video_processor.process_segments_with_audio_check),
        'create_enhanced_media_mapping': safe_execute_function(GoogleServiceAPI.create_enhanced_media_mapping),
        # 'create_media_symlinks': safe_execute_function(GoogleServiceAPI.create_media_symlinks_service),
        'create_clusters': safe_execute_function(jushn.create_clusters),
        'process_clusters_analysis': safe_execute_function(jushn.process_clusters_with_detailed_analysis),
        'auto_label_clusters': safe_execute_function(jushn.auto_label_clusters),


    },
    'video_processing_service': {
        'process_videos_katna': safe_execute_function(jushn.process_videos_with_katna),
        'assign_video_clusters': safe_execute_function(jushn.assign_video_to_clusters),
        'process_videos_parallel': safe_execute_function(jushn.process_videos_parallel)
    },
    'parallel_processing_service': {
        'assign_images_clusters': safe_execute_function(jushn.assign_images_to_clusters),
        'process_images_parallel': safe_execute_function(jushn.process_images_parallel)
    },
    'cluster_analysis_service': {
        'find_similar_clusters': safe_execute_function(jushn.find_similar_image_clusters),
        'add_ratings_clusters': safe_execute_function(jushn.add_ratings_to_clusters),
        'create_tiered_embeddings': safe_execute_function(jushn.create_tiered_embeddings),
        'process_reel_clusters': safe_execute_function(jushn.process_all_reel_clusters),
        'adjust_clusters': safe_execute_function(clustering.adjust_clusters),
        'rename_cluster_files': safe_execute_function(clustering.rename_cluster_files)
    },
    'results_service': {
        'download_results': safe_execute_function(GoogleServiceAPI.download_spoty_results),
        'initialize_spotify_environment': safe_execute_function(GoogleServiceAPI.initialize_spotify_environment),
        'insta_login': safe_execute_function(image_processor.insta_login),
        # 'process_reel_clusters_results': safe_execute_function(pipeline.process_reel_clusters),
        'select_and_run_pipeline': safe_execute_function(pipeline.select_and_run_pipeline),
        'upload_videos_to_shared_drive': safe_execute_function(GoogleServiceAPI.upload_videos_to_shared_drive)
    },
    'spoty_service': {
        'download_songs': safe_execute_function(GoogleServiceAPI.download_songs_for_spoty),
        'initialize_spotify': safe_execute_function(Spoty.initialize_spotify),
        'process_analysis': safe_execute_function(Spoty.process_analysis),
        'process_bars': safe_execute_function(Spoty.process_bars),
        'process_lyrics': safe_execute_function(Spoty.process_lyrics),
        'process_info': safe_execute_function(Spoty.process_info),
        'prepare_embeddings': safe_execute_function(Spoty.prepare_song_embedding_data),
        'update_json_with_analysis': safe_execute_function(Spoty.update_json_with_analysis_data),
        'process_embeddings': safe_execute_function(Spoty.process_song_embeddings),
        'sync_pkl_timestamp': safe_execute_function(Spoty.sync_pkl_timestamp),
        'upload_results': safe_execute_function(GoogleServiceAPI.upload_spoty_results)
    },
    'whatsapp_service': {
        'process_whatsapp': safe_execute_function(whatsapp.send_whatsapp)
    },
    # Add to the SERVICE_FUNCTIONS dictionary
    'whatsapp_event_service': {
        'process_whatsapp_events': safe_execute_function(WhatsAppEventProcessor.run)
    },
    'lead_status_service': {
        'process_lead_status': safe_execute_function(LeadStatusManager.process_queue)
    }
}

@safe_execute_function
def get_current_container_name():
    """Get the current container name from environment"""
    try:
        if os.path.exists('/.dockerenv'):
            service_name = os.environ.get('SERVICE_NAME', 'unknown')
            with open('/etc/hostname', 'r') as f:
                container_id = f.read().strip()[:12]
            container_name = f"{service_name}-{container_id}"
            print(f"Container Name: {container_name}")
            return container_name
        return "test"
    except Exception as e:
        print(f"❌ Error detecting container: {str(e)}")
        return "test"

@safe_execute_function
def container_check(row):
    """Check if the task is for this container"""
    try:
        current_container = get_current_container_name()
        print(f"Checking container match: {row.get('Container_Name')} vs {current_container}")
        return row.get('Container_Name') == current_container
    except Exception as e:
        print(f"❌ Container check error: {str(e)}")
        return False

@safe_execute_function
def get_service_functions():
    """Get list of functions for the current service"""
    try:
        service_name = os.environ.get('SERVICE_NAME')
        if not service_name or service_name not in SERVICE_FUNCTIONS:
            print(f"⚠️ Invalid SERVICE_NAME: {service_name}")
            return []
        
        functions = [with_progress(func) for func in SERVICE_FUNCTIONS[service_name].values()]
        print(f"✅ Loaded {len(functions)} functions for service: {service_name}")
        return functions
    except Exception as e:
        print(f"❌ Error getting service functions: {str(e)}")
        return []

@safe_execute_function
def process_row_with_config():
    """Process row with configuration initialization"""
    try:
        # Parse JSON strings if needed
        json_fields = ['links', 'whatsapp_template_params']
        for field in json_fields:
            if hasattr(config, field) and isinstance(getattr(config, field), str):
                try:
                    setattr(config, field, json.loads(getattr(config, field)))
                except:
                    setattr(config, field, [] if field in ['links', 'whatsapp_template_params'] else {})

        # Set Chat_ID to User_ID if not provided
        if not config.Chat_ID:
            config.Chat_ID = config.User_ID

        # Create directories
        base_path = safe_path_join(config.User_ID, config.Chat_ID)
        if base_path:
            os.makedirs(base_path, exist_ok=True)
            print(f"Created directory: {base_path}")
            
            media_path = safe_path_join(base_path, 'Media')
            os.makedirs(media_path, exist_ok=True)
            print(f"Created media directory: {media_path}")
        
        print_config()
        return True
        
    except Exception as e:
        print(f"❌ Error in process_row_with_config: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        service_name = os.environ.get('SERVICE_NAME', 'unknown')
        print(f"\n🚀 Starting {service_name} service...")
        
        current_container_name = get_current_container_name()
        print(f"Using Container: {current_container_name}")

        status_reporter = DockerStatusReporter(current_container_name)
        status_reporter.start()

        # Complete mapping for all possible fields
        mapping = {
            'User_ID': 'User_ID',
            'Chat_ID': 'Chat_ID',
            'Reel_ID': 'Reel_ID',
            'Container_Name': 'Container_Name',
            'Container_ID': 'Container_ID',
            'user_prompt': 'user_prompt',
            'rating_filter': 'rating_filter',
            'process_type': 'process_type',
            'embeding_path': 'embeding_path',
            'links': 'links',
            'songname': 'songname',
            'user_query': 'user_query',
            'whatsapp_recipient': 'whatsapp_recipient',
            'whatsapp_template_name': 'whatsapp_template_name',
            'whatsapp_template_params': 'whatsapp_template_params',
            'whatsapp_message': 'whatsapp_message',
            'whatsapp_video_url': 'whatsapp_video_url',
            'is_processed': 'is_processed',
            'error_message': 'error_message'
        }

        print("Initializing Supabase wrapper...")
        supabase_wrapper = SupabaseWrapper()

        service_functions = get_service_functions()
        all_functions = [process_row_with_config] + service_functions + [print_service_completion]

        print("Registering docker_tasks table...")
        supabase_wrapper.register_table(
            'docker_tasks_prod',
            mapping,
            all_functions,
            pre_check=container_check
        )

        print("Starting listener...")
        supabase_wrapper.start_listener(interval=0.5)

        # while True:
        #     try:
        #         time.sleep(1)
        #     except Exception as e:
        #         print(f"⚠️ Error in main loop: {str(e)}")
        #         print("Attempting to continue...")
        #         time.sleep(5)
        counter = 0  # Initialize counter outside the loop

        while True:
            try:
                # Flush stdout every 60 seconds
                counter += 1
                if counter >= 60:
                    sys.stdout.flush()
                    counter = 0
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ Error in main loop: {str(e)}")
                print("Attempting to continue...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n⚠️ Received shutdown signal...")
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
    finally:
        print("⚠️ Attempting graceful shutdown...")
        try:
            supabase_wrapper.stop()
        except Exception as e:
            print(f"❌ Error during shutdown: {str(e)}")
        print("✅ Shutdown complete")