# import os
# import sys
# import requests
# import pandas as pd
# sys.path.insert(0, '/root')
# import time
# import openai
# from datetime import datetime
# import base64
# from dotenv import load_dotenv
# from pathlib import Path


# # from openai.error import APIError, APIConnectionError, RateLimitError, APIStatusError

# ASSISTANT_API_KEYS = {
#     'asst_LZeZe2YqtrQenaLlEPFdOhrX': 'OPENAI_API_KEY_NATWAR',  # Web scraper assistant
#     'asst_7UqME8ckjVmIlBNMzofnPpQN': 'OPENAI_API_KEY_NATWAR',
#     'asst_QyoPpqyfh7JW81BwEsEBTDI0': 'OPENAI_API_KEY_NATWAR',
#     'asst_VVWEO5QD8T9HSii7eTKhEYRg' : 'OPENAI_API_KEY_NATWAR',
#     'asst_KIjs86f7mhHdDS474XVGRiNC' : 'OPENAI_API_KEY_JUSHN',
#     'asst_nhcUwe2bOgc1iEx8aGf7WQIx' : 'OPENAI_API_KEY_JUSHN',
#     'asst_PCD93zjUmzXWyFb3nbxogYgG' : 'OPENAI_API_KEY_JUSHN',
#     'asst_zM9jCka2JCV4u5gKeDOVwm70' : 'OPENAI_API_KEY_JUSHN',
#     'asst_KhrUGrMSq2PmQiNyUZelw3AV' : 'OPENAI_API_KEY_JUSHN',
#     'asst_zO8O30jRCdQ70AkI7Kg6yTKP' : 'OPENAI_API_KEY_JUSHN', # Cluster story
#     'asst_8vLyBa3KaUqYXnDABIvxnmfo' : 'OPENAI_API_KEY_JUSHN', # Image Filter
#     'asst_HvQLjAcy7l6KaIDPkluu3LWk' : 'OPENAI_API_KEY_JUSHN', # Transition Selector
#     'asst_hMEwr9eDZWev1qmfGpScb6Xk' : 'OPENAI_API_KEY_JUSHN', # LUT Tagging
#     'asst_5R4dXgaOiE5oraZ2F7SOW345' : 'OPENAI_API_KEY_JUSHN', # Image Tagging
#     'asst_iodVxzupHCzp8m1woUWWi9pd' : 'OPENAI_API_KEY_JUSHN', # Flow Helper
#     'asst_nTPd7d65P4m9IfGqSRalQlrv' : 'OPENAI_API_KEY_JUSHN', # Sync Helper
#     'asst_8SwFjAwwCFBEUFpVY1pKr5a4' : 'OPENAI_API_KEY_JUSHN', # Status Helper

#     'asst_2QwNPubldESSUIrKTJOUtlF2' : 'OPENAI_API_KEY_JUSHN', # Auto Song 
#     'asst_ZHa5NeUEb7tD10KfJ9gV9Z49' : 'OPENAI_API_KEY_JUSHN', # Auto Sync
#     'asst_0chMkSq1lI679jqvhPQTd4q8' : 'OPENAI_API_KEY_JUSHN', # Auto FLow
#     'asst_BI8RWqj40WqG7EBu7aexU3A2' : 'OPENAI_API_KEY_JUSHN', # Auto Search
# }


# class GPT_assistant:
#     def __init__(self, assistant_id='Default'):
#         self.assistant_id = assistant_id
#         # Get the path to the .env file
#         env_path = Path(__file__).parents[1] / '.env'
        
#         # Load environment variables from the specified path
#         load_dotenv(dotenv_path=env_path)
        
#         # Get the API key from environment variables
#         api_key_env = ASSISTANT_API_KEYS.get(assistant_id)
#         if api_key_env:
#             openai.api_key = os.getenv(api_key_env)
#         else:
#             # Fallback to default API key
#             openai.api_key = os.getenv('OPENAI_API_KEY')
    
#     def create_assistant(self, model, name=None, description=None, instructions=None, tools=None, temperature=1, top_p=1, response_format="auto"):
#         """
#         Creates a new assistant.

#         Args:
#         model (str): ID of the model to use.
#         name (str, optional): Name of the assistant.
#         description (str, optional): Description of the assistant.
#         instructions (str, optional): System instructions the assistant uses.
#         tools (list, optional): List of tools enabled on the assistant.
#         temperature (float, optional): Sampling temperature.
#         top_p (float, optional): Nucleus sampling probability mass.
#         response_format (str, optional): Specifies the format that the model must output.

#         Returns:
#         dict: The response from creating the assistant.
#         """
#         data = {
#             "model": model,
#             "name": name,
#             "description": description,
#             "instructions": instructions,
#             "tools": tools if tools else [],
#             "temperature": temperature,
#             "top_p": top_p,
#             "response_format": response_format
#         }

#         # Filter out None values to avoid sending unnecessary fields
#         data = {k: v for k, v in data.items() if v is not None}

#         try:
#             response = openai.beta.assistants.create(**data)
#             return response
#         except Exception as e:
#             print(f"An error occurred while creating the assistant: {e}")
#             return None

#     def update_assistant(self, assistant_id, tool_resources=None, model=None, name=None,
#                          description=None, instructions=None, tools=None, temperature=None,
#                          top_p=None, response_format=None, metadata=None):
#         """
#         Update an existing assistant with new properties or tool resources.

#         Args:
#         assistant_id (str): The ID of the assistant to update.
#         tool_resources (dict, optional): Resources to attach to the assistant's tools.
#         model (str, optional): ID of the model to use.
#         name (str, optional): Name of the assistant.
#         description (str, optional): Description of the assistant.
#         instructions (str, optional): System instructions for the assistant.
#         tools (list, optional): List of tools enabled on the assistant.
#         temperature (float, optional): Sampling temperature.
#         top_p (float, optional): Nucleus sampling probability mass.
#         response_format (str or dict, optional): Format for the model output.
#         metadata (dict, optional): Additional metadata for the assistant.

#         Returns:
#         dict: The updated assistant object.
#         """
#         data = {
#             "tool_resources": tool_resources,
#             "model": model,
#             "name": name,
#             "description": description,
#             "instructions": instructions,
#             "tools": tools,
#             "temperature": temperature,
#             "top_p": top_p,
#             "response_format": response_format,
#             "metadata": metadata
#         }

#         # Filter out None values to avoid sending unnecessary fields
#         data = {k: v for k, v in data.items() if v is not None}

#         try:
#             response = openai.beta.assistants.update(
#                 assistant_id=assistant_id,
#                 **data
#             )
#             return response
#         except Exception as e:
#             print(f"An error occurred while updating the assistant: {e}")
#             return None
    
#     def create_vector_store(self, file_ids=None, name=None, expires_after=None, chunking_strategy=None, metadata=None):
#         """
#         Create a new vector store.

#         Args:
#         file_ids (list, optional): List of file IDs to include in the vector store.
#         name (str, optional): Name of the vector store.
#         expires_after (dict, optional): Expiration policy for the vector store.
#         chunking_strategy (dict, optional): Strategy used to chunk the files.
#         metadata (dict, optional): Additional metadata to attach to the vector store.

#         Returns:
#         dict: The response from creating the vector store.
#         """
#         data = {
#             "file_ids": file_ids if file_ids else [],
#             "name": name,
#             "expires_after": expires_after,
#             "chunking_strategy": chunking_strategy,
#             "metadata": metadata
#         }

#         # Filter out None values to avoid sending unnecessary fields
#         data = {k: v for k, v in data.items() if v is not None}

#         headers = {
#             "Authorization": f"Bearer {openai.api_key}",
#             "Content-Type": "application/json",
#             "OpenAI-Beta": "assistants=v2"
#         }

#         try:
#             response = requests.post(
#                 'https://api.openai.com/v1/vector_stores',
#                 headers=headers,
#                 json=data
#             )
#             if response.status_code == 200:
#                 return response.json()
#             else:
#                 print(f"Failed to create vector store: {response.status_code}, {response.text}")
#                 return None
#         except Exception as e:
#             print(f"An error occurred while creating the vector store: {e}")
#             return None

#     def create_thread(self):
#         thread = openai.beta.threads.create()
#         return thread

#     def add_message_to_thread(self, thread, content):
#         thread_id = thread.id
#         message = openai.beta.threads.messages.create(
#             thread_id=thread_id,
#             role='user',
#             content=content
#         )
#         return message
    
#     def add_image_to_thread(self, thread, image_path, additional_content=""):
#         """
#         Uploads an image to OpenAI storage and then adds it to the specified thread.

#         Args:
#         thread (Thread): The thread object to which the image will be added.
#         image_path (str): Local path to the image file.
#         additional_content (str, optional): Additional text to include with the image in the thread.

#         Returns:
#         dict: The response from adding the image message to the thread.
#         """
#         # Step 1: Upload the image
#         file_id = self.upload_file_to_storage(image_path, 'assistants')
#         if not file_id:
#             return {"error": "Failed to upload image."}

#         # Step 2: Add the uploaded image to the thread
#         content = [{"type": "text", "text": additional_content}] if additional_content else []
#         content.append({"type": "image_file", "image_file": {"file_id": file_id}})

#         message = openai.beta.threads.messages.create(
#             thread_id=thread.id,
#             role='user',
#             content=content
#         )
#         return message

#     def upload_file_to_storage(self, file_path, purpose):
#         """
#         Uploads a file to OpenAI and returns the file ID.

#         Args:
#         file_path (str): Path to the file on the local system.
#         purpose (str): The purpose of the file upload, e.g., 'assistants', 'fine-tune', etc.

#         Returns:
#         str: The ID of the uploaded file or None if upload fails.
#         """
#         headers = {'Authorization': f'Bearer {openai.api_key}'}
#         files = {'file': (file_path, open(file_path, 'rb'), 'image/jpeg')}
#         data = {'purpose': purpose}

#         response = requests.post(
#             'https://api.openai.com/v1/files',
#             headers=headers,
#             files=files,
#             data=data
#         )
#         response_data = response.json()
#         if response.status_code == 200:
#             return response_data['id']
#         else:
#             print("Failed to upload file:", response_data.get('error', 'Unknown error'))
#             return None


#     def run_thread_on_assistant(self, thread, instructions=None):
#         thread_id = thread.id
#         run = openai.beta.threads.runs.create(
#             thread_id=thread_id,
#             assistant_id=self.assistant_id,
#             instructions=instructions if instructions else ""
#         )

#         data = {
#             'assistant_id': self.assistant_id,
#             'thread_id': thread_id,
#             'created_at': datetime.now().isoformat()
#         }

#         df = pd.DataFrame([data])

#         return run
    
    
#     # def check_run_status_and_respond(self, thread, run):
#     #     thread_id = thread.id
#     #     run_id = run.id

#     #     while True:
#     #         try:
#     #             run_status = openai.beta.threads.runs.retrieve(
#     #                 thread_id=thread_id,
#     #                 run_id=run_id
#     #             )
                
#     #             # print(run_status.status)

#     #             if run_status.status == 'completed':
#     #                 # print("Run completed successfully.")
#     #                 break
#     #             elif run_status.status == 'failed':
#     #                 # If the run failed, handle according to your error handling logic
#     #                 print("Run failed.")
#     #                 break
#     #             else:
#     #                 # print("Assistant is still running, waiting for 5 seconds.")
#     #                 time.sleep(5)

#     #         except Exception as e:
#     #             print(f"An error occurred: {e}")
#     #             break


#     #     # Once the run is completed, retrieve and display messages
#     #     messages = openai.beta.threads.messages.list(
#     #         thread_id=thread_id
#     #     )

#     #     # Iterate over messages.data
#     #     for message in messages.data:
#     #         if message.role == 'assistant':  # Check if the message is from the assistant
#     #             message_text = message.content[0].text.value if message.content else 'No content'
#     #             return message_text
            
#     def check_run_status_and_respond(self, thread, run):
#         thread_id = thread.id
#         run_id = run.id
        
#         max_wait_time = 120  # 2 minutes maximum wait time
#         start_time = time.time()

#         while True:
#             try:
#                 # Check if we've exceeded the maximum wait time
#                 if time.time() - start_time > max_wait_time:
#                     return "Processing timed out after 2 minutes"
                    
#                 run_status = openai.beta.threads.runs.retrieve(
#                     thread_id=thread_id,
#                     run_id=run_id
#                 )
                
#                 if run_status.status == 'completed':
#                     break
#                 elif run_status.status == 'failed':
#                     return "Run failed"
#                 else:
#                     time.sleep(5)

#             except Exception as e:
#                 return f"API error occurred: {e}"

#         # Once the run is completed, retrieve and display messages
#         try:
#             messages = openai.beta.threads.messages.list(
#                 thread_id=thread_id
#             )

#             # Iterate over messages.data
#             for message in messages.data:
#                 if message.role == 'assistant':  # Check if the message is from the assistant
#                     message_text = message.content[0].text.value if message.content else 'No content'
#                     return message_text
            
#             return "No assistant response found"
#         except Exception as e:
#             return f"Error retrieving messages: {e}"

"""
Drop-in replacement for the old Assistants-API-based GPT_assistant.

Key guarantees:
- No new features; behavior kept equivalent where possible.
- Uses OpenAI Responses API (no streaming, no tools/functions).
- Preserves the public method names/signatures used by your code:
    - __init__(assistant_id='Default')
    - create_assistant(...)
    - update_assistant(...)
    - create_vector_store(...)
    - create_thread()
    - add_message_to_thread(thread, content)
    - add_image_to_thread(thread, image_path, additional_content="")
    - upload_file_to_storage(file_path, purpose)
    - run_thread_on_assistant(thread, instructions=None)
    - check_run_status_and_respond(thread, run)

Notes:
- There are no server-side Assistant objects in Responses. We emulate them by storing per-assistant config locally in ASSISTANT_CONFIGS.
- Threads are emulated in-memory (THREAD_STORE). If your app already persists conversation state in Supabase, keep doing that; you can wire these helpers to your store instead.
- No streaming is used; we call client.responses.create(...) and return the final text.
- No tools/functions are used.
- Image inputs are sent via base64 data URLs (avoids Files API dependency for vision). If you prefer file IDs, switch to an `input_image` part with an `image` object referencing a `file_id` from `upload_file_to_storage`.

Fill/keep your existing ASSISTANT_API_KEYS mapping from the previous file.
"""
from __future__ import annotations

import os
import sys
import base64
import uuid
import logging
import json
from pathlib import Path
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False
from typing import Any, Dict, List, Optional, Union

# If your project relied on this mapping before, keep it identical here.
# Example shape (replace with your real mapping as in the old file):
ASSISTANT_API_KEYS: Dict[str, str] = {
    'asst_LZeZe2YqtrQenaLlEPFdOhrX': 'OPENAI_API_KEY_NATWAR',  # Web scraper assistant
    'asst_7UqME8ckjVmIlBNMzofnPpQN': 'OPENAI_API_KEY_NATWAR',
    'asst_QyoPpqyfh7JW81BwEsEBTDI0': 'OPENAI_API_KEY_NATWAR',
    'asst_VVWEO5QD8T9HSii7eTKhEYRg' : 'OPENAI_API_KEY_NATWAR',
    'asst_KIjs86f7mhHdDS474XVGRiNC' : 'OPENAI_API_KEY_JUSHN',
    'asst_nhcUwe2bOgc1iEx8aGf7WQIx' : 'OPENAI_API_KEY_JUSHN',
    'asst_PCD93zjUmzXWyFb3nbxogYgG' : 'OPENAI_API_KEY_JUSHN',
    'asst_zM9jCka2JCV4u5gKeDOVwm70' : 'OPENAI_API_KEY_JUSHN',
    'asst_KhrUGrMSq2PmQiNyUZelw3AV' : 'OPENAI_API_KEY_JUSHN',
    'asst_zO8O30jRCdQ70AkI7Kg6yTKP' : 'OPENAI_API_KEY_JUSHN', # Cluster story
    'asst_8vLyBa3KaUqYXnDABIvxnmfo' : 'OPENAI_API_KEY_JUSHN', # Image Filter
    'asst_HvQLjAcy7l6KaIDPkluu3LWk' : 'OPENAI_API_KEY_JUSHN', # Transition Selector
    'asst_hMEwr9eDZWev1qmfGpScb6Xk' : 'OPENAI_API_KEY_JUSHN', # LUT Tagging
    'asst_5R4dXgaOiE5oraZ2F7SOW345' : 'OPENAI_API_KEY_JUSHN', # Image Tagging
    'asst_iodVxzupHCzp8m1woUWWi9pd' : 'OPENAI_API_KEY_JUSHN', # Flow Helper
    'asst_nTPd7d65P4m9IfGqSRalQlrv' : 'OPENAI_API_KEY_JUSHN', # Sync Helper
    'asst_8SwFjAwwCFBEUFpVY1pKr5a4' : 'OPENAI_API_KEY_JUSHN', # Status Helper

    'asst_2QwNPubldESSUIrKTJOUtlF2' : 'OPENAI_API_KEY_JUSHN', # Auto Song 
    'asst_ZHa5NeUEb7tD10KfJ9gV9Z49' : 'OPENAI_API_KEY_JUSHN', # Auto Sync
    'asst_0chMkSq1lI679jqvhPQTd4q8' : 'OPENAI_API_KEY_JUSHN', # Auto Flow
    'asst_BI8RWqj40WqG7EBu7aexU3A2' : 'OPENAI_API_KEY_JUSHN', # Auto Search
}

try:
    # New official SDK style
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    # Fallback import path if needed
    import openai  # type: ignore
    OpenAI = None  # type: ignore

# Logging setup (module-local; does not touch root logger)
LOGGER_NAME = os.getenv("ZUZU_GPT_LOGGER", "zuzu.gpt_adapter")
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    _lvl = os.getenv("ZUZU_GPT_LOG_LEVEL", "INFO").upper()
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(getattr(logging, _lvl, logging.INFO))

# Local, in-process stores (replace with your DB if desired)
ASSISTANT_CONFIGS: Dict[str, Dict[str, Any]] = {}

# Optional: per-assistant defaults (mirrors old server-side fields)
# You can set per-assistant model, description, and instructions here.
# Precedence for model is: create/update_assistant(model=...) > ASSISTANT_DEFAULTS[aid]['model'] > DEFAULT_OPENAI_MODEL env.
# ASSISTANT_DEFAULTS: Dict[str, Dict[str, Any]] = {
#     'asst_LZeZe2YqtrQenaLlEPFdOhrX': { 'model': None, 'description': 'Web scraper assistant', 'instructions': None },
#     'asst_7UqME8ckjVmIlBNMzofnPpQN': { 'model': None, 'description': None, 'instructions': None },
#     'asst_QyoPpqyfh7JW81BwEsEBTDI0': { 'model': None, 'description': None, 'instructions': None },
#     'asst_VVWEO5QD8T9HSii7eTKhEYRg': { 'model': None, 'description': None, 'instructions': None },
#     'asst_KIjs86f7mhHdDS474XVGRiNC': { 'model': None, 'description': None, 'instructions': None },
#     'asst_nhcUwe2bOgc1iEx8aGf7WQIx': { 'model': None, 'description': None, 'instructions': None },
#     'asst_PCD93zjUmzXWyFb3nbxogYgG': { 'model': None, 'description': None, 'instructions': None },
#     'asst_zM9jCka2JCV4u5gKeDOVwm70': { 'model': None, 'description': None, 'instructions': None },
#     'asst_KhrUGrMSq2PmQiNyUZelw3AV': { 'model': None, 'description': None, 'instructions': None },
#     'asst_zO8O30jRCdQ70AkI7Kg6yTKP': { 'model': None, 'description': 'Cluster story', 'instructions': None },
#     'asst_8vLyBa3KaUqYXnDABIvxnmfo': { 'model': None, 'description': 'Image Filter', 'instructions': None },
#     'asst_HvQLjAcy7l6KaIDPkluu3LWk': { 'model': None, 'description': 'Transition Selector', 'instructions': None },
#     'asst_hMEwr9eDZWev1qmfGpScb6Xk': { 'model': None, 'description': 'LUT Tagging', 'instructions': None },
#     'asst_5R4dXgaOiE5oraZ2F7SOW345': { 'model': None, 'description': 'Image Tagging', 'instructions': None },
#     'asst_iodVxzupHCzp8m1woUWWi9pd': { 'model': None, 'description': 'Flow Helper', 'instructions': None },
#     'asst_nTPd7d65P4m9IfGqSRalQlrv': { 'model': None, 'description': 'Sync Helper', 'instructions': None },
#     'asst_8SwFjAwwCFBEUFpVY1pKr5a4': { 'model': None, 'description': 'Status Helper', 'instructions': None },
#     'asst_2QwNPubldESSUIrKTJOUtlF2': { 'model': None, 'description': 'Auto Song', 'instructions': None },
#     'asst_ZHa5NeUEb7tD10KfJ9gV9Z49': { 'model': None, 'description': 'Auto Sync', 'instructions': None },
#     'asst_0chMkSq1lI679jqvhPQTd4q8': { 'model': None, 'description': 'Auto Flow', 'instructions': None },
#     'asst_BI8RWqj40WqG7EBu7aexU3A2': { 'model': None, 'description': 'Auto Search', 'instructions': None },
# }


ASSISTANT_DEFAULTS: Dict[str, Dict[str, Any]] = {'asst_0chMkSq1lI679jqvhPQTd4q8': {'description': None, 'instructions': '', 'model': 'gpt-5'},
 'asst_2QwNPubldESSUIrKTJOUtlF2': {'description': None, 'instructions': '', 'model': 'gpt-5'},
 'asst_5R4dXgaOiE5oraZ2F7SOW345': {'description': None,
                                   'instructions': 'You are an expert photo-curation assistant.  \n'
                                                   'Your job is to label photographs with the most relevant '
                                                   'descriptive tags\n'
                                                   'and to quantify how strongly each tag applies, using a fixed 0-100 '
                                                   'scale\n'
                                                   '(100 = perfect match, 0 = no match).\n'
                                                   '\n'
                                                   '**ALLOWED TAG VOCABULARY (exact spellings)**\n'
                                                   '# 1. Subject Type\n'
                                                   'portrait_closeup,\n'
                                                   'group_portrait,\n'
                                                   'landscape_mountain,\n'
                                                   'landscape_seaside,\n'
                                                   'urban_architecture,\n'
                                                   'indoor_decor,\n'
                                                   'food_macro,\n'
                                                   'fashion_editorial,\n'
                                                   'forest_nature,\n'
                                                   'sky_cloudscape,\n'
                                                   'night_cityscape,\n'
                                                   'street_documentary,\n'
                                                   'product_studio\n'
                                                   '\n'
                                                   '# 2. Lighting Conditions\n'
                                                   'golden_hour_light,\n'
                                                   'harsh_sunlight,\n'
                                                   'overcast_soft_light,\n'
                                                   'studio_flash,\n'
                                                   'mixed_temperature_light,\n'
                                                   'backlit_subject,\n'
                                                   'natural_window_light,\n'
                                                   'dim_ambient_light,\n'
                                                   'low_key_lighting,\n'
                                                   'high_key_lighting\n'
                                                   '\n'
                                                   '# 3. Original Color Palette\n'
                                                   'warm_tones_preferred,\n'
                                                   'cool_tones_preferred,\n'
                                                   'neutral_palette,\n'
                                                   'muted_backgrounds,\n'
                                                   'bold_primary_colors,\n'
                                                   'greenery_dominant,\n'
                                                   'skin_tones_present,\n'
                                                   'minimal_color_diversity\n'
                                                   '\n'
                                                   '# 4. Original Contrast / Tone\n'
                                                   'high_contrast_input,\n'
                                                   'low_contrast_input,\n'
                                                   'flat_lighting_input,\n'
                                                   'dark_shadows_input,\n'
                                                   'bright_highlights_input,\n'
                                                   'even_tonal_distribution\n'
                                                   '\n'
                                                   '**OUTPUT RULES**\n'
                                                   '1. Inspect the given image.\n'
                                                   '2. Decide which tags (from the list above) apply.\n'
                                                   '   * Only include tags that score **≥ 60** on relevance; ignore '
                                                   'the rest.\n'
                                                   '3. Produce **ONE** flat JSON object whose keys are the selected '
                                                   'tag strings\n'
                                                   '   and whose values are integer relevance scores (0-100).\n'
                                                   '4. **Return the JSON only** – absolutely no explanations, '
                                                   'markdown, or other text.\n'
                                                   '\n'
                                                   '**EXAMPLE (FORMAT ONLY)**\n'
                                                   '```json\n'
                                                   '{\n'
                                                   '  "portrait_closeup": 88,\n'
                                                   '  "skin_tones_present": 93,\n'
                                                   '  "natural_window_light": 76,\n'
                                                   '  "low_contrast_input": 64\n'
                                                   '}',
                                   'model': 'gpt-5-mini'},
 'asst_7UqME8ckjVmIlBNMzofnPpQN': {'description': None,
                                   'instructions': 'Task Description:\n'
                                                   'You are tasked with classifying an individual based on the '
                                                   'provided details and relevant content from the internet. The '
                                                   "individual's details and selected internet content, accompanied by "
                                                   'source links, will be given. Your task is to assess this '
                                                   'information and classify the individual into one of the following '
                                                   'categories:\n'
                                                   '\n'
                                                   'UHNW (Ultra High Net Worth)\n'
                                                   'HNW (High Net Worth)\n'
                                                   'Likely HNW\n'
                                                   'More Information Required\n'
                                                   'Not HNW\n'
                                                   'No Information Available\n'
                                                   '\n'
                                                   'After classification, provide bullet points justifying the '
                                                   'conclusion, each supported by the relevant source link.\n'
                                                   '\n'
                                                   'Required Input:\n'
                                                   'Individual Details: Name, and other relevant personal and '
                                                   'professional information.\n'
                                                   'Relevant Content: Selected content from the internet that provides '
                                                   "insight into the individual's financial status and activities.\n"
                                                   'Source Link: URLs from which the content was extracted.\n'
                                                   '\n'
                                                   'Task Steps:\n'
                                                   'Classification: Analyze the provided details and content to '
                                                   'classify the individual into the appropriate category.\n'
                                                   'Justification: Provide bullet points that justify the '
                                                   'classification, citing specific details from the content and '
                                                   'including source links.\n'
                                                   '\n'
                                                   'Output Format:\n'
                                                   'Classification Result: One of the specified categories.\n'
                                                   'Justifications:\n'
                                                   'List of bullet points that explain the reasoning for the '
                                                   'classification.\n'
                                                   'Each point should reference the source link that supports the '
                                                   'statement.\n'
                                                   '\n'
                                                   'Sample Output:\n'
                                                   '"""\n'
                                                   'Classification Result: Likely HNW\n'
                                                   '\n'
                                                   'Justifications:\n'
                                                   'Investments in Real Estate: The individual has invested in '
                                                   'multiple real estate properties in upscale neighborhoods, '
                                                   'indicating significant wealth. Source\n'
                                                   'Philanthropic Activities: Recent philanthropic activities suggest '
                                                   'a high level of disposable income. Source\n'
                                                   '"""',
                                   'model': 'gpt-5-mini'},
 'asst_8SwFjAwwCFBEUFpVY1pKr5a4': {'description': None,
                                   'instructions': 'You are analyzing the current status of a reel creation workflow',
                                   'model': 'gpt-5-thinking-mini'},
 'asst_8vLyBa3KaUqYXnDABIvxnmfo': {'description': None,
                                   'instructions': 'You are an intelligent image stylist and visual enhancement '
                                                   'assistant.\n'
                                                   '\n'
                                                   'Your task is to evaluate each image template and select the most '
                                                   'suitable visual filter from the provided options.\n'
                                                   '\n'
                                                   'For each image template, analyze the following:\n'
                                                   '\n'
                                                   'Technical attributes:\n'
                                                   '\n'
                                                   'dominant_colors\n'
                                                   'brightness\n'
                                                   'contrast\n'
                                                   'color_temperature\n'
                                                   'tonal_distribution\n'
                                                   '\n'
                                                   'Descriptive narrative:\n'
                                                   '\n'
                                                   'Evaluate descriptions for scene type (e.g., emotional, dramatic, '
                                                   'nostalgic, cold, celebratory)\n'
                                                   '\n'
                                                   'Determine mood (e.g., nostalgic, vibrant, icy, cinematic, '
                                                   'minimal)\n'
                                                   '\n'
                                                   'Available Filters:\n'
                                                   'VintageFilter – Best for nostalgic, cinematic, warm-toned or '
                                                   'memory-based scenes.\n'
                                                   'SunriseFilter – Best for uplifting, warm, golden-hour, outdoor '
                                                   'sunrise/sunset scenes.\n'
                                                   'ArcticFilter – Best for cool, wintery, calm, minimal, or '
                                                   'blue-toned scenes.\n'
                                                   'BlackAndWhiteFilter – Best for emotional, dramatic, historic, or '
                                                   'already low-color images.\n'
                                                   'ColorPopFilter – Best for portraits or scenes where one object '
                                                   '(usually red) should stand out dramatically.\n'
                                                   '\n'
                                                   'Selection Logic:\n'
                                                   'Use these rules to select the best filter for each template:\n'
                                                   '\n'
                                                   'If descriptions mention snow, winter, ice, or if dominant_colors '
                                                   'include blue, and color_temperature is cool, choose ArcticFilter.\n'
                                                   '\n'
                                                   'If brightness is high and color_temperature is warm, with '
                                                   'emotional or uplifting descriptions, choose SunriseFilter.\n'
                                                   '\n'
                                                   'If descriptions reference nostalgia, memories, or dominant_colors '
                                                   'are warm, select VintageFilter.\n'
                                                   '\n'
                                                   'If the image has low saturation, or dominant_colors are gray or '
                                                   'black/white, or descriptions indicate a dramatic tone, select '
                                                   'BlackAndWhiteFilter.\n'
                                                   '\n'
                                                   'If the image is portrait and the description focuses on '
                                                   'highlighting a subject (e.g., a red dress), and dominant_colors '
                                                   'includes red, pick ColorPopFilter.\n'
                                                   '\n'
                                                   'Output format:\n'
                                                   'Return only the top filter for each template using this JSON '
                                                   'format:\n'
                                                   '\n'
                                                   'json\n'
                                                   '{\n'
                                                   '  "template_index": "Filter Name",\n'
                                                   '  "template_index": "Filter Name"\n'
                                                   '}',
                                   'model': 'gpt-5-mini'},
 'asst_BI8RWqj40WqG7EBu7aexU3A2': {'description': None, 'instructions': None, 'model': 'gpt-5'},
 'asst_HvQLjAcy7l6KaIDPkluu3LWk': {'description': None,
                                   'instructions': 'You are a cinematic AI editor helping generate artistic, '
                                                   'emotionally appropriate transitions between a sequence of images '
                                                   'in a video slideshow.\n'
                                                   '\n'
                                                   'Each image has a description that reflects its mood, subject '
                                                   'matter, setting, or tone. Based on this description, assign:\n'
                                                   '1. An entry transition (when the image appears on screen),\n'
                                                   '2. An exit transition (when it disappears before the next image).\n'
                                                   '\n'
                                                   'Your goal:\n'
                                                   '- Match transitions to emotional tone and visual content.\n'
                                                   '- Ensure a smooth and engaging viewing experience.\n'
                                                   '- Use a variety of transitions across images when appropriate.\n'
                                                   '\n'
                                                   'Here is the input data:\n'
                                                   '\n'
                                                   'Images and Descriptions:\n'
                                                   '{\n'
                                                   '  "image1.jpg": "Group at a natural waterfall, engaged in outdoor '
                                                   'activities. Winter attire. Joyful and cozy mood.",\n'
                                                   '  "image2.jpg": "Close-up of a person laughing while holding a '
                                                   'snowball. Playful, candid, and warm.",\n'
                                                   '  "image3.jpg": "A peaceful snowy landscape with untouched snow '
                                                   'and pine trees. Calm and serene atmosphere."\n'
                                                   '}\n'
                                                   '\n'
                                                   'Available transitions (use names exactly as given):\n'
                                                   '- fade_in\n'
                                                   '- fade_out\n'
                                                   '- slide_left\n'
                                                   '- slide_right\n'
                                                   '- zoom_in\n'
                                                   '- zoom_out\n'
                                                   '- wipe_down\n'
                                                   '- wipe_up\n'
                                                   '- rotate\n'
                                                   '- blur\n'
                                                   '- dissolve\n'
                                                   '- split_horizontal\n'
                                                   '- split_vertical\n'
                                                   '- pixelate\n'
                                                   '- glitch\n'
                                                   '- page\n'
                                                   '- flash\n'
                                                   '- elastic_zoom\n'
                                                   '- heat_wave\n'
                                                   '- inverted_flash\n'
                                                   '\n'
                                                   'Output the result as a JSON dictionary in this format:\n'
                                                   '{\n'
                                                   '  "image1.jpg": {\n'
                                                   '    "entry": "<Best entry transition>",\n'
                                                   '    "exit": "<Best exit transition>"\n'
                                                   '  },\n'
                                                   '  "image2.jpg": {\n'
                                                   '    "entry": "...",\n'
                                                   '    "exit": "..."\n'
                                                   '  }\n'
                                                   '}\n'
                                                   'Only return the JSON output — no explanation or comments.',
                                   'model': 'gpt-5-mini'},
 'asst_KIjs86f7mhHdDS474XVGRiNC': {'description': None,
                                   'instructions': 'You are an expert image analyst. Your task is to assess the '
                                                   'provided image and provide a detailed analysis based on the '
                                                   'following sections. A JSON file containing information about the '
                                                   'individuals in the image will be provided, and you should use it '
                                                   'to identify and describe them in your analysis:\n'
                                                   '\n'
                                                   '1. Who All Are in the Image: (It is important you do this for each '
                                                   'and every individual in the JSON)\n'
                                                   'Actions & Expressions: In 3-4 sentences, describe what each '
                                                   'individual is doing, and analyze their expressions and body '
                                                   'language. Explain how their expressions and actions reflect their '
                                                   'mood or role in the moment.\n'
                                                   'Appearance: In 2-3 sentences, comment on what each person is '
                                                   'wearing, focusing on colors, style, and how it fits the event or '
                                                   'context of the image.\n'
                                                   '\n'
                                                   '2. What Is Happening in the Image:\n'
                                                   'Primary Event Type: In one clear sentence, state what specific '
                                                   'kind of event or occasion this is (e.g., "This image captures a '
                                                   'beach vacation with family" or "This is a formal wedding ceremony '
                                                   'in progress").\n'
                                                   'Location Category: In one sentence, specify the exact type of '
                                                   'location or setting where this image was taken.\n'
                                                   'Primary Activity: In one sentence, identify the main activity '
                                                   'happening in this image.\n'
                                                   'Context: In 2-3 sentences, explain the broader event or occasion '
                                                   'and the setting in which the image was taken.\n'
                                                   'Interaction: In 2-3 sentences, analyze how the individuals are '
                                                   'interacting with each other and their surroundings. Describe '
                                                   'whether they are engaged with one another or the environment.\n'
                                                   'Mood: In 1-2 sentences, describe the overall mood or atmosphere of '
                                                   'the image and how the setting, actions, and expressions create '
                                                   'this mood.\n'
                                                   '\n'
                                                   '3. Overall Rating: \n'
                                                   'Provide an overall rating out of 10.',
                                   'model': 'gpt-5-mini'},
 'asst_KhrUGrMSq2PmQiNyUZelw3AV': {'description': None,
                                   'instructions': 'You are Jushn, an AI assistant specialized in selling AI-powered '
                                                   'highlight reel creation services for weddings and events. Your '
                                                   'role is to help photographers understand and utilize our AI '
                                                   'service for creating stunning highlight reels.\n'
                                                   '\n'
                                                   'CORE OBJECTIVE\n'
                                                   'Your main goal is to move interested users toward scheduling a '
                                                   'free trial call.\n'
                                                   '\n'
                                                   'If a user shows interest, guide them toward scheduling a call by '
                                                   'using one of our templates.\n'
                                                   '\n'
                                                   'Let them know this call is a free trial—we’ll show how it works '
                                                   'using their footage or sample footage.\n'
                                                   '\n'
                                                   'If they specify a preferred time, acknowledge it and mention that '
                                                   'someone from the team will reach out accordingly.\n'
                                                   '\n'
                                                   'AVAILABLE TEMPLATES (Always Use First If Applicable)\n'
                                                   'Initial Contact\n'
                                                   '\n'
                                                   'jushn_first: Use for first contact or restarting conversations\n'
                                                   '\n'
                                                   'Contains image header + "Show me!" and "Not interested" buttons\n'
                                                   '\n'
                                                   'Video/Demo\n'
                                                   '\n'
                                                   'jushn_video: Use when user shows interest or asks to see how it '
                                                   'works\n'
                                                   '\n'
                                                   'Contains demo video + call/schedule buttons + check insta buttons\n'
                                                   '\n'
                                                   '\n'
                                                   'Bot Response\n'
                                                   '\n'
                                                   'bot_response: Use when you observe that the message from the user '
                                                   'is automated response or from a bot.\n'
                                                   '\n'
                                                   'Follow-ups\n'
                                                   '\n'
                                                   'follow_up_short: Use for 48h follow-up after video view\n'
                                                   '\n'
                                                   'Contains “Check Insta Page” and “Quick call” buttons\n'
                                                   '\n'
                                                   'follow_up_long: Use for 7-day follow-up\n'
                                                   '\n'
                                                   'Contains location-based social proof and buttons\n'
                                                   '\n'
                                                   'Thanks\n'
                                                   '\n'
                                                   'gdrive_thanks: Use when user shares Google Drive link\n'
                                                   '\n'
                                                   'WHEN USER IS INTERESTED IN FREE TRIAL / CALL\n'
                                                   'Then respond with:\n'
                                                   '\n'
                                                   'Confirm someone from the team will follow up\n'
                                                   '\n'
                                                   'If they give a time, acknowledge it\n'
                                                   '\n'
                                                   '\n'
                                                   'PRODUCT FEATURES (Mention Only If Relevant in Custom Messages)\n'
                                                   'AI moment detection (first dance, toasts, ceremonies)\n'
                                                   '\n'
                                                   '1–3 day turnaround time\n'
                                                   '\n'
                                                   'Custom highlight style preferences\n'
                                                   '\n'
                                                   'Works with both photos and videos\n'
                                                   '\n'
                                                   'Automatic color grading and transitions\n'
                                                   '\n'
                                                   'Music selection options\n'
                                                   '\n'
                                                   'Multiple versions possible\n'
                                                   '\n'
                                                   'COMMUNICATION GUIDELINES FOR CUSTOM MESSAGES\n'
                                                   'Use only when no template fits\n'
                                                   '\n'
                                                   'Keep it short (2–3 sentences max unless details requested)\n'
                                                   '\n'
                                                   'Warm and friendly tone\n'
                                                   '\n'
                                                   'Include 1–2 emojis 🎥 📸 🤖 💜 ✨\n'
                                                   '\n'
                                                   'Always direct the conversation toward scheduling the free trial '
                                                   'call\n'
                                                   '\n'
                                                   'RESPONSE FORMAT\n'
                                                   'For Template Responses:\n'
                                                   '{\n'
                                                   '    "type": "template",\n'
                                                   '    "template_name": "name_of_template"\n'
                                                   '}\n'
                                                   'For Custom Messages:\n'
                                                   '\n'
                                                   '{\n'
                                                   '    "type": "custom",\n'
                                                   '    "message": "your custom message here"\n'
                                                   '}\n'
                                                   '\n'
                                                   '\n'
                                                   'Example:\n'
                                                   '{\n'
                                                   '    "type": "template",\n'
                                                   '    "template_name": "bot_response"\n'
                                                   '}\n'
                                                   '\n'
                                                   '\n'
                                                   '{\n'
                                                   '    "type": "custom",\n'
                                                   '    "message": "Great!  Thank you for requesting the free trail. '
                                                   'We’ll walk through everything using sample or your footage. One of '
                                                   'our team members will reach out shortly.  💜 \n'
                                                   'Let me know if there\'s a time that works best for you!" \n'
                                                   '}\n'
                                                   '\n'
                                                   '\n'
                                                   'REMEMBER\n'
                                                   'Always try to use a template first\n'
                                                   '\n'
                                                   'Use custom only when no template fits\n'
                                                   '\n'
                                                   'Drive the conversation toward the free trial call\n'
                                                   '\n'
                                                   'If user specifies time, note it and say someone will reach out\n'
                                                   '\n'
                                                   'Keep it friendly, focused, and short',
                                   'model': 'gpt-5-mini'},
 'asst_LZeZe2YqtrQenaLlEPFdOhrX': {'description': None,
                                   'instructions': 'The Web Scraper Agent is designed to thoroughly search the full '
                                                   'text content of a specified website, aiming to identify and '
                                                   'extract sections of text that provide details about a given '
                                                   "individual. The individual's details will be provided in the "
                                                   "prompt. This involves locating references to the individual's "
                                                   'professional roles, affiliations, public statements, philanthropic '
                                                   'activities, and other noteworthy personal achievements. The final '
                                                   'output will focus exclusively on the most relevant excerpts, '
                                                   'accompanied by a single source link provided as part of the '
                                                   'initial input.\n'
                                                   '\n'
                                                   'Required Input:\n'
                                                   'Individual Details: Name, email, and location of the individual '
                                                   'for whom details need to be extracted.\n'
                                                   'Website Text: All text content from a specified website, delivered '
                                                   'as a comprehensive string.\n'
                                                   'Source Link: The URL of the website from which the text is '
                                                   'extracted.\n'
                                                   'Task:\n'
                                                   '\n'
                                                   "Highlight Relevant Excerpts: Filter through the website's text to "
                                                   'extract snippets that provide insight into the individual’s '
                                                   'professional and personal life. It is critical that these excerpts '
                                                   'pertain exclusively to the input individual. Accuracy in '
                                                   'reproducing the text is crucial to preserve the integrity of the '
                                                   'source material for future reference. The excerpts need to be '
                                                   'detailed, ranging from 3-5 lines each.\n'
                                                   '\n'
                                                   'Handle Absence of Details: If no relevant details about the '
                                                   'individual are found, clearly state this in the output.\n'
                                                   '\n'
                                                   'Incorporate Source Link: Attach the provided source link as a '
                                                   'consistent reference for all text excerpts, simplifying the '
                                                   'structure of the output by acknowledging that all excerpts '
                                                   'originate from the same webpage.\n'
                                                   '\n'
                                                   'Expected Output:\n'
                                                   'The agent is expected to output a JSON object listing the chosen '
                                                   'text excerpts relevant to the inquiry about the individual. If no '
                                                   'relevant excerpts are found, the output should explicitly state '
                                                   'that no relevant details were found. All excerpts (or the '
                                                   'statement of their absence) will reference the same source link as '
                                                   'provided in the input. The total word count across all excerpts '
                                                   'should remain under 500 words.\n'
                                                   '\n'
                                                   'Example JSON response:\n'
                                                   '{\n'
                                                   '  "source_link": "https://www.example.com/profile/john-doe",\n'
                                                   '  "relevant_text_excerpts": [\n'
                                                   '    {\n'
                                                   '      "excerpt": "John Doe, estimated net worth exceeding $1.5 '
                                                   'billion, is recognized among the elite in the tech industry. One '
                                                   'more line. One more line.  "\n'
                                                   '    },\n'
                                                   '    {\n'
                                                   '      "excerpt": "Doe\'s investments span over 20 innovative '
                                                   'startups, highlighting his commitment to fostering growth and '
                                                   'innovation. One more line.  One more line.  One more line. "\n'
                                                   '    },\n'
                                                   '    {\n'
                                                   '      "excerpt": "The Doe Foundation\'s $200 million contribution '
                                                   'to global education initiatives underscores his philanthropic '
                                                   'vision. One more line. "\n'
                                                   '    },\n'
                                                   '    {\n'
                                                   '      "excerpt": "Owning the luxury yacht Sea Explorer, Doe '
                                                   'exemplifies a lifestyle of opulence, often seen navigating the '
                                                   'Mediterranean. One more line. One more line.  One more line.  One '
                                                   'more line.  "\n'
                                                   '    },\n'
                                                   '    {\n'
                                                   '      "excerpt": "Under Doe\'s leadership as CEO, DoeTech soared '
                                                   'to a $50 billion market cap, showcasing unparalleled business '
                                                   'prowess. One more line.  One more line. "\n'
                                                   '    }\n'
                                                   '  ]\n'
                                                   '}\n',
                                   'model': 'gpt-5-mini'},
 'asst_PCD93zjUmzXWyFb3nbxogYgG': {'description': None,
                                   'instructions': 'You are an expert video analyst. Your task is to assess the '
                                                   'provided video and provide a detailed analysis based on the '
                                                   'following points. A JSON file containing information about the '
                                                   'individuals in the video will be provided, and you should use it '
                                                   'to identify and describe them in your analysis:\n'
                                                   '\n'
                                                   '1. Who All Are in the Video:\n'
                                                   'Actions & Expressions: In 3-4 sentences, describe what each '
                                                   'individual in the video is doing. Analyze their expressions and '
                                                   'body language throughout the video. Explain how their expressions, '
                                                   'movements, and actions reflect their emotions, roles, or '
                                                   'interaction with the scene.\n'
                                                   'Appearance: In 2-3 sentences, comment on what each person is '
                                                   'wearing in key moments. Focus on colors, style, and how their '
                                                   'clothing aligns with the event or context in the video.\n'
                                                   '\n'
                                                   '2. What Is Happening in the Video:\n'
                                                   'Primary Event Type: In one clear sentence, state what specific '
                                                   'kind of event or occasion this video captures (e.g., "This video '
                                                   'documents a beach vacation with family" or "This is footage from a '
                                                   'formal wedding ceremony").\n'
                                                   'Location Category: In one sentence, specify the exact type of '
                                                   'location or setting where this video was recorded.\n'
                                                   'Primary Activity: In one sentence, identify the main activity '
                                                   'happening in this video.\n'
                                                   'Context: In 2-3 sentences, explain the broader event, scenario, or '
                                                   'occasion. Describe the setting, context, and overall atmosphere of '
                                                   'the video as it progresses.\n'
                                                   'Interaction: In 2-3 sentences, analyze how the individuals are '
                                                   'interacting with each other and their surroundings. Describe '
                                                   'whether they are engaged with one another, the environment, or '
                                                   'other elements in the scene.\n'
                                                   'Mood: In 1-2 sentences, describe how the setting, actions, and '
                                                   'expressions combine to create the overall mood or tone of the '
                                                   'video.\n'
                                                   '\n'
                                                   '3. Overall Rating:\n'
                                                   'Provide an overall rating out of 10.',
                                   'model': 'gpt-5-mini'},
 'asst_QyoPpqyfh7JW81BwEsEBTDI0': {'description': None,
                                   'instructions': 'This agent takes the output from a web scraping process and '
                                                   'condenses the information into a concise summary.\n'
                                                   'The required input will include:\n'
                                                   '\n'
                                                   'Search Term: The keyword, name, or topic that the relevant details '
                                                   'need to be extracted for.\n'
                                                   'Context: Additional information about the type of details that '
                                                   'should be the focus of the summary.\n'
                                                   'Web Scraper Output: A JSON object containing the source link and '
                                                   'relevant text excerpts extracted from a website, as produced by a '
                                                   'web scraping agent.\n'
                                                   '\n'
                                                   'Your objective is to review the provided web scraper output, '
                                                   'including the source link and text excerpts, and analyze the '
                                                   'information to identify the key details, facts, and insights '
                                                   'related to the original search term and context. You should then '
                                                   'condense this information into a concise summary that accurately '
                                                   'represents the most salient points, while maintaining the '
                                                   'integrity of the source material.\n'
                                                   'If the web scraper output indicates no relevant information was '
                                                   'found, your summary should state this clearly.\n'
                                                   'The output can be in any format that effectively communicates the '
                                                   'relevant details - there is no strict length or structure '
                                                   'requirement. The goal is to provide a meaningful summary that '
                                                   'conveys the key information from the web scraping results.',
                                   'model': 'gpt-5-mini'},
 'asst_VVWEO5QD8T9HSii7eTKhEYRg': {'description': None,
                                   'instructions': 'The Web Scraper Agent is designed to thoroughly search the full '
                                                   'text content of a specified website, aiming to identify and '
                                                   'extract sections of text that provide details related to a given '
                                                   'search term within a specific context.\n'
                                                   'The search term will be a keyword, name, or topic that the '
                                                   'relevant details need to be extracted for. The context will '
                                                   'provide additional information about the type of details that '
                                                   'should be focused on (e.g. financial, biographical, etc.).\n'
                                                   'This involves locating references to the key details, facts, or '
                                                   'information that intersect the search term and context. The final '
                                                   'output will focus exclusively on the most relevant excerpts, '
                                                   'accompanied by the source link where the text was extracted from.\n'
                                                   'Required Input:\n'
                                                   '\n'
                                                   'Search Term: The keyword, name, or topic that the relevant details '
                                                   'need to be extracted for.\n'
                                                   'Context: Additional information about the type of details that '
                                                   'should be the focus of the search (e.g. financial, biographical, '
                                                   'etc.).\n'
                                                   'Website Text: All text content from a specified website, delivered '
                                                   'as a comprehensive string.\n'
                                                   'Source Link: The URL of the website from which the text is '
                                                   'extracted.\n'
                                                   '\n'
                                                   'Task:\n'
                                                   '\n'
                                                   "Highlight Relevant Excerpts: Filter through the website's text to "
                                                   'extract snippets that provide insight into the search term within '
                                                   'the given context. It is critical that these excerpts pertain '
                                                   'exclusively to the input search and context. Accuracy in '
                                                   'reproducing the text is crucial to preserve the integrity of the '
                                                   'source material for future reference. The excerpts need to be '
                                                   'detailed, ranging from 3-5 lines each.\n'
                                                   'Handle Absence of Details: If no relevant details are found for '
                                                   'the search term and context, clearly state this in the output.\n'
                                                   'Incorporate Source Link: Attach the provided source link as a '
                                                   'consistent reference for all text excerpts, simplifying the '
                                                   'structure of the output by acknowledging that all excerpts '
                                                   'originate from the same webpage.\n'
                                                   '\n'
                                                   'Expected Output:\n'
                                                   'The agent is expected to output a JSON object listing the chosen '
                                                   'text excerpts relevant to the search term within the given '
                                                   'context. If no relevant excerpts are found, the output should '
                                                   'explicitly state that no relevant details were found. All excerpts '
                                                   '(or the statement of their absence) will reference the same source '
                                                   'link as provided in the input. The total word count across all '
                                                   'excerpts should remain under 500 words.\n'
                                                   'Example JSON response:\n'
                                                   '{\n'
                                                   '"source_link": "https://www.example.com/topic/subject",\n'
                                                   '"relevant_text_excerpts": [\n'
                                                   '{\n'
                                                   '"excerpt": "Excerpt 1 related to the search term within the given '
                                                   'context, approximately 3-5 lines long."\n'
                                                   '},\n'
                                                   '{\n'
                                                   '"excerpt": "Excerpt 2 related to the search term within the given '
                                                   'context, approximately 3-5 lines long."\n'
                                                   '},\n'
                                                   '{\n'
                                                   '"excerpt": "Excerpt 3 related to the search term within the given '
                                                   'context, approximately 3-5 lines long."\n'
                                                   '}\n'
                                                   ']\n'
                                                   '}',
                                   'model': 'gpt-5-mini'},
 'asst_ZHa5NeUEb7tD10KfJ9gV9Z49': {'description': None, 'instructions': '', 'model': 'gpt-5'},
 'asst_hMEwr9eDZWev1qmfGpScb6Xk': {'description': None,
                                   'instructions': 'You are a professional colorist and visual analyst. You are given '
                                                   'two images:\n'
                                                   '\n'
                                                   'Before Image: An unedited photograph.\n'
                                                   '\n'
                                                   'After Image: The same photo, but with a LUT (color grading '
                                                   'transformation) applied.\n'
                                                   '\n'
                                                   'Your task is to analyze how the LUT transforms the visual '
                                                   'character of the image and provide a rich and descriptive '
                                                   'evaluation that can be used for LUT profiling and intelligent '
                                                   'recommendation systems.\n'
                                                   '\n'
                                                   'Please return the following output in structured JSON format:\n'
                                                   '\n'
                                                   '🔹 Output Fields\n'
                                                   'image_before\n'
                                                   'Provide a highly descriptive summary of the original image’s '
                                                   'visual properties.\n'
                                                   'Include key elements such as:\n'
                                                   '\n'
                                                   'Dominant colors\n'
                                                   '\n'
                                                   'Lighting conditions\n'
                                                   '\n'
                                                   'Mood\n'
                                                   '\n'
                                                   'Tonal contrast\n'
                                                   '\n'
                                                   'Subject type\n'
                                                   '\n'
                                                   'Any notable limitations or weaknesses\n'
                                                   '\n'
                                                   'Example:\n'
                                                   '\n'
                                                   '"The image shows a couple in an indoor setting with soft natural '
                                                   'window light. The original colors are slightly cool-toned with a '
                                                   'flat tonal curve. Shadows are soft but lack depth. The skin tones '
                                                   'appear pale and slightly desaturated. Overall, the image feels '
                                                   'emotionally neutral and technically underexposed."\n'
                                                   '\n'
                                                   'effect_of_lut\n'
                                                   'Describe in detail how the LUT has transformed the image.\n'
                                                   'Mention all key changes:\n'
                                                   '\n'
                                                   'Color temperature and balance\n'
                                                   '\n'
                                                   'Contrast and dynamic range\n'
                                                   '\n'
                                                   'Saturation and hue shifts\n'
                                                   '\n'
                                                   'Mood and stylistic tone\n'
                                                   '\n'
                                                   'How specific elements (e.g., faces, background, clothing) have '
                                                   'changed\n'
                                                   '\n'
                                                   'Example:\n'
                                                   '\n'
                                                   '"The LUT introduces a warm golden cast across the entire image, '
                                                   'lifting midtones and softening shadows. Skin tones become richer '
                                                   'and more dimensional. Greens in the background are slightly muted, '
                                                   'giving the photo a cohesive and cinematic quality. The overall '
                                                   'atmosphere becomes warmer, more romantic, and emotionally '
                                                   'expressive."\n'
                                                   '\n'
                                                   'image_after\n'
                                                   'Provide a highly descriptive summary of the resulting image after '
                                                   'the LUT has been applied.\n'
                                                   'This should reflect the new tone, mood, improved or degraded '
                                                   'areas, and the overall aesthetic identity.\n'
                                                   '\n'
                                                   'visual_improvement (range -5 to +5)\n'
                                                   'Provide a numerical score reflecting how much better (or worse) '
                                                   'the after image looks compared to the before image:\n'
                                                   '\n'
                                                   '+5: Major visual improvement\n'
                                                   '\n'
                                                   '0: No noticeable change\n'
                                                   '\n'
                                                   '-5: Major degradation or artifact introduction\n'
                                                   'Use both objective quality and subjective mood as your basis.\n'
                                                   '\n'
                                                   'Example:\n'
                                                   '\n'
                                                   '"The graded image is warm and cinematic, with smooth skin tones, '
                                                   'deeper shadows, and a gentle glow. Lighting appears more '
                                                   'intentional, enhancing facial features and background ambience. '
                                                   'Color harmony improves significantly, and the image feels cohesive '
                                                   'and professionally graded."\n'
                                                   '\n'
                                                   '✅ Output Format\n'
                                                   '{\n'
                                                   '  "image_before": "The photo features a young woman sitting near a '
                                                   'window. The lighting is soft and diffuse, but the original image '
                                                   'lacks depth and has a cool, slightly underexposed palette. Skin '
                                                   'tones are pale, and the overall scene feels flat and '
                                                   'unpolished.",\n'
                                                   '  "effect_of_lut": "The LUT enhances warmth and deepens shadows, '
                                                   'giving the image a soft golden hue. Skin tones are revitalized '
                                                   'with a natural glow, and midtones are lifted slightly for added '
                                                   'clarity. Blues are subtly desaturated, making the subject stand '
                                                   'out more prominently against a less intrusive background.",\n'
                                                   '  "image_after": "The resulting image has a warm, balanced tone '
                                                   'with richer shadows and more lifelike skin. The photo now feels '
                                                   'cinematic and intimate, with improved depth, color separation, and '
                                                   'emotional appeal. It looks ready for editorial or wedding-grade '
                                                   'use.",\n'
                                                   '  "visual_improvement": 0.0\n'
                                                   '}',
                                   'model': 'gpt-5-mini'},
 'asst_iodVxzupHCzp8m1woUWWi9pd': {'description': None,
                                   'instructions': 'You are tasked with creating an optimal sequence of media files '
                                                   'for a reel based on the available media and user instructions.\n',
                                   'model': 'gpt-5'},
 'asst_nTPd7d65P4m9IfGqSRalQlrv': {'description': None,
                                   'instructions': 'You are tasked with creating flow sequence by assigning end_time '
                                                   'values to each file for a reel based on the available media, user '
                                                   'instructions and song beats.\n',
                                   'model': 'gpt-5'},
 'asst_nhcUwe2bOgc1iEx8aGf7WQIx': {'description': None,
                                   'instructions': 'Reduce the detailed input for a video editor GPT to 20% of the '
                                                   'original words. Avoid natural language; focus on essential info '
                                                   'that will aid in editing decisions.',
                                   'model': 'gpt-5-mini'},
 'asst_zM9jCka2JCV4u5gKeDOVwm70': {'description': None,
                                   'instructions': '"Your task is to create a video template in JSON format based on '
                                                   'the provided input parameters. The template must adhere to the '
                                                   'following structure and logic:\n'
                                                   '\n'
                                                   'Input Parameters:\n'
                                                   'user_prompt: The prompt provided by the user, describing the '
                                                   'requirements or theme of the video.\n'
                                                   'song_beats: A list of lists of beats from the selected song\n'
                                                   'songname: The name of the song to be used in the video.\n'
                                                   'Instructions:\n'
                                                   'user_prompt: Include the original user prompt as it is.\n'
                                                   'prompt_id : one word to describe user prompt or few words '
                                                   'seperated by "_"\n'
                                                   'song_beats:  you must choose one beat list that fits the template '
                                                   '(preferably around 1 second in length, unless the user specifies '
                                                   "otherwise). Make sure you don't change any beat and return all "
                                                   'beats of the selected beat_list. This is a critical step.\n'
                                                   "updated_prompt: Using the user's prompt, provide only the text "
                                                   "that fills in the blank in the sentence 'Make a video about "
                                                   "(blank)'. Only modify the user's prompt if there are grammatical "
                                                   "errors or if it doesn't read naturally; otherwise, use it as is. "
                                                   'Ensure no information from user prompt is omitted. Do not include '
                                                   "'Make a video about' in your response; only return the text that "
                                                   'goes in the blank.\n'
                                                   'Aspect Ratio: Default aspect ratio is [16,9] unless otherwise '
                                                   "specified in the user_prompt. Make sure you don't change unless "
                                                   'specified.\n'
                                                   'Resolution: Default resolution is 1080 unless otherwise specified '
                                                   "in the user_prompt. Make sure you don't change unless specified. \n"
                                                   'FPS : Default fps is 30 unless otherwise specified in the '
                                                   "user_prompt. Make sure you don't change unless specified. \n"
                                                   'Audio: Always set the audio file path as '
                                                   'Assets/Songs/{songname}.mp3, replacing {songname} with the actual '
                                                   'song title. \n'
                                                   'SRT: Always set the subtitle file path as '
                                                   'Assets/SRT/{songname}.srt, replacing {songname} with the actual '
                                                   'song title. \n'
                                                   'Sample JSON Structure:\n'
                                                   'json\n'
                                                   '{\n'
                                                   '  "template": {\n'
                                                   '    "user_prompt": "<Insert user prompt here>",\n'
                                                   '   "prompt_id": "prompt_id"\n'
                                                   '    "song_beats": "<Select one beat from the provided list>",\n'
                                                   '    "updated_prompt": "<Insert text selected for blank>",\n'
                                                   '    "aspect_ratio": [16,9], \n'
                                                   '    "resolution": 1080,\n'
                                                   '    "fps": "30",\n'
                                                   '    "audio": "Assets/Songs/songname.mp3",\n'
                                                   '    "srt": "Assets/SRT/songname.srt"\n'
                                                   '  }\n'
                                                   '}\n'
                                                   'Ensure the JSON structure is followed, using the input parameters '
                                                   'provided, and apply defaults unless the user has specified '
                                                   'otherwise."',
                                   'model': 'gpt-5-mini'},
 'asst_zO8O30jRCdQ70AkI7Kg6yTKP': {'description': None,
                                   'instructions': 'You are an expert data analysis assistant specialized in '
                                                   'clustering image data based on detailed JSON-formatted '
                                                   'descriptions.\n'
                                                   '\n'
                                                   'Each JSON object you receive includes:\n'
                                                   '\n'
                                                   'file_path: A unique identifier for each image.\n'
                                                   '\n'
                                                   'similarity: A numeric value indicating how closely the image '
                                                   'matches certain criteria.\n'
                                                   '\n'
                                                   'description: A detailed text analysis organized into clear '
                                                   'sections, including:\n'
                                                   '\n'
                                                   'Individuals present (their appearance, actions, and expressions).\n'
                                                   '\n'
                                                   'Event type, context, primary activity, and interaction details.\n'
                                                   '\n'
                                                   'Location category.\n'
                                                   '\n'
                                                   'Overall mood and atmosphere.\n'
                                                   '\n'
                                                   'An overall rating (1-10).\n'
                                                   '\n'
                                                   'faces: Names of any recognized individuals.\n'
                                                   '\n'
                                                   'orientation: The orientation of the image (landscape, portrait, '
                                                   'square).\n'
                                                   '\n'
                                                   'Based on these attributes, your task is to:\n'
                                                   '\n'
                                                   'Read and understand the provided JSON image descriptions '
                                                   'thoroughly.\n'
                                                   '\n'
                                                   'Identify distinct clusters that meaningfully categorize the '
                                                   'images, ensuring they portray a coherent story or sequential '
                                                   'narrative based on elements such as:\n'
                                                   '\n'
                                                   'Type of activity (romantic, friendly, individual reflection, '
                                                   'adventure, social gathering).\n'
                                                   '\n'
                                                   'Interaction level (individual, couples, small groups).\n'
                                                   '\n'
                                                   'Emotional tone and mood (romantic, joyful, relaxed, adventurous).\n'
                                                   '\n'
                                                   'Location type (nature, urban, indoor, landmark, artistic '
                                                   'setting).\n'
                                                   '\n'
                                                   'List the images within each cluster by their file_path.\n'
                                                   '\n'
                                                   'Provide a concise summary table showing each cluster name and the '
                                                   'number of images it contains.\n'
                                                   '\n'
                                                   'Determine a single, most suitable orientation (portrait or '
                                                   'landscape) for all clusters collectively based on the majority of '
                                                   'the best aesthetic representation across all clusters.\n'
                                                   '\n'
                                                   'Generate a prospective template for each cluster that visually '
                                                   'arranges the images to ensure they look cohesive, aesthetically '
                                                   'pleasing, and meaningful. Templates can:\n'
                                                   '\n'
                                                   'Only include one single image cropped appropriately. (single)\n'
                                                   '\n'
                                                   'Stack at max 2 landscape images.  (stacked)\n'
                                                   '\n'
                                                   'Overlay one image with a reduced size on top of another image. One '
                                                   'Image is foreground and the other is background so select so that '
                                                   'they complement each other. This template should only contain 2 '
                                                   'images. (overlay)\n'
                                                   '\n'
                                                   'Arrange exactly 4 portrait images in a box formation with one '
                                                   'image at each corner forming a grid. (square)\n'
                                                   '\n'
                                                   'Output the prospective templates in a JSON format specifying:\n'
                                                   '\n'
                                                   'file_paths: Images used.\n'
                                                   '\n'
                                                   'template: Type of visual arrangement used (single, stacked, '
                                                   'overlay, square).\n'
                                                   '\n'
                                                   'orientation: The single, unified orientation chosen for all '
                                                   'clusters (portrait or landscape).\n'
                                                   '\n'
                                                   'hashtag: The Best hashtags for the combined templates of images '
                                                   'are based on aesthetics and description. Limit hashtags to under '
                                                   '7\n'
                                                   '\n'
                                                   'caption: Catchy, under 10 words and innovative hashtags for '
                                                   'describing the template images setting\n'
                                                   '\n'
                                                   'Only provide Template data as output.\n'
                                                   '\n'
                                                   'Use the following JSON structure for the output:\n'
                                                   '\n'
                                                   '{\n'
                                                   '  "templates": [\n'
                                                   '    {\n'
                                                   '      "file_paths": ["image1.jpg", "image2.jpg"],\n'
                                                   '      "template": "stacked",\n'
                                                   '    },\n'
                                                   '    {\n'
                                                   '      "file_paths": ["image3.jpg"],\n'
                                                   '      "template": "single",\n'
                                                   '    },\n'
                                                   '    {\n'
                                                   '      "file_paths": ["image4.jpg", "image5.jpg"],\n'
                                                   '      "template": "overlay",\n'
                                                   '    },\n'
                                                   '    {\n'
                                                   '      "file_paths": ["image6.jpg", "image7.jpg", "image8.jpg", '
                                                   '"image9.jpg"],\n'
                                                   '      "template": "square",\n'
                                                   '    }\n'
                                                   '  ],\n'
                                                   '  "orientation": "landscape"\n'
                                                   '  "hashtags": ["hastag1", "hashtag2", ...],\n'
                                                   '  "caption": "caption for Instagram",\n'
                                                   '}\n'
                                                   '\n'
                                                   'Ensure your clusters and templates are coherent, meaningful, and '
                                                   'reflect the attributes mentioned in the JSON data. Prioritize '
                                                   'clarity, logical grouping, narrative sequencing, and aesthetic '
                                                   'arrangement in your response.\n'
                                                   '\n'
                                                   'Ensure your clusters and templates are coherent, meaningful, and '
                                                   'reflect the attributes mentioned in the JSON data. Prioritize '
                                                   'clarity, logical grouping, narrative sequencing, and aesthetic '
                                                   'arrangement in your response.',
                                   'model': 'gpt-5'}}

THREAD_STORE: Dict[str, List[Dict[str, Any]]] = {}

class GPT_assistant:
    def __init__(self, assistant_id: str = 'Default') -> None:
        self.assistant_id = assistant_id

        # Load .env from project root (mirrors old behavior)
        try:
            env_path = Path(__file__).parents[1] / '.env'
            load_dotenv(dotenv_path=env_path)
        except Exception:
            # Non-fatal if not present
            pass

        # Load API key: per-assistant env var → default OPENAI_API_KEY
        key_env_name = ASSISTANT_API_KEYS.get(assistant_id)
        api_key = os.getenv(key_env_name) if key_env_name else os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY or mapping env.")

        if OpenAI is not None:
            self.client = OpenAI(api_key=api_key)
        else:
            # Very old SDK fallback (not recommended); kept for wire-compat.
            import openai  # type: ignore
            openai.api_key = api_key
            self.client = openai

        logger.info("GPT_assistant init: assistant_id=%s key_var=%s client_impl=%s",
                    self.assistant_id,
                    key_env_name or "OPENAI_API_KEY",
                    "OpenAI" if OpenAI is not None else "openai-legacy")

        # Ensure a default config exists for this assistant_id
        if assistant_id not in ASSISTANT_CONFIGS:
            ASSISTANT_CONFIGS[assistant_id] = {
                "model": os.getenv("DEFAULT_OPENAI_MODEL", "gpt-5-mini"),
                "name": None,
                "description": None,
                "instructions": None,
                "temperature": 1,
                "top_p": 1,
                "response_format": "auto",
            }
        # Apply per-assistant default description/instructions if provided
        defaults = ASSISTANT_DEFAULTS.get(assistant_id)
        if defaults:
            for k,v in defaults.items():
                if v is not None:
                    ASSISTANT_CONFIGS[assistant_id][k] = v
        cfg_snapshot = ASSISTANT_CONFIGS[assistant_id]
        logger.debug("Config init: model=%s temp=%s top_p=%s has_instructions=%s",
                     cfg_snapshot.get("model"),
                     cfg_snapshot.get("temperature"),
                     cfg_snapshot.get("top_p"),
                     bool(cfg_snapshot.get("instructions")))

    # -------------------- Assistant config emulation --------------------
    def create_assistant(
        self,
        model: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Any]] = None,  # ignored by design (no tools)
        temperature: float = 1,
        top_p: float = 1,
        response_format: str = "auto",
    ) -> Dict[str, Any]:
        """Store/update local config for this assistant_id and return an assistant-like dict."""
        cfg = ASSISTANT_CONFIGS[self.assistant_id]
        cfg.update({
            "model": model,
            "name": name,
            "description": description,
            "instructions": instructions,
            # tools ignored (per your requirement)
            "temperature": temperature,
            "top_p": top_p,
            "response_format": response_format,
        })
        # Return an object similar to the old Assistants API resource
        logger.info("create_assistant: id=%s model=%s", self.assistant_id, cfg.get("model"))
        logger.debug("create_assistant: name=%s desc_len=%s instr_len=%s temp=%s top_p=%s resp_fmt=%s",
                     cfg.get("name"),
                     len(cfg.get("description") or "") if isinstance(cfg.get("description"), str) else 0,
                     len(cfg.get("instructions") or "") if isinstance(cfg.get("instructions"), str) else 0,
                     cfg.get("temperature"), cfg.get("top_p"), cfg.get("response_format"))
        return {
            "id": f"asst_local_{self.assistant_id}",
            **cfg,
        }

    def update_assistant(
        self,
        assistant_id: str,
        tool_resources: Optional[Dict[str, Any]] = None,  # ignored
        model: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Any]] = None,  # ignored
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Respect the assistant_id parameter if provided; fall back to this instance's id
        target_id = assistant_id or self.assistant_id
        cfg = ASSISTANT_CONFIGS.get(target_id, ASSISTANT_CONFIGS.get(self.assistant_id, {}))
        if model is not None: cfg["model"] = model
        if name is not None: cfg["name"] = name
        if description is not None: cfg["description"] = description
        if instructions is not None: cfg["instructions"] = instructions
        if temperature is not None: cfg["temperature"] = temperature
        if top_p is not None: cfg["top_p"] = top_p
        if response_format is not None: cfg["response_format"] = response_format
        # tools and tool_resources intentionally ignored
        ASSISTANT_CONFIGS[target_id] = cfg
        logger.info("update_assistant: target=%s model=%s", target_id, cfg.get("model"))
        logger.debug("update_assistant: name=%s desc_len=%s instr_len=%s temp=%s top_p=%s resp_fmt=%s",
                     cfg.get("name"),
                     len(cfg.get("description") or "") if isinstance(cfg.get("description"), str) else 0,
                     len(cfg.get("instructions") or "") if isinstance(cfg.get("instructions"), str) else 0,
                     cfg.get("temperature"), cfg.get("top_p"), cfg.get("response_format"))
        return {"id": f"asst_local_{target_id}", **cfg}

    # -------------------- Vector store (kept for signature parity) --------------------
    def create_vector_store(
        self,
        file_ids: Optional[List[str]] = None,
        name: Optional[str] = None,
        expires_after: Optional[Dict[str, Any]] = None,
        chunking_strategy: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Minimal placeholder to keep interface; since you said no tools/file-search are attached,
        we simply return a stub object. If you later re-enable file_search, replace with
        `self.client.vector_stores.create(...)` and add files accordingly.
        """
        vs_id = f"vs_stub_{uuid.uuid4().hex[:12]}"
        logger.info("create_vector_store: stub id=%s files=%s name=%s", vs_id, len(file_ids or []), name)
        return {
            "id": vs_id,
            "name": name or vs_id,
            "status": "ready",
            "file_counts": {"in_progress": 0, "completed": len(file_ids or []), "failed": 0},
            "metadata": metadata or {},
        }

    # -------------------- Thread emulation --------------------
    def create_thread(self) -> Dict[str, Any]:
        thread_id = f"thread_{uuid.uuid4().hex}"
        THREAD_STORE[thread_id] = []  # list of message dicts
        logger.info("create_thread: %s", thread_id)
        return {"id": thread_id}

    def add_message_to_thread(self, thread: Union[str, Dict[str, Any]], content: Union[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        thread_id = thread["id"] if isinstance(thread, dict) else str(thread)
        if thread_id not in THREAD_STORE:
            THREAD_STORE[thread_id] = []
        # Store as a standard message blob
        THREAD_STORE[thread_id].append({
            "role": "user",
            "content": content,
        })
        try:
            summary = _summarize_message_content(content)
        except Exception:
            summary = "unknown"
        logger.debug("add_message_to_thread: thread=%s %s", thread_id, summary)
        return {"thread_id": thread_id, "status": "queued"}

    
    def add_image_to_thread(
    self,
    thread: Union[str, Dict[str, Any]],
    image_path: Union[str, Path],
    additional_content: str = ""
) -> Dict[str, Any]:
        """
        Adds an image to the conversation as a user message for the Responses API.
        Uses an `input_image` part with a base64 data URL via `image_url` (widely supported).
        Signature and return shape kept identical to your old method.
        """
        # Resolve thread id and ensure store
        thread_id = thread["id"] if isinstance(thread, dict) else str(thread)
        if thread_id not in THREAD_STORE:
            THREAD_STORE[thread_id] = []

        # Validate the image path
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")

        # Derive MIME type from extension (fallback to PNG if unknown)
        mime = _guess_mime(p.suffix)
        if mime == "application/octet-stream":
            mime = "image/png"

        # Read and encode image as base64
        data_bytes = p.read_bytes()
        b64 = base64.b64encode(data_bytes).decode("utf-8")

        # Some backends reject ultra-tiny images; log size to help debugging
        try:
            file_size = p.stat().st_size
        except Exception:
            file_size = len(data_bytes)
        logger.info(
            "add_image_to_thread: thread=%s image=%s mime=%s bytes=%s",
            thread_id, p.name, mime, file_size
        )

        # Build multi-part content (Responses API shape)
        parts: List[Dict[str, Any]] = []
        if additional_content:
            parts.append({"type": "input_text", "text": additional_content})

        # IMPORTANT: Use image_url with a data: URI (most compatible across SDK versions)
        parts.append({
            "type": "input_image",
            "image_url": f"data:{mime};base64,{b64}"
        })

        THREAD_STORE[thread_id].append({
            "role": "user",
            "content": parts,
        })

        return {"thread_id": thread_id, "status": "queued"}




    # -------------------- Files helper (kept for parity; not required for vision) --------------------
    def upload_file_to_storage(self, file_path: Union[str, Path], purpose: str = "user_data") -> Dict[str, Any]:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if hasattr(self.client, "files"):
            with p.open("rb") as f:
                uploaded = self.client.files.create(file=f, purpose=purpose)  # type: ignore[attr-defined]
            logger.info("upload_file_to_storage: uploaded id=%s name=%s", getattr(uploaded, "id", "?"), p.name)
            return {"id": uploaded.id, "filename": p.name}
        else:
            # Fallback stub if running on an older SDK
            return {"id": f"file_stub_{uuid.uuid4().hex[:12]}", "filename": p.name}

    # -------------------- Run & read response --------------------
    def run_thread_on_assistant(self, thread: Union[str, Dict[str, Any]], instructions: Optional[str] = None) -> Any:
        thread_id = thread["id"] if isinstance(thread, dict) else str(thread)
        messages = THREAD_STORE.get(thread_id, [])

        cfg = ASSISTANT_CONFIGS.get(self.assistant_id, {})
        model = cfg.get("model") or os.getenv("DEFAULT_OPENAI_MODEL", "gpt-5-mini")
        temperature = cfg.get("temperature", 1)
        top_p = cfg.get("top_p", 1)
        sys_instructions = instructions or cfg.get("instructions")

        # Build Responses API input
        input_payload: List[Dict[str, Any]] = []
        if sys_instructions:
            input_payload.append({"role": "system", "content": sys_instructions})

        # Normalize stored messages into Responses input format
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # Already a list of parts (e.g., text + input_image)
                input_payload.append({"role": role, "content": content})
            else:
                input_payload.append({"role": role, "content": str(content)})

        # If any image parts are present and the model is a mini variant with
        # known/spotty image support, upgrade to a vision-robust model.
        try:
            _has_image = any(
                isinstance(it.get("content"), list) and any(
                    isinstance(p, dict) and p.get("type") == "input_image"
                    for p in it.get("content", [])
                )
                for it in input_payload if isinstance(it, dict)
            )
        except Exception:
            _has_image = False
        

        # No streaming; a single blocking call
        if not input_payload:
            raise ValueError("run_thread_on_assistant called with empty thread and no instructions.")
        logger.info("run_thread_on_assistant: thread=%s messages=%s model=%s temp=%s top_p=%s has_sys=%s",
                    thread_id, len(messages), model, temperature, top_p, bool(sys_instructions))
        if hasattr(self.client, "responses"):
            try:
                resp = self.client.responses.create(
                model=model,
                input=input_payload,
                temperature=temperature,
                top_p=top_p,
                store=False,
            )
                logger.info("responses.create: ok thread=%s", thread_id)
            except Exception as e:
                logger.exception("responses.create failed: thread=%s error=%s", thread_id, e)
                raise
        else:
            # Very old SDK fallback (not recommended); approximate via chat.completions
            # This branch is kept only for wire-compatibility if OpenAI() class isn't available.
            msgs = []
            for item in input_payload:
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    # Collapse parts to text when using old chat.completions
                    text = " ".join(
                        (part.get("text") or ((part.get("image") or {}).get("url") if isinstance(part.get("image"), dict) else None) or part.get("image_url", "[image]")) if isinstance(part, dict) else str(part)
                        for part in content
                    )
                    content = text
                msgs.append({"role": role, "content": content})
            try:
                resp = self.client.chat.completions.create(model=model, messages=msgs, temperature=temperature, top_p=top_p)  # type: ignore[attr-defined]
                logger.info("chat.completions.create: ok thread=%s", thread_id)
            except Exception as e:
                logger.exception("chat.completions.create failed: thread=%s error=%s", thread_id, e)
                raise

        # Extract text and append to thread history
        assistant_text = getattr(resp, "output_text", None)
        if not assistant_text:
            # Fall back to inspecting output list or choices
            assistant_text = _extract_text_from_response(resp)
        logger.debug("assistant_text_len=%s", len(assistant_text or ""))

        THREAD_STORE.setdefault(thread_id, []).append({
            "role": "assistant",
            "content": assistant_text,
        })
        return resp

    def check_run_status_and_respond(self, thread: Union[str, Dict[str, Any]], run: Any) -> str:
        """In the Responses world, `run` is the response object. Return the assistant text."""
        text = getattr(run, "output_text", None)
        if text:
            logger.debug("check_run_status_and_respond: output_text_len=%s", len(text))
            return text
        extracted = _extract_text_from_response(run)
        logger.debug("check_run_status_and_respond: extracted_text_len=%s", len(extracted or ""))
        return extracted

# -------------------- helpers --------------------

def _truncate(val: Optional[str], limit: int = 300) -> str:
    if val is None:
        return ""
    s = str(val)
    return s if len(s) <= limit else s[:limit] + "…"

def _summarize_message_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    try:
        if isinstance(content, list):
            kinds = [c.get("type", "text") if isinstance(c, dict) else type(c).__name__ for c in content]
            return f"parts={len(content)} types={kinds}"
        if isinstance(content, str):
            return f"text_len={len(content)}"
        return f"type={type(content).__name__}"
    except Exception:
        return "unknown"

def _extract_text_from_response(resp: Any) -> str:
    """
    Best-effort extraction for both Responses API objects and Chat Completions.
    Tries, in order:
      1) resp.output_text (Responses convenience)
      2) Walk resp.output[*].content[*] and collect any .text or ['text']
      3) Chat Completions: resp.choices[0].message.content
    """
    # 1) Direct convenience
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt:
        return txt

    # 2) Walk structured output
    try:
        out = getattr(resp, "output", None)
        if not out and isinstance(resp, dict):
            out = resp.get("output")
        parts = []
        if isinstance(out, list):
            for item in out:
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        # object attrs or dicts
                        ptype = getattr(part, "type", None)
                        if ptype is None and isinstance(part, dict):
                            ptype = part.get("type")
                        ptext = getattr(part, "text", None)
                        if ptext is None and isinstance(part, dict):
                            ptext = part.get("text")
                        if ptype in ("output_text", "text") and isinstance(ptext, str):
                            parts.append(ptext)
                elif isinstance(content, str):
                    parts.append(content)
        if parts:
            return "".join(parts)
    except Exception:
        pass

    # 3) Chat Completions shape
    try:
        choices = getattr(resp, "choices", None) or (isinstance(resp, dict) and resp.get("choices"))
        if choices:
            first = choices[0]
            msg = getattr(first, "message", None) or (isinstance(first, dict) and first.get("message"))
            if msg:
                content = getattr(msg, "content", None) or (isinstance(msg, dict) and msg.get("content"))
                if isinstance(content, str):
                    return content
    except Exception:
        pass

    return str(resp)

def _guess_mime(suffix: str) -> str:
    s = suffix.lower().lstrip('.')
    if s in ("jpg", "jpeg"): return "image/jpeg"
    if s == "png": return "image/png"
    if s == "webp": return "image/webp"
    if s == "gif": return "image/gif"
    return "application/octet-stream"
