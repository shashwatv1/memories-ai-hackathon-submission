import os
import json
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import regex as re
from scipy.datasets import face
from GPT_assistant import GPT_assistant
import json
import cv2
from typing import Dict, Any, List
from datetime import datetime
import time
import shutil
import helper_functions
from pathlib import Path
import config
from instagrapi import Client
import subprocess
import tempfile
from moviepy.editor import VideoFileClip
import glob
from difflib import SequenceMatcher
import mediapipe as mp
import traceback
import logging

# Configure MediaPipe logging
mp.solutions.drawing_utils.DrawingSpec = lambda *args, **kwargs: None
logging.getLogger("mediapipe").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# config.whatsapp_recipient = 'bare_art_canvas'
# config.user_query = 'Aisolutions24'

# config.User_ID = "3_Darke"
# config.Chat_ID = "1_Wedding_Test"
# config.Reel_ID = "Reelgood_2"

global cl
cl = None

# with open('lut_tags.json', 'r') as f:
#     LUT_DB = json.load(f)
#     print("Lut DB loaded")

# def select_best_lut(image_tags):
#     """
#     Select the best LUT based on image tags and LUT database
    
#     Args:
#         image_tags (dict): Dictionary of image tags and their scores
        
#     Returns:
#         list: List of tuples containing (LUT filename, similarity score)
#     """
#     # Create tag space, skipping invalid entries
#     tag_space = sorted({tag for v in LUT_DB.values()
#                        if isinstance(v, dict) and "best_image_tags" in v
#                        for d in v["best_image_tags"] for tag in d})
    
#     # Create LUT vectors, skipping invalid entries
#     lut_vecs = {}
#     for fname, lut in LUT_DB.items():
#         if isinstance(lut, dict) and "best_image_tags" in lut:
#             tags = {k: v for d in lut["best_image_tags"] for k, v in d.items()}
#             lut_vecs[fname] = vectorise(tags, tag_space)
    
#     # Load and vectorise image tags
#     v = vectorise(image_tags, tag_space)
    
#     # Calculate similarities
#     sims = {
#         lut: (v @ L) / (np.linalg.norm(v)*np.linalg.norm(L)+1e-9)
#         for lut, L in lut_vecs.items()
#     }
    
#     # Return top match
#     return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:1]

def vectorise(tag_scores, tag_space):
    """Convert tag scores to normalized vector"""
    return np.array([tag_scores.get(t, 0)/100 for t in tag_space], float)


class ImageOrientation:
    """Class to identify and classify image orientation."""
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    SQUARE = "square"
    
    @staticmethod
    def get_orientation(image):
        """Determine the orientation of an image."""
        width, height = image.size
        
        if width > height:
            return ImageOrientation.LANDSCAPE
        elif height > width:
            return ImageOrientation.PORTRAIT
        else:
            return ImageOrientation.SQUARE

class ImageCropper:
    """
    Utility class for intelligent image cropping based on face detection.
    This class focuses on cropping images to specific aspect ratios while
    ensuring that faces are properly framed within the resulting crop.
    """
    
    @staticmethod
    def convert_bbox_to_face_format(bbox):
        """
        Convert bounding box format from [x1, y1, x2, y2] to (x, y, w, h).
        
        Args:
            bbox: List or tuple with format [x1, y1, x2, y2]
            
        Returns:
            Tuple (x, y, w, h) for OpenCV face format
        """
        if len(bbox) != 4:
            return None
        
        x1, y1, x2, y2 = bbox
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    
    @staticmethod
    def process_face_bounding_boxes(img_array, face_data):
        """
        Process pre-existing face bounding box data.
        
        Args:
            img_array: Numpy array representing the image (used only for dimensions)
            face_data: List of face detection results with bounding boxes
            
        Returns:
            List of (x, y, w, h) tuples for detected faces
        """
        if not face_data or len(face_data) == 0:
            print("No face data available for this image")
            return []
        
        faces = []
        height, width = img_array.shape[:2]
        
        # Process each face in the data
        for face in face_data:
            if "bbox" in face and len(face["bbox"]) == 4:
                # Convert bbox format from [x1, y1, x2, y2] to (x, y, w, h)
                x1, y1, x2, y2 = face["bbox"]
                
                # Validate coordinates are within image bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                # Only add valid faces
                if x2 > x1 and y2 > y1:
                    w = x2 - x1
                    h = y2 - y1
                    faces.append((int(x1), int(y1), int(w), int(h)))
            
        return faces
    
    @staticmethod
    def estimate_face_positions(img_array):
        """
        Make an educated guess for face positions when detection fails.
        Designed to work with various portrait-style images of people.
        
        Args:
            img_array: Numpy array representing the image
            
        Returns:
            List of estimated (x, y, w, h) face regions
        """
        height, width = img_array.shape[:2]
        
        # Estimate face width based on image dimensions
        # For portrait-oriented photos, faces are typically 15-25% of image width
        face_width = int(width * 0.2)
        face_height = face_width
        
        # Most photos place important subjects in the upper third
        # following the rule of thirds
        center_x = width // 2
        center_y = int(height * 0.3)  # Upper third of the image
        
        # Calculate position for a primary face
        face_x = center_x - (face_width // 2)
        face_y = center_y - (face_height // 2)
        
        print("No reliable face detection. Using estimated position.")
        
        # For landscape-oriented photos (width > height), consider multiple faces
        # side by side, especially in wedding photos or group photos
        if width > height * 1.2:
            # Estimate two faces side by side
            face_width = int(width * 0.15)  # Slightly smaller faces
            face_height = face_width
            
            left_face_x = int(width * 0.3) - (face_width // 2)
            right_face_x = int(width * 0.7) - (face_width // 2)
            face_y = center_y - (face_height // 2)
            
            print("Wide image detected. Estimating multiple faces.")
            return [
                (left_face_x, face_y, face_width, face_height),
                (right_face_x, face_y, face_width, face_height)
            ]
        
        # Return the estimated face position
        return [(face_x, face_y, face_width, face_height)]
    
    @staticmethod
    def create_face_importance_map(img_array, faces, face_weight=150.0, sigma_factor=0.6):
        """
        Create an importance map that highlights faces.
        
        Args:
            img_array: Numpy array representing the image
            faces: List of (x, y, w, h) tuples for detected faces
            face_weight: Weight for face importance
            sigma_factor: Factor to determine the Gaussian spread
            
        Returns:
            Numpy array with the face importance map
        """
        if img_array is None or len(img_array.shape) < 2:
            return np.zeros((1, 1), dtype=np.float32)
            
        height, width = img_array.shape[:2]
        face_importance = np.zeros((height, width), dtype=np.float32)
        
        # If no faces, return empty importance map
        if not faces:
            return face_importance
            
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # For multi-person photos, add extra importance to the area between faces
        if len(faces) >= 2:
            # Sort faces by x-coordinate
            sorted_faces = sorted(faces, key=lambda f: f[0])
            
            # Calculate centers of faces
            face_centers = [(x + w//2, y + h//2) for (x, y, w, h) in sorted_faces]
            
            # For each adjacent pair of faces, check proximity
            for i in range(len(face_centers) - 1):
                x1, y1 = face_centers[i]
                x2, y2 = face_centers[i+1]
                
                # Check if faces are close enough to be possibly connected
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                # Only add between-faces importance if faces are reasonably close
                if dx < width * 0.3 and dy < height * 0.3:
                    # Calculate midpoint between faces
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    
                    # Size of the area between faces
                    area_size = max(dx, dy) * 0.8
                    
                    # Add a Gaussian centered between the faces
                    # This ensures the space between faces gets high importance
                    sigma = area_size * 0.5
                    between_gaussian = np.exp(-(
                        ((x_coords - mid_x) ** 2 + (y_coords - mid_y) ** 2) /
                        (2 * sigma ** 2)
                    ))
                    
                    # Add to importance map with a high weight
                    face_importance += between_gaussian * (face_weight * 0.8)
        
        # Add importance for each face
        for (x, y, w, h) in faces:
            # Calculate the center of the face
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Size of the face (used for scaling the Gaussian)
            face_size = max(w, h)
            
            # Calculate sigma based on face size
            sigma = max(1.0, face_size * sigma_factor)  # Ensure sigma is never zero
            
            # Create a Gaussian importance for this face
            gaussian = np.exp(-(
                ((x_coords - face_center_x) ** 2 + (y_coords - face_center_y) ** 2) /
                (2 * sigma ** 2)
            ))
            
            # Add to the total importance map
            face_importance += gaussian * face_weight
            
            # Add slightly increased importance above the face to include the top of the head
            if h > 0:  # Valid height
                # Add a smaller Gaussian above the face
                top_center_y = max(0, y - h // 4)  # Position slightly above the face
                top_gaussian = np.exp(-(
                    ((x_coords - face_center_x) ** 2 + (y_coords - top_center_y) ** 2) /
                    (2 * (sigma * 0.7) ** 2)  # Smaller spread for the top area
                ))
                
                # Add to importance map with lower weight
                face_importance += top_gaussian * (face_weight * 0.4)
        
        # If we have a single detected face, consider that there might be an undetected
        # face nearby. Add importance regions next to the face.
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            face_size = max(w, h)
            sigma = max(1.0, face_size * sigma_factor)
            
            # Add importance regions to the sides with larger width
            for offset_x in [-w, w]:  # Check both left and right
                side_center_x = min(max(0, face_center_x + offset_x), width-1)
                
                # Create a Gaussian importance for potential undetected face
                side_gaussian = np.exp(-(
                    ((x_coords - side_center_x) ** 2 + (y_coords - face_center_y) ** 2) /
                    (2 * (sigma * 0.8) ** 2)
                ))
                
                # Add to importance map with medium weight
                face_importance += side_gaussian * (face_weight * 0.4)
        
        # Normalize to prevent overflow
        if np.max(face_importance) > 0:
            face_importance = face_importance / np.max(face_importance) * face_weight
            
        return face_importance

    @staticmethod
    def adjust_to_exact_ratio(x1, y1, x2, y2, target_ratio, max_width, max_height):
        """
        Adjust a bounding box to match the exact target ratio.
        
        Args:
            x1, y1, x2, y2: Current bounding box coordinates
            target_ratio: Target width/height ratio
            max_width, max_height: Maximum image dimensions
            
        Returns:
            Tuple (x1, y1, x2, y2) with adjusted coordinates
        """
        width = x2 - x1
        height = y2 - y1
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.01:
            # Already close enough to target ratio
            return (int(x1), int(y1), int(x2), int(y2))
        
        if current_ratio > target_ratio:
            # Too wide, need to adjust width
            new_width = height * target_ratio
            diff = width - new_width
            x1 += diff / 2
            x2 -= diff / 2
        else:
            # Too tall, need to adjust height
            new_height = width / target_ratio
            diff = height - new_height
            y1 += diff / 2
            y2 -= diff / 2
        
        # Ensure we don't go outside image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(max_width, x2)
        y2 = min(max_height, y2)
        
        return (int(x1), int(y1), int(x2), int(y2))

    @staticmethod
    def crop_image(img, target_ratio, body_margin=0.5, min_zoom=0.7, face_data=None):
        """
        Crop an image to a target aspect ratio ensuring all faces are visible with
        extra margin to include bodies.
        
        Args:
            img: PIL Image object
            target_ratio: Target width/height ratio (width/height)
            body_margin: Margin factor to add around faces (as a percentage of face size)
            min_zoom: Minimum zoom level (1.0 = no zoom out, 0.5 = half size)
            face_data: Optional list of pre-detected faces with bounding boxes
            
        Returns:
            PIL Image: Cropped image with all faces visible
        """
        if img is None:
            raise ValueError("Input image cannot be None")
            
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # Get original dimensions
        orig_height, orig_width = img_array.shape[:2]
        orig_ratio = orig_width / orig_height
        
        # Calculate current aspect ratio
        current_ratio = img.width / img.height
        
        # If already at target ratio (within small tolerance), return original
        if abs(current_ratio - target_ratio) < 0.01:
            return img
        
        # Handle raw bbox format from face_data if it's directly provided as a list
        faces = []
        if isinstance(face_data, list) and len(face_data) == 4 and all(isinstance(x, (int, float)) for x in face_data):
            # Direct bbox format [x1, y1, x2, y2]
            x1, y1, x2, y2 = face_data
            w = x2 - x1
            h = y2 - y1
            faces = [(int(x1), int(y1), int(w), int(h))]
        elif isinstance(face_data, dict) and "bbox" in face_data:
            # Single face in dictionary format
            bbox = face_data["bbox"]
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                faces = [(int(x1), int(y1), int(w), int(h))]
        elif face_data:
            # Process standard face data format (list of dictionaries)
            faces = ImageCropper.process_face_bounding_boxes(img_array, face_data)
        
        # If no faces found in the provided data, try to estimate face positions
        if not faces or len(faces) == 0:
            faces = ImageCropper.estimate_face_positions(img_array)
        
        if not faces or len(faces) == 0:
            # Fallback to center crop if no faces detected/estimated
            print("No faces detected. Using center crop.")
            if target_ratio > current_ratio:  # Need to crop height
                crop_width = orig_width
                crop_height = int(crop_width / target_ratio)
                x1 = 0
                y1 = (orig_height - crop_height) // 2
            else:  # Need to crop width
                crop_height = orig_height
                crop_width = int(crop_height * target_ratio)
                x1 = (orig_width - crop_width) // 2
                y1 = 0
            
            x2 = x1 + crop_width
            y2 = y1 + crop_height
            return img.crop((x1, y1, x2, y2))
        
        # Calculate bounding box containing all faces
        min_x = min([x for x, y, w, h in faces])
        min_y = min([y for x, y, w, h in faces])
        max_x = max([x + w for x, y, w, h in faces])
        max_y = max([y + h for x, y, w, h in faces])
        
        # Calculate average face size for margin calculation
        avg_face_width = sum([w for x, y, w, h in faces]) / len(faces)
        avg_face_height = sum([h for x, y, w, h in faces]) / len(faces)
        
        # Calculate if any face is too large compared to the image dimensions
        largest_face_width = max([w for x, y, w, h in faces])
        largest_face_height = max([h for x, y, w, h in faces])
        
        # If largest face covers more than 40% of either dimension, adjust min_zoom
        if largest_face_width > orig_width * 0.4 or largest_face_height > orig_height * 0.4:
            # Calculate face ratio to image size
            face_ratio = max(largest_face_width / orig_width, largest_face_height / orig_height)
            # Adjust min_zoom based on face size (smaller value = more zoom out)
            adjusted_min_zoom = min(0.9, face_ratio * 1.5)  # Adjust with some margin
            min_zoom = min(min_zoom, adjusted_min_zoom)

        # For multiple faces, potentially increase margins to keep them all in frame
        if len(faces) > 1:
            # Calculate the spread of faces
            face_spread_x = max_x - min_x
            face_spread_y = max_y - min_y
            
            # If faces are spread out, increase margins
            if face_spread_x > orig_width * 0.5 or face_spread_y > orig_height * 0.5:
                body_margin = max(body_margin, 0.7)  # Increase margin for widely spread faces

        # Add margins for body visibility
        # Use larger margins for top and bottom to ensure faces are fully included
        margin_x = avg_face_width * body_margin * 1.2
        margin_y_top = avg_face_height * body_margin * 1.5  # Increased top margin
        margin_y_bottom = avg_face_height * body_margin * 2.5  # Increased bottom margin for bodies
        
        # Expanded bounding box with margins
        box_x1 = max(0, min_x - margin_x)
        box_y1 = max(0, min_y - margin_y_top)
        box_x2 = min(orig_width, max_x + margin_x)
        box_y2 = min(orig_height, max_y + margin_y_bottom)
        
        # Calculate current box dimensions
        box_width = box_x2 - box_x1
        box_height = box_y2 - box_y1
        box_ratio = box_width / box_height
        
        # Adjust box to match target ratio while containing all faces with margins
        if box_ratio > target_ratio:
            # Box is wider than target ratio, need to increase height
            # Center the height adjustment
            required_height = box_width / target_ratio
            extra_height = required_height - box_height
            box_y1 = max(0, box_y1 - extra_height / 2)
            box_y2 = min(orig_height, box_y2 + extra_height / 2)
        else:
            # Box is taller than target ratio, need to increase width
            # Center the width adjustment
            required_width = box_height * target_ratio
            extra_width = required_width - box_width
            box_x1 = max(0, box_x1 - extra_width / 2)
            box_x2 = min(orig_width, box_x2 + extra_width / 2)
        
        # Recalculate dimensions
        box_width = box_x2 - box_x1
        box_height = box_y2 - box_y1
        box_ratio = box_width / box_height
        
        # Check if we need to zoom out to fit target ratio
        zoom_out_needed = (
            box_x1 <= 0 or box_y1 <= 0 or 
            box_x2 >= orig_width or box_y2 >= orig_height or
            abs(box_ratio - target_ratio) > 0.01
        )
        
        # Also force zoom out if the bounding box is very large compared to the image
        box_coverage = (box_width * box_height) / (orig_width * orig_height)
        
        # Lower the threshold to 90% to encourage more zooming out
        if box_coverage > 0.9:  # If bounding box covers more than 90% of the image
            zoom_out_needed = True
        
        # Always zoom out a bit for better composition with multiple faces
        if len(faces) >= 2:
            zoom_out_needed = True
        
        if zoom_out_needed:
            # Calculate how much we need to zoom out to include all faces with margins
            if target_ratio > orig_ratio:
                # Target is wider than original, constrained by width
                zoom_width = box_width / orig_width
                zoom_factor = zoom_width
            else:
                # Target is taller than original, constrained by height
                zoom_height = box_height / orig_height
                zoom_factor = zoom_height
            
            # Apply minimum zoom factor with increased zoom out
        # Reduce zoom factor by 20% to ensure we capture more of the surrounding area
            zoom_factor = max(zoom_factor, min_zoom) * 0.8
            
            # Create a new canvas with zoomed-out dimensions
            new_width = int(orig_width / zoom_factor)
            new_height = int(orig_height / zoom_factor)
            
            # Resize the original image to zoom out
            zoomed_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Calculate crop dimensions for target ratio
            if target_ratio > new_width / new_height:
                # Target is wider than zoomed image, crop height
                crop_width = new_width
                crop_height = int(crop_width / target_ratio)
            else:
                # Target is taller than zoomed image, crop width
                crop_height = new_height
                crop_width = int(crop_height * target_ratio)
            
            # Instead of center crop, position the crop to prioritize the face area
            # Calculate the scale factor between original and zoomed image
            scale_factor = new_width / orig_width
            
            # Scale the face box to the new dimensions
            scaled_box_x1 = box_x1 * scale_factor
            scaled_box_y1 = box_y1 * scale_factor
            scaled_box_x2 = box_x2 * scale_factor
            scaled_box_y2 = box_y2 * scale_factor
            
            # Calculate the center of the face box in the zoomed image
            face_center_x = (scaled_box_x1 + scaled_box_x2) / 2
            face_center_y = (scaled_box_y1 + scaled_box_y2) / 2
            
            # Position the crop box centered on the face center
            crop_x1 = max(0, min(new_width - crop_width, face_center_x - crop_width / 2))
            crop_y1 = max(0, min(new_height - crop_height, face_center_y - crop_height / 2))
            
            # Ensure we don't go out of bounds
            if crop_x1 + crop_width > new_width:
                crop_x1 = new_width - crop_width
            if crop_y1 + crop_height > new_height:
                crop_y1 = new_height - crop_height
                
            crop_x2 = crop_x1 + crop_width
            crop_y2 = crop_y1 + crop_height
            
            
            # Crop the zoomed-out image
            result = zoomed_img.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))
        else:
            # Just crop to the calculated box
            # Adjust to exact target ratio if needed
            adjusted_box = ImageCropper.adjust_to_exact_ratio(box_x1, box_y1, box_x2, box_y2, target_ratio, orig_width, orig_height)
            result = img.crop(adjusted_box)
        
        return result
    
class Template:
    """Base template class."""
    def __init__(self, output_size=None, background_color=(0, 0, 0), padding=10):

        self.output_size = output_size
        self.background_color = background_color
        self.padding = padding 
    
    def apply(self, images):
        """Apply template to images."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_cropped_image(self, img, target_ratio, face_data=None):
        """Crop an image to the target ratio."""
        return ImageCropper.crop_image(img, target_ratio, face_data=face_data)
    
    def get_average_color(self, img):
        """Get the average color of an image to use for padding."""
        # Convert to numpy array for calculations
        img_array = np.array(img)
        
        # Use the edge pixels to determine the padding color
        # This gives a better result than using the entire image
        height, width = img_array.shape[:2]
        
        # Create a mask for the border pixels
        border_size = min(20, min(width, height) // 10)  # At most 20px border or 10% of the smallest dimension
        
        # Extract border pixels
        top_border = img_array[:border_size, :, :]
        bottom_border = img_array[-border_size:, :, :]
        left_border = img_array[:, :border_size, :]
        right_border = img_array[:, -border_size:, :]
        
        # Combine all borders
        borders = np.concatenate([
            top_border.reshape(-1, 3),
            bottom_border.reshape(-1, 3),
            left_border.reshape(-1, 3),
            right_border.reshape(-1, 3)
        ])
        
        # Calculate average color
        avg_color = np.mean(borders, axis=0).astype(np.uint8)
        
        # Convert to RGB tuple
        return tuple(avg_color)
    
    def add_padding(self, img):
        """Add padding around the image using a color derived from the image."""
        if self.padding <= 0:
            return img
        
        # Get current dimensions
        width, height = img.size
        
        # Get dominant color from the image for padding
        padding_color = self.get_average_color(img)
        
        # Create new image with padding
        padded_width = width + 2 * self.padding
        padded_height = height + 2 * self.padding
        padded_img = Image.new("RGB", (padded_width, padded_height), padding_color)
        
        # Paste original image in center
        padded_img.paste(img, (self.padding, self.padding))
        
        return padded_img
    
class SingleImageTemplate(Template):
    """Template for using a single image with optional cropping to match orientation."""
    def __init__(self, output_size=(1080, 1080), background_color=(0, 0, 0)):
        super().__init__(output_size, background_color)

    def apply(self, images, faces=None):
        """
        Returns the first image, optionally cropped to match a specific orientation.
        
        Args:
            images: List of images (only the first one is used)
            
        Returns:
            The processed image with desired orientation
        """
        if not images:
            raise ValueError("This template requires at least 1 image")
        
        img = images[0]
        face_info = faces[0] if faces and len(faces) > 0 else None
        print(face_info)
        
        # If output_size is specified, crop and resize the image
        if self.output_size:
            output_width, output_height = self.output_size
            output_ratio = output_width / output_height
            
            # Get current image dimensions and ratio
            img_width, img_height = img.size
            img_ratio = img_width / img_height
            
            # Determine if we need to crop
            if abs(img_ratio - output_ratio) > 0.01:
                # Crop image to match the target aspect ratio
                img = self.get_cropped_image(img, output_ratio, face_data=face_info)
                
                # Resize to match the output dimensions
                img = img.resize((output_width, output_height), Image.LANCZOS)
        
        return img

class LandscapeStackTemplate(Template):
    """Template for stacking any number and orientation of images."""
    def __init__(self, output_size=(1080, 1080), background_color=(0, 0, 0)):
        super().__init__(output_size, background_color)
    
    def apply(self, images, faces=None):
        """
        Stack images vertically, handling both landscape and portrait orientations.
        Preserves aspect ratio by intelligently cropping images.
        
        Args:
            images: List of images of any orientation
            
        Returns:
            A portrait image created by stacking the input images
        """
        if len(images) < 1:
            raise ValueError(f"This template requires at least 1 image, {len(images)} provided")
        
        # Get dimensions
        output_width, output_height = self.output_size
        output_ratio = output_width / output_height
        
        # Create new image with correct size and background
        result = Image.new("RGB", (output_width, output_height), self.background_color)
        
        # Calculate equal height distribution for images (no spacing)
        image_height = output_height // len(images)
        
        # Process each image
        for i, img in enumerate(images):
            
            face_info = faces[i] if faces and i < len(faces) else None
            print(face_info)
            
            # Calculate position
            y_pos = i * image_height
            
            # Check if image needs rotation correction
            img_width, img_height = img.size
            img_orientation = ImageOrientation.PORTRAIT if img_height > img_width else ImageOrientation.LANDSCAPE
            
            # Ensure we're maintaining the correct orientation
            # No need to rotate landscape images - they should stay landscape
            
            # Get original aspect ratio
            img_width, img_height = img.size
            img_ratio = img_width / img_height
            
            # If aspect ratio doesn't match the cell, crop the image
            cell_ratio = output_width / image_height
            
            if abs(img_ratio - cell_ratio) > 0.01:  # Check if ratios are significantly different
                # Crop the image to match the cell ratio
                img = self.get_cropped_image(img, cell_ratio, face_data=face_info)
            
            # Resize to match the cell dimensions
            img = img.resize((output_width, image_height), Image.LANCZOS)
            
            # Paste image
            result.paste(img, (0, y_pos))
            
        return result
    
class OverlayTemplate(Template):
    """Template for overlaying an image on top of another image."""
    def __init__(self, output_size=(1080, 1080), background_color=(0, 0, 0), scale_factor=0.75, blur_radius=5):
        super().__init__(output_size, background_color)
        self.scale_factor = scale_factor  # How much to scale down the foreground image
        self.blur_radius = blur_radius    # Blur radius for the background
    
    def apply(self, images, faces=None):
        """
        Places a foreground image on top of a background image with a slight reduction in size.
        Blurs the background for better focus on the foreground.
        Preserves aspect ratios of both images using intelligent cropping.
        
        Args:
            images: List of images [foreground, background]
            faces: List of face detection data for each image
            
        Returns:
            A composite image with the foreground overlaid on the blurred background
        """
        if len(images) != 2:
            raise ValueError(f"This template requires exactly 2 images, {len(images)} provided")
        
        # Get the foreground and background images
        foreground = images[0]
        background = images[1]
        
        # Get face data if available
        fg_faces = faces[0] if faces and len(faces) > 0 else None
        bg_faces = faces[1] if faces and len(faces) > 1 else None
        
        # Get dimensions
        output_width, output_height = self.output_size
        output_ratio = output_width / output_height
        
        # Determine template orientation
        template_orientation = ImageOrientation.PORTRAIT if output_height > output_width else ImageOrientation.LANDSCAPE
        
        # Check foreground orientation
        fg_orientation = ImageOrientation.get_orientation(foreground)
        
        # If foreground doesn't match template orientation, log warning
        if fg_orientation != template_orientation:
            print(f"Warning: Foreground image orientation ({fg_orientation}) doesn't match template orientation ({template_orientation})")
        
        # Crop background to match output ratio if needed
        bg_width, bg_height = background.size
        bg_ratio = bg_width / bg_height
        
        if abs(bg_ratio - output_ratio) > 0.01:
            background = self.get_cropped_image(background, output_ratio, face_data=bg_faces)
        
        # Resize background to match output size
        background = background.resize((output_width, output_height), Image.LANCZOS)
        
        # Apply blur effect to background
        # Note: This requires PIL's ImageFilter module
        blurred_background = background.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        
        # Create new image with background filling entire canvas
        result = Image.new("RGB", (output_width, output_height), self.background_color)
        result.paste(blurred_background, (0, 0))
        
        # Determine foreground target ratio based on template orientation
        fg_target_ratio = 1.15
        
        # Crop foreground to match target ratio if needed
        fg_width, fg_height = foreground.size
        fg_ratio = fg_width / fg_height
        
        if abs(fg_ratio - fg_target_ratio) > 0.01:
            foreground = self.get_cropped_image(foreground, fg_target_ratio, face_data=fg_faces)
        
        # Calculate scaled dimensions for foreground
        fg_width, fg_height = foreground.size
        
        if template_orientation == ImageOrientation.PORTRAIT:
            new_fg_width = int(output_width * self.scale_factor)
            new_fg_height = int(new_fg_width * fg_height / fg_width)
        else:
            new_fg_height = int(output_height * self.scale_factor)
            new_fg_width = int(new_fg_height * fg_width / fg_height)
        
        # Resize foreground
        foreground = foreground.resize((new_fg_width, new_fg_height), Image.LANCZOS)
        
        # Calculate position to center the foreground
        fg_x = (output_width - new_fg_width) // 2
        fg_y = (output_height - new_fg_height) // 2
        
        # Paste foreground image
        result.paste(foreground, (fg_x, fg_y))
        
        return result
    
class PortraitGridTemplate(Template):
    """Template for arranging images in a portrait grid (2x2 in portrait orientation)."""
    def __init__(self, output_size=(1080, 1080), background_color=(0, 0, 0), padding=10):
        super().__init__(output_size, background_color)
        self.padding = padding  # Padding between images
    
    def apply(self, images, faces=None):
        """
        Arranges images in a portrait-oriented grid (2 columns, 2 rows),
        placing images at each corner of the portrait canvas.
        
        Args:
            images: List of images to arrange in a portrait grid
            faces: List of face detection data for each image
            
        Returns:
            A portrait image with the input images arranged in the corners
        """
        num_images = len(images)
        if num_images < 1:
            raise ValueError("This template requires at least 1 image")
        
        # Grid dimensions for portrait layout - 2 columns, 2 rows
        grid_cols = 2
        grid_rows = 2
        
        # Get dimensions
        output_width, output_height = self.output_size
        
        # Calculate the average color from all images to use for padding
        avg_colors = []
        for img in images[:grid_cols*grid_rows]:  # Only use images that will be displayed
            avg_colors.append(self.get_average_color(img))
        
        # Average all the colors
        if avg_colors:
            avg_color = tuple(sum(c) // len(c) for c in zip(*avg_colors))
        else:
            avg_color = self.background_color
            
        # Create new image with correct size and background
        result = Image.new("RGB", (output_width, output_height), avg_color)
        
        # Calculate size for each cell in the grid, accounting for padding
        cell_width = (output_width - (grid_cols - 1) * self.padding) // grid_cols
        cell_height = (output_height - (grid_rows - 1) * self.padding) // grid_rows
        cell_ratio = cell_width / cell_height
        
        # Process and place each image
        for i, img in enumerate(images):
            if i >= grid_cols * grid_rows:  # Limit to 4 images
                break
            
            # Get face data for this image if available
            face_info = faces[i] if faces and i < len(faces) else None
                
            # Calculate position in grid
            row = i // grid_cols
            col = i % grid_cols
            
            # Check for EXIF orientation
            try:
                exif = img._getexif()
                if exif is not None and 274 in exif:  # 274 is the orientation tag
                    orientation = exif[274]
                    # Apply rotation based on EXIF orientation
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                pass
            
            # Get original aspect ratio
            img_width, img_height = img.size
            img_ratio = img_width / img_height
            
            # If aspect ratio doesn't match the cell, crop the image
            if abs(img_ratio - cell_ratio) > 0.01:
                img = self.get_cropped_image(img, cell_ratio, face_data=face_info)
            
            # Resize to match the cell dimensions
            img = img.resize((cell_width, cell_height), Image.LANCZOS)
            
            # Calculate position to place image
            x_pos = col * (cell_width + self.padding)
            y_pos = row * (cell_height + self.padding)
            
            # Paste image
            result.paste(img, (x_pos, y_pos))
        
        return result
    
class TemplateFactory:
    """Factory class for creating different templates."""
    @staticmethod
    def get_template(template_name, **kwargs):
        """
        Get template by name.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Additional parameters for the template
            
        Returns:
            Template instance
        """
        templates = {
            "stacked": LandscapeStackTemplate,
            "overlay": OverlayTemplate,
            "square": PortraitGridTemplate,
            "single": SingleImageTemplate
            # Add more templates here in the future
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available templates: {list(templates.keys())}")
        
        return templates[template_name](**kwargs)
    
class ImageProcessor:
    """Main class to process images using templates."""
    def __init__(self, template=None):
        self.template = template
    
    def set_template(self, template):
        """Set the template to use for processing."""
        self.template = template
    
    def process_from_json(self, json_data):
        """
        Process images according to JSON configuration.
        
        Args:
            json_data: Dictionary or JSON string with configuration
            
        Returns:
            Processed image according to template
        """
        # Parse JSON if string
        if isinstance(json_data, str):
            try:
                config = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")
        else:
            config = json_data
            
        # Validate required fields
        required_fields = ["file_paths", "template"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Get template
        template_name = config["template"]
        self.template = TemplateFactory.get_template(template_name)
        
        # Load images
        file_paths = config["file_paths"]
        
        face_data = config["face_data"] if "face_data" in config else None
        
        # Get final orientation if specified
        final_orientation = config.get("orientation", None)
        if isinstance(final_orientation, list):
            final_orientation = None  # Ignore if it's a list (old format)
            
        # Process images
        return self.process_images(file_paths, final_orientation, faces=face_data)
    
    def process_images(self, image_paths, final_orientation=None, faces=None):
        """
        Process images according to the template.
        
        Args:
            image_paths: List of paths to images to process
            final_orientation: Optional desired orientation for the final image
            faces: List of face detection data for each image
            
        Returns:
            Processed image according to template
        """
        if not self.template:
            raise ValueError("No template specified")
            
        # Load images
        images = []
        
        for i, path in enumerate(image_paths):
            try:
                # Use Pillow's EXIF-aware image loading to handle orientation correctly
                img = Image.open(Path(path))
                
                # Check if image has EXIF data with orientation
                try:
                    exif = img._getexif()
                    if exif is not None and 274 in exif:  # 274 is the orientation tag
                        orientation = exif[274]
                        # Apply rotation based on EXIF orientation
                        if orientation == 3:
                            img = img.rotate(180, expand=True)
                        elif orientation == 6:
                            img = img.rotate(270, expand=True)
                        elif orientation == 8:
                            img = img.rotate(90, expand=True)
                except (AttributeError, KeyError, IndexError):
                    # No EXIF or no orientation tag, continue without rotation
                    pass
                
                # Check if this is for the overlay template (needs special handling)
                if isinstance(self.template, OverlayTemplate) and i == 0:
                    # For overlay template, first image is foreground
                    template_orientation = (ImageOrientation.PORTRAIT 
                                           if self.template.output_size[1] > self.template.output_size[0] 
                                           else ImageOrientation.LANDSCAPE)
                    
                    # Check actual image dimensions
                    img_width, img_height = img.size
                    img_orientation = ImageOrientation.PORTRAIT if img_height > img_width else ImageOrientation.LANDSCAPE
                    
                    # For portrait template, if image is landscape but filename suggests portrait, rotate it
                    if template_orientation == ImageOrientation.PORTRAIT and img_orientation == ImageOrientation.LANDSCAPE:
                        # Check filename for hints (common for mobile photos)
                        filename = os.path.basename(path).lower()
                        if "portrait" in filename or "vertical" in filename or "img_" in filename:
                            print(f"Auto-rotating image {filename} to match template orientation")
                            img = img.rotate(270, expand=True)
                
                # For stack template, we need to make sure landscape images remain landscape
                if isinstance(self.template, LandscapeStackTemplate):
                    # Check if the image appears to be in the wrong orientation
                    img_width, img_height = img.size
                    filename = os.path.basename(path).lower()
                    
                    # If the image is portrait but should be landscape (for stack template)
                    if img_width < img_height:
                        if "landscape" in filename or "horizontal" in filename:
                            print(f"Auto-rotating image {filename} to maintain landscape orientation")
                            img = img.rotate(90, expand=True)
                
                # Apply subject enhancement if face data is available
                # if faces and i < len(faces):
                #     print(f"Enhancing image {i+1} with AI-powered subject enhancement")
                #     # Get face data for current image
                #     face_data = faces[i]
                #     # Initialize enhancer if not already done
                #     if not hasattr(self, 'enhancer'):
                #         self.enhancer = SubjectEnhancer()
                #     # Apply enhancement with default parameters
                #     img = self.enhancer.enhance_subject(
                #         img, 
                #         face_data,
                #         subject_brightness=1.15,
                #         subject_contrast=1.15,
                #         background_dim=0.85
                #     )
                # else:
                #     # Even without face data, try body segmentation
                #     print(f"Attempting body segmentation for image {i+1}")
                #     if not hasattr(self, 'enhancer'):
                #         self.enhancer = SubjectEnhancer()
                #     img = self.enhancer.enhance_subject(
                #         img, 
                #         None,
                #         subject_brightness=1.15,
                #         subject_contrast=1.15,
                #         background_dim=0.85
                #     )
                
                images.append(img)
            except Exception as e:
                print(f"Error loading image 12345 {path}: {e}")
        
        if not images:
            raise ValueError("No valid images provided")
        
        # Apply template
        result = self.template.apply(images, faces=faces)
        
        # Add padding around the result
        result = self.template.add_padding(result)
        
        return result
    
    def save_result(self, result, output_path):
        """Save the processed image to the specified path."""
        result.save(output_path)
        print(f"Result saved to {output_path}")
        
def process_images_from_json(json_input, output_path="output.jpg"):
    """
    Process images based on JSON configuration and save the result.
    
    Args:
        json_input: JSON string or dictionary with configuration
        output_path: Path to save the output image
        
    Returns:
        PIL.Image: The resulting processed image
        
    Example:
        >>> json_config = '''
        ... {
        ...     "file_paths": ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"],
        ...     "template": "square", 
        ...     "orientation": "square"
        ... }
        ... '''
        >>> result = process_images_from_json(json_config, "result.jpg")
    """
    try:
        processor = ImageProcessor()
        result = processor.process_from_json(json_input)
        if result:
            processor.save_result(result, output_path)
            return result
        else:
            print("No result generated. Check for errors above.")
            return None
    except Exception as e:
        print(f"Error processing images: {e}")
        return None
    
def GPT_story_cluster(filename) -> str:
    """Process text content through GPT for relevance"""
    with open(filename) as f:
        data = json.load(f)
        data = json.dumps(data["images"])
        assistant_id = 'asst_zO8O30jRCdQ70AkI7Kg6yTKP'
        gpt_assistant = GPT_assistant(assistant_id)
        thread = gpt_assistant.create_thread()
        gpt_assistant.add_message_to_thread(thread, data)
        run = gpt_assistant.run_thread_on_assistant(thread)
        response = gpt_assistant.check_run_status_and_respond(thread, run)
        return response.replace("```json\n", "").replace("\n```", "")
    
def process_image_path(template_data, path_mapping, face_assignments=None):
    """
    Convert short file paths in templates to complete paths using the provided mapping.
    Also includes face detection data if available.
    
    Args:
        template_data (str or dict): Template data containing file paths to convert
        path_mapping (dict): Mapping of filenames to their complete data
        face_assignments (dict, optional): Mapping of image paths to face detection data
        
    Returns:
        dict: Updated template data with complete file paths and face data
    """
    import re
    import json
    import os
    import traceback
    from difflib import SequenceMatcher
    
    # Parse the template data if it's a string
    if isinstance(template_data, str):
        try:
            # Handle potentially malformed JSON (missing closing bracket)
            template_data = re.sub(r',\s*$', '}', template_data)
            templates = json.loads(template_data)["templates"]
        except json.JSONDecodeError:
            error_msg = traceback.format_exc()
            print(f"Error parsing template JSON: {error_msg}")
            return {"templates": [], "error": "Failed to parse template data"}
    else:
        templates = template_data.get("templates", [])
        
    # Create a lookup map for fast access
    path_lookup = {}
    
    # Get all possible keys for matching
    all_keys = []
    
    # Populate the lookup map
    for filename, file_data in path_mapping.items():
        try:
            # Extract the prefix before the first dot as the key
            short_key = filename.split('.')[0].split('-')[0]
            path_lookup[short_key] = file_data["symlink_path"]
            all_keys.append(short_key)
            
            # Also add mapping for the full filename without extension
            filename_without_ext = filename.split('.')[0]
            path_lookup[filename_without_ext] = file_data["symlink_path"]
            all_keys.append(filename_without_ext)
        except Exception as e:
            print(f"Error processing path mapping for {filename}: {str(e)}")
    
    # Helper function to find closest match
    def find_closest_match(key, candidates):
        if not candidates:
            return None
        
        max_similarity = 0
        best_match = None
        
        for candidate in candidates:
            similarity = SequenceMatcher(None, key, candidate).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = candidate
        
        # Only return if the similarity is above a threshold
        return best_match if max_similarity > 0.6 else None
    
    # Process each template
    for template in templates:
        # Create a new list for complete paths
        complete_paths = []
        # New list to store face data for each image
        face_data_list = []
        
        for short_path in template["file_paths"]:
            try:
                # Extract the key (part before the extension)
                key = short_path.split('.')[0]
                full_path = None
                
                if key in path_lookup:
                    # Add the complete path if exact match found
                    full_path = path_lookup[key]
                    complete_paths.append(full_path)
                else:
                    # Try to find the closest match
                    closest_key = find_closest_match(key, all_keys)
                    
                    if closest_key:
                        full_path = path_lookup[closest_key]
                        print(f"Closest match found for {short_path}: using {closest_key} -> {full_path}")
                        complete_paths.append(full_path)
                
                # Look for face data for this image
                faces = []
                if face_assignments and full_path:
                    # Try to find an exact match
                    if full_path in face_assignments:
                        faces = face_assignments[full_path].get("faces", [])
                    else:
                        # Try partial matching for the filename
                        for path in face_assignments:
                            if os.path.basename(full_path) in path or os.path.basename(path) in full_path:
                                faces = face_assignments[path].get("faces", [])
                                break
                
                face_data_list.append(faces)
                
            except Exception as e:
                error_trace = traceback.format_exc()
                print(f"Error processing path {short_path}: {str(e)}\nTraceback: {error_trace}")
                complete_paths.append(short_path)
                face_data_list.append([])
        
        # Replace the short paths with complete paths
        template["file_paths"] = complete_paths
        # Add face data to the template
        template["face_data"] = face_data_list
    
    return {"templates": templates}

def cluster_story_pipeline():
    """Process images through GPT for relevance"""

    filename = f"{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/gpt_story_prompt.json"
    template_data = json.loads(GPT_story_cluster(filename))
    caption_data = {"caption":template_data["caption"], "hashtag":template_data["hashtags"]}
    
    with open(f"{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/captions.json", "w") as json_file:
        json.dump(caption_data, json_file, indent=4)
    
    print(config.Reel_ID, template_data)

    face_data = f"{config.User_ID}/{config.Chat_ID}/image_face_assignments.json"
    with open(face_data, 'r') as f:
        face_assignments = json.load(f)
        
    i = 0
    processed_results = []
    
    Lut_exists = False
    if os.path.exists(f'{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/Lut_Processed'):
        Lut_exists = True
    
    with open(f'{config.User_ID}/{config.Chat_ID}/Media/filename_mapping.json') as f:
        path_mapping = json.load(f)

        processed_template_data = process_image_path(template_data, path_mapping, face_assignments)
        print(processed_template_data)
            
        for template in processed_template_data["templates"]:
            if Lut_exists:
                template["file_paths"] = [
                    os.path.join(os.path.dirname(path).replace(os.path.join("Media", "Images"), os.path.join(config.Reel_ID, "LUT_Processed")), os.path.basename(path))
                    for path in template["file_paths"]
                ]
            template["orientation"] = template_data["orientation"]
            os.makedirs(f"{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/Templates/", exist_ok=True)
            process_images_from_json(json.dumps(template), f"{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/Templates/{config.Reel_ID}_output_{i}.jpg")
            # Add to processed results list with all relevant data
            processed_results.append({
                "template_index": i,
                "template_type": template.get("template", "unknown"),
                "file_paths": template.get("file_paths", []),
                "orientation": template.get("orientation", "unknown")
            })
            i += 1
        
    with open(f"{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/template_results.json", "w") as results_file:
        json.dump({
            "caption": template_data.get("caption", ""),
            "hashtags": template_data.get("hashtags", []),
            "processed_templates": processed_results
        }, results_file, indent=4)
    print("Template Images Created")

def upload_images_to_drive():
    """
    Uploads images from a local directory to Google Drive in a structured folder hierarchy:
    {target_folder}/User_ID/Chat_ID/Uploaded_Images/
    
    Args:
        service_account_file: Path to service account credentials file
        local_image_path: Path to the directory containing images to upload
        target_folder_id: ID of the target folder in Google Drive
        user_id: User ID for folder structure (uses directory name if None)
        chat_id: Chat ID for folder structure (uses timestamp if None)
        image_types: List of image file extensions to process
        batch_size: Number of images to process before pausing briefly (to avoid API limits)
        resume: Whether to resume from previous run if interrupted
        
    Returns:
        Dictionary with upload results
    """
    
    service_account_file = 'service-account.json'
    target_folder_id = '1be5p41JtvBbSxKpBaxrcotet0RZCzt5Y'  # Your shared drive folder ID
    local_image_path = f"./{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/Templates"
    image_types: List[str] = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    batch_size: int = 20
    resume: bool = True
    
    # Try to import the GoogleDriveServiceOperations class
    try:
        from GoogleServiceAPI import GoogleDriveServiceOperations
    except ImportError:
        return {
            "status": "error",
            "message": "GoogleServiceAPI module not found. Make sure it's in your Python path."
        }
    
    tracking_dir = os.path.join(config.User_ID, config.Chat_ID)
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_file = os.path.join(tracking_dir, "uploaded_images.json")
    
    # Load previously uploaded images if resuming
    uploaded_images = set()
    if resume and os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r') as f:
                uploaded_images = set(json.load(f))
                print(f"Resuming previous upload. Found {len(uploaded_images)} already uploaded images.")
        except Exception as e:
            print(f"Warning: Could not load tracking data: {str(e)}")
    
    try:
        # Initialize Drive operations
        drive_ops = GoogleDriveServiceOperations(service_account_file)
        
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
        
        # Create folder structure in Drive
        # 1. Create or find User_ID folder in target folder
        user_folder_id = find_or_create_folder(drive_ops.service, config.User_ID, target_folder_id)
        print(f"User folder ID: {user_folder_id}")
        
        # 2. Create or find Chat_ID folder in User_ID folder
        chat_folder_id = find_or_create_folder(drive_ops.service, config.Chat_ID, user_folder_id)
        print(f"Chat folder ID: {chat_folder_id}")
        
        # 3. Create or find Uploaded_Images folder in Chat_ID folder
        images_folder_name = "Template_Images"
        images_folder_id = find_or_create_folder(drive_ops.service, images_folder_name, chat_folder_id)
        print(f"Images folder ID: {images_folder_id}")
        
        # Get all image files from the local directory
        all_files = []
        for root, _, files in os.walk(local_image_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_types):
                    # Create relative path for organizing in subdirectories
                    rel_path = os.path.relpath(root, local_image_path)
                    if rel_path == '.':
                        rel_path = ''
                    
                    full_path = os.path.join(root, file)
                    all_files.append((full_path, rel_path, file))
        
        if not all_files:
            return {
                "status": "error",
                "message": f"No image files found in {local_image_path} with extensions {image_types}"
            }
        
        print(f"Found {len(all_files)} images to upload")
        
        # Process timestamp for logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results tracking
        upload_log = {
            "process_id": f"image_upload_{timestamp}",
            "start_time": datetime.now().isoformat(),
            "local_image_path": local_image_path,
            "uploaded_files": [],
            "errors": [],
            "skipped_files": []
        }
        
        # Track folder structure in log
        upload_log["folder_structure"] = {
            "target_folder": {
                "id": target_folder_id,
                "name": folder_name
            },
            "user_folder": {
                "id": user_folder_id,
                "name": config.User_ID
            },
            "chat_folder": {
                "id": chat_folder_id,
                "name": config.Chat_ID
            },
            "images_folder": {
                "id": images_folder_id,
                "name": images_folder_name,
                "link": f"https://drive.google.com/drive/folders/{images_folder_id}"
            }
        }
        
        # Upload images
        total_images = len(all_files)
        images_processed = 0
        current_batch = 0
        
        for full_path, rel_path, file_name in all_files:
            try:
                # Skip if already uploaded (for resumption)
                file_hash = f"{rel_path}|{file_name}"
                if resume and file_hash in uploaded_images:
                    print(f"Skipping already uploaded image: {file_name}")
                    upload_log['skipped_files'].append({
                        "file_name": file_name,
                        "path": rel_path,
                        "reason": "Already uploaded"
                    })
                    images_processed += 1
                    continue
                
                images_processed += 1
                current_batch += 1
                print(f"Processing image ({images_processed}/{total_images}): {file_name}")
                
                # Handle subdirectories by creating the same structure in Drive
                target_folder = images_folder_id
                
                if rel_path:
                    # Split the relative path into components and create each folder level
                    path_parts = rel_path.split(os.sep)
                    current_parent = images_folder_id
                    
                    for part in path_parts:
                        if part:  # Skip empty parts
                            current_parent = find_or_create_folder(drive_ops.service, part, current_parent)
                    
                    target_folder = current_parent
                
                # Upload the image
                try:
                    upload_result = drive_ops.upload_file(
                        full_path,
                        parent_folder_id=target_folder,
                        file_name=file_name
                    )
                    
                    if upload_result['status'] == 'success':
                        upload_log['uploaded_files'].append({
                            "file_name": file_name,
                            "relative_path": rel_path,
                            "drive_id": upload_result['data']['id'],
                            "drive_link": f"https://drive.google.com/file/d/{upload_result['data']['id']}/view"
                        })
                        
                        # Mark as uploaded
                        uploaded_images.add(file_hash)
                        with open(tracking_file, 'w') as f:
                            json.dump(list(uploaded_images), f)
                        
                        print(f"Uploaded image {images_processed}/{total_images}: {file_name}")
                    else:
                        upload_log['errors'].append({
                            "type": "upload_error",
                            "file": file_name,
                            "path": rel_path,
                            "error": upload_result.get('message')
                        })
                        print(f"Error uploading image: {file_name} - {upload_result.get('message')}")
                except Exception as e:
                    error_msg = f"Error uploading {file_name}: {str(e)}"
                    upload_log['errors'].append({
                        "type": "upload_error",
                        "file": file_name,
                        "path": rel_path,
                        "error": error_msg
                    })
                    print(error_msg)
                
                # Pause briefly after each batch to avoid hitting API rate limits
                if current_batch >= batch_size:
                    current_batch = 0
                    print(f"Pausing briefly after batch of {batch_size} uploads...")
                    time.sleep(2)  # 2-second pause
                
            except Exception as e:
                error_msg = f"Error processing image {file_name}: {str(e)}"
                print(error_msg)
                upload_log['errors'].append({
                    "type": "processing_error",
                    "file": file_name,
                    "path": rel_path,
                    "error": error_msg
                })
        
        # Finalize upload log
        upload_log["end_time"] = datetime.now().isoformat()
        upload_log["duration"] = (datetime.fromisoformat(upload_log["end_time"]) - 
                                 datetime.fromisoformat(upload_log["start_time"])).total_seconds()
        
        # Save upload log
        log_filename = f"image_upload_log_{timestamp}.json"
        log_path = os.path.join(tracking_dir, log_filename)
        with open(log_path, 'w') as f:
            json.dump(upload_log, f, indent=2)
        
        # Upload the log file to Drive
        log_upload = drive_ops.upload_file(
            log_path,
            parent_folder_id=images_folder_id,
            file_name=log_filename
        )
        
        return {
            "status": "success",
            "process_id": upload_log["process_id"],
            "images_found": total_images,
            "images_uploaded": len(upload_log['uploaded_files']),
            "images_skipped": len(upload_log['skipped_files']),
            "errors": len(upload_log['errors']),
            "folder_structure": {
                "user_id": config.User_ID,
                "chat_id": config.Chat_ID,
                "images_folder": images_folder_name,
                "images_folder_id": images_folder_id,
                "images_folder_link": f"https://drive.google.com/drive/folders/{images_folder_id}"
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to upload images: {str(e)}"
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

def resize_image(image_path, size=(1080, 1080)):
    img = Image.open(image_path).convert("RGB")
    resized_path = tempfile.mktemp(suffix=".jpg")
    img.save(resized_path, "JPEG")
    return resized_path

def create_insta_images():
    """
    Logs into Instagram, resizes images to fit standard Instagram dimensions without cropping,
    and uploads images and videos as a draft feed post.

    :param username: Instagram username
    :param password: Instagram password
    :param media_files: List of file paths to images and/or video files
    :param caption: Caption for the draft post
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.mp4', '.mov')
    media_files = []
    temp_resized = []
    
    media_paths = [f"{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/Templates/"]
    
    with open(f"{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/captions.json", "r") as json_file:
        loaded_data = json.load(json_file)
        caption = loaded_data['caption']
        hashtags = loaded_data['hashtag']
    
    for tags in hashtags:
        caption += f" {tags}"
            
    for path in media_paths:
        print(f"Processing path: {path}")
        if os.path.isdir(path):
            print(f"Found directory: {path}")
            for file in sorted(os.listdir(path)):
                full_path = os.path.join(path, file)
                if full_path.lower().endswith(valid_extensions):
                    if full_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        resized = resize_image(full_path)
                        media_files.append(resized)
                        temp_resized.append(resized)
                    else:
                        media_files.append(full_path)
        elif os.path.isfile(path) and path.lower().endswith(valid_extensions):
            if path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                resized = resize_image(path)
                media_files.append(resized)
                temp_resized.append(resized)
            else:
                media_files.append(path)

    if not media_files:
        print("No valid media files found.")
        return

    print("Prepared media for upload:", media_files)

    try:
        if len(media_files) == 1:
            media = media_files[0]
            if media.lower().endswith(('.mp4', '.mov')):
                cl.video_upload(media, caption=caption, extra_data={'upload_id': '', 'is_draft': True})
            else:
                cl.photo_upload(media, caption=caption, extra_data={'upload_id': '', 'is_draft': True})
        else:
            cl.album_upload(media_files, caption=caption, extra_data={'upload_id': '', 'is_draft': True})

        print(f"Uploaded {len(media_files)} media items as a draft feed post.")
    except Exception as e:
        print("Failed to upload media:", str(e))
    finally:
        for file in temp_resized:
            if os.path.exists(file):
                os.remove(file)

class SubjectEnhancer:
    """Class to enhance human subjects in images while dimming the background."""
    
    def __init__(self):
        """Initialize MediaPipe Selfie Segmentation."""
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)  # 1 for landscape
    
    @staticmethod
    def create_face_mask(img_array, faces, padding_factor=2.0):
        """Create a mask highlighting face areas with padding."""
        height, width = img_array.shape[:2]
        face_mask = np.zeros((height, width), dtype=np.float32)
        
        # Process face data
        processed_faces = []
        if faces and isinstance(faces, list):
            for face in faces:
                if isinstance(face, dict) and 'bbox' in face:
                    bbox = face['bbox']
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        w = x2 - x1
                        h = y2 - y1
                        processed_faces.append((int(x1), int(y1), int(w), int(h)))
                elif isinstance(face, tuple) and len(face) == 4:
                    processed_faces.append(face)
        
        # Create face regions with padding
        for (x, y, w, h) in processed_faces:
            center_x = x + w/2
            center_y = y + h/2
            
            # Expand face region
            new_w = w * padding_factor
            new_h = h * padding_factor
            
            x1 = max(0, int(center_x - new_w/2))
            y1 = max(0, int(center_y - new_h/2))
            x2 = min(width, int(center_x + new_w/2))
            y2 = min(height, int(center_y + new_h/2))
            
            # Create gradient falloff for face region
            y_coords, x_coords = np.ogrid[0:height, 0:width]
            dist_from_center = np.sqrt(
                ((x_coords - (x1 + x2)/2)/(new_w/3))**2 +
                ((y_coords - (y1 + y2)/2)/(new_h/3))**2
            )
            falloff = np.clip(1 - dist_from_center, 0, 1)
            face_mask = np.maximum(face_mask, falloff)
        
        return face_mask
    
    def create_body_mask(self, img_array):
        """Create a mask for the entire body using MediaPipe Selfie Segmentation."""
        # Convert to RGB if needed and ensure uint8 type for MediaPipe
        if img_array.dtype == np.float32:
            # If the input is float32 in [0,1] range, convert to uint8
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if img_array.dtype == np.uint8 else img_array
        else:
            return np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
        
        try:
            # Process the image with MediaPipe
            results = self.selfie_segmentation.process(rgb_image)
            
            if results.segmentation_mask is not None:
                body_mask = results.segmentation_mask.astype(np.float32)
            else:
                body_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
            
            # Apply Gaussian blur for smoother edges
            body_mask = cv2.GaussianBlur(body_mask, (7, 7), 3)
            
            return body_mask
            
        except Exception as e:
            return np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
    
    def create_subject_mask(self, img_array, faces, padding_factor=2.0):
        """
        Create a combined mask using both face detection and body segmentation.
        
        Args:
            img_array: numpy array of the image
            faces: list of face detection data with bbox information
            padding_factor: how much to expand around the face
            
        Returns:
            A binary mask array where 1 indicates subject areas
        """
        # Get face and body masks
        face_mask = self.create_face_mask(img_array, faces, padding_factor)
        body_mask = self.create_body_mask(img_array)
        
        # Combine masks with higher weight for face regions
        combined_mask = np.maximum(face_mask, body_mask)
        
        # Ensure smooth transitions
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 2)
        
        return combined_mask
    
    def enhance_subject(self, img, face_data, subject_brightness=1.15, subject_contrast=1.15, background_dim=0.85):
        """
        Enhance subjects in the image while dimming the background.
        
        Args:
            img: PIL Image
            face_data: list of face detection data with bbox information
            subject_brightness: brightness factor for subjects
            subject_contrast: contrast factor for subjects
            background_dim: dimming factor for background
            
        Returns:
            PIL Image with enhanced subjects
        """
        try:
            # Convert PIL image to numpy array
            img_array = np.array(img)
            
            # Store original uint8 array for body segmentation
            img_array_uint8 = img_array.copy()
            
            # Convert to float32 and scale to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            # Get the mask using the uint8 version for MediaPipe
            mask = self.create_subject_mask(img_array_uint8, face_data)
            
            # Expand mask to 3 channels if needed
            if len(img_array.shape) == 3:
                mask = np.expand_dims(mask, axis=-1)
            
            # Create enhanced version
            enhanced = img_array.copy()
            
            # Convert to HSV for better color handling
            if len(img_array.shape) == 3:
                # Scale to [0, 255] and convert to uint8 for cv2
                enhanced_255 = (enhanced * 255).clip(0, 255)
                
                enhanced_uint8 = enhanced_255.astype(np.uint8)
                enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
                
                hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
                
                # Convert to float32 for calculations
                hsv = hsv.astype(np.float32)
                
                # Enhance saturation and value channels
                hsv[..., 1] = np.clip(hsv[..., 1] * 1.1, 0, 255)  # Saturation
                hsv[..., 2] = np.clip(hsv[..., 2] * subject_brightness, 0, 255)  # Value
                
                # Convert back to uint8 for cv2
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                enhanced_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                
                # Convert back to float32 [0, 1]
                enhanced = enhanced.astype(np.float32) / 255.0
            
            # Apply contrast enhancement
            enhanced = np.clip((enhanced - 0.5) * subject_contrast + 0.5, 0, 1)
            
            # Create dimmed version for background
            dimmed = img_array * background_dim
            
            # Blend enhanced subjects with dimmed background using mask
            result = enhanced * mask + dimmed * (1 - mask)
            
            # Ensure result is in [0, 1] range
            result = np.clip(result, 0, 1)
            
            # Convert back to uint8 [0, 255] range
            result = (result * 255).clip(0, 255).astype(np.uint8)
            
            # Convert back to PIL Image
            return Image.fromarray(result)
            
        except Exception as e:
            # Return original image if enhancement fails
            return img

def create_insta_video():
    """
    Logs into Instagram, resizes images to fit standard Instagram dimensions without cropping,
    and uploads images and videos as a draft feed post.

    :param username: Instagram username
    :param password: Instagram password
    :param media_files: List of file paths to images and/or video files
    :param caption: Caption for the draft post
    """    
    temp_resized = []
    
    # List all files in the directory
    mp4_files = glob.glob(os.path.join(f"{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/", "*.mp4"))
    
    print(mp4_files)
    if not mp4_files:
        return None
    
    # Get the most recent file based on modification time
    media = max(mp4_files, key=os.path.getmtime)
    print(media)
    caption = "My captions"

    print("Prepared media for upload:", media)
    
    try:
        media = reencode_video_for_instagram(media)
        cl.video_upload(media, caption=caption)
        print(f"Uploaded media items as a draft feed post.")
                
    except Exception as e:
        print("Failed to upload media:", str(e))
    finally:
        for file in temp_resized:
            if os.path.exists(file):
                os.remove(file)
     
def reencode_video_for_instagram(input_path):
    clip = VideoFileClip(input_path)
    # Resize and pad to 1080x1920 without cropping
    clip_resized = clip.resize(height=1920) if clip.h < clip.w else clip.resize(width=1080)
    final = clip_resized.on_color(
        size=(1080, 1920),
        color=(0, 0, 0),
        pos=("center", "center")
    )
    output_path = tempfile.mktemp(suffix=".mp4")
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path             
                 
def delete_folder():
    # Check if the folder exists
    folder_path = f"{config.User_ID}/{config.Chat_ID}/{config.Reel_ID}/Templates/"
    print(f"Checking if folder exists: {folder_path}")
    if os.path.exists(folder_path):
        try:
            # Remove the folder and all its contents
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' and all its contents have been deleted successfully!")
        except Exception as e:
            print(f"Error deleting folder: {e}")
    else:
        print(f"Folder '{folder_path}' does not exist.")

def insta_login():
    """
    Logs into Instagram using the provided credentials.
    Waits for OTP input if 2FA is triggered.
    """
    global cl
    cl = Client()

    # Define 2FA (OTP) handler
    def otp_handler(username):
        print("Two-factor authentication required.")
        otp = input("Enter the OTP sent to your device: ")
        return otp

    cl.two_factor_login = otp_handler

    try:
        cl.login(config.whatsapp_recipient, config.user_query)
        print("Logged into Instagram successfully!")
    except Exception as e:
        print(f"Login failed: {e}")

# cluster_story_pipeline()
# upload_images_to_drive()
# insta_login()
# create_insta_images()
# create_insta_video()
# delete_folder()