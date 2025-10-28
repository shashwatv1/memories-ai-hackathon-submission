import os
import json
import numpy as np
from sklearn.cluster import KMeans
import pickle
from collections import defaultdict
import cv2
from decorators import *
import math

@log_execution
@handle_errors
@sync_with_drive(
    inputs=[
        'reel_clusters.json',
        'all_reel_clusters.json',
        'average_reel_clusters.json',
        'good_reel_clusters.json',
        'very_good_reel_clusters.json',
        'best_reel_clusters.json'
    ],
    outputs=[
        # 'adjusted_reel_clusters.json',
        # 'adjusted_all_reel_clusters.json',
        # 'adjusted_average_reel_clusters.json',
        # 'adjusted_good_reel_clusters.json',
        # 'adjusted_very_good_reel_clusters.json',
        # 'adjusted_best_reel_clusters.json'
    ]
)
def adjust_clusters(input_json=None, output_json=None, max_cluster_size=50, image_threshold=0.8, min_video_duration=5.0):
    """
    Adjusts clusters based on specific criteria:
    1. Splits clusters larger than max_cluster_size
    2. Moves short videos from image-dominant clusters to appropriate clusters
    
    Args:
        input_json (str): Input JSON file (if None, processes all tier JSONs)
        output_json (str): Output JSON file
        max_cluster_size (int): Maximum allowed cluster size
        image_threshold (float): Threshold for image dominance (0.0-1.0)
        min_video_duration (float): Minimum video duration in seconds
    """
    base_path = os.path.join(config.User_ID, config.Chat_ID)
    
    # Process a specific file if provided, otherwise process all tier files
    if input_json and output_json:
        adjust_single_cluster_file(
            os.path.join(base_path, input_json),
            os.path.join(base_path, output_json),
            max_cluster_size,
            image_threshold,
            min_video_duration
        )
    else:
        # Process all tier files
        tiers = [
            # ('reel_clusters.json', 'adjusted_reel_clusters.json'),
            ('all_reel_clusters.json', 'adjusted_all_reel_clusters.json'),
            ('average_reel_clusters.json', 'adjusted_average_reel_clusters.json'),
            ('good_reel_clusters.json', 'adjusted_good_reel_clusters.json'),
            ('very_good_reel_clusters.json', 'adjusted_very_good_reel_clusters.json'),
            ('best_reel_clusters.json', 'adjusted_best_reel_clusters.json')
        ]
        
        for input_file, output_file in tiers:
            input_path = os.path.join(base_path, input_file)
            output_path = os.path.join(base_path, output_file)
            
            if os.path.exists(input_path):
                print(f"Processing {input_file}...")
                adjust_single_cluster_file(
                    input_path,
                    output_path,
                    max_cluster_size,
                    image_threshold,
                    min_video_duration
                )
            else:
                print(f"Skipping {input_file} - file not found")

def adjust_single_cluster_file(input_path, output_path, max_cluster_size, image_threshold, min_video_duration):
    """Process and adjust a single cluster file"""
    
    # Load cluster data
    try:
        with open(input_path, 'r') as f:
            cluster_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {input_path}: {str(e)}")
        return
    
    # Get original cluster info
    clusters = cluster_data.get('clusters', {})
    if not clusters:
        print(f"No clusters found in {input_path}")
        return
    
    print(f"Original clusters: {len(clusters)}")
    
    # Adjust clusters - this will modify the clusters dict
    adjusted_clusters = adjust_clusters_core(clusters, max_cluster_size, image_threshold, min_video_duration)
    
    # Update cluster data with adjusted clusters
    cluster_data['clusters'] = adjusted_clusters
    cluster_data['total_clusters'] = len(adjusted_clusters)
    
    # Add adjustment parameters to metadata
    if 'params' not in cluster_data:
        cluster_data['params'] = {}
    cluster_data['params']['adjusted'] = True
    cluster_data['params']['max_cluster_size'] = max_cluster_size
    cluster_data['params']['image_threshold'] = image_threshold
    cluster_data['params']['min_video_duration'] = min_video_duration
    
    # Save adjusted cluster data
    with open(output_path, 'w') as f:
        json.dump(cluster_data, f, indent=4)
    
    print(f"Adjusted clusters: {len(adjusted_clusters)}")
    print(f"Saved to {output_path}")

def adjust_clusters_core(clusters, max_cluster_size, image_threshold, min_video_duration):
    """Core algorithm to adjust clusters based on criteria"""
    
    # Create working copy of clusters
    adjusted_clusters = {}

    def get_video_duration(file_path):
        try:
            # First check if we can get duration from the video_face_assignments.json
            base_path = os.path.join(config.User_ID, config.Chat_ID)
            # Make sure base_path is absolute
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(base_path)
                
            json_path = os.path.join(base_path, "video_face_assignments.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        video_data = json.load(f)
                    
                    # Get the file ID (UUID)
                    file_id = os.path.basename(file_path).split('.')[0]
                    
                    # Search for matching video entry
                    for video_path, video_info in video_data.items():
                        if file_id in video_path:
                            # Found matching video in the JSON
                            if 'duration' in video_info:
                                return video_info['duration']
                            break
                except Exception as e:
                    print(f"Error reading video durations from JSON: {str(e)}")
            
            # Fall back to the original method if JSON method fails
            if not os.path.exists(file_path):
                # Try to handle relative paths
                alternative_path = os.path.join(base_path, file_path)
                
                # Also try finding the file in Media/Videos directory
                video_folder = os.path.join(base_path, "Media", "Videos")
                file_id = os.path.basename(file_path).split('.')[0]
                
                for vf in os.listdir(video_folder):
                    # Compare just the filename part without the extension
                    vf_base = os.path.splitext(vf)[0]
                    if vf_base == file_id:
                        alternative_path = os.path.join(video_folder, vf)
                        break
                
                if not os.path.exists(alternative_path):
                    print(f"Video file not found: {file_id}")
                    return 0.0
                file_path = alternative_path
                
            # Check file extension
            if not file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return 0.0  # Not a video file
                
            # Use cv2 to get video duration
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"Could not open video: {file_path}")
                return 0.0
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            return duration
        except Exception as e:
            print(f"Error getting video duration for {file_path}: {str(e)}")
            return 0.0
    
    # Function to detect if a file is a video
    def is_video_file(file_path):
        base_path = os.path.join(config.User_ID, config.Chat_ID)
        video_folder = os.path.join(base_path, "Media", "Videos")
        image_folder = os.path.join(base_path, "Media", "Images")
        
        # Get base filename without extension
        file_id = os.path.basename(file_path).split('.')[0]
        
        # Check if file exists in video directory
        for vf in os.listdir(video_folder):
            if vf.startswith(file_id):
                return True
                
        # Check if file exists in image directory
        for img in os.listdir(image_folder):
            if img.startswith(file_id):
                return False
                
        # If file not found in either directory, check extension
        return file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    # Function to calculate embeddings similarity (for finding nearby clusters)
    def calculate_similarity(cluster1, cluster2):
        # Count common faces
        faces1 = set()
        faces2 = set()
        
        for item in cluster1:
            faces1.update(item.get('faces', []))
        
        for item in cluster2:
            faces2.update(item.get('faces', []))
        
        common_faces = len(faces1.intersection(faces2))
        
        # Return a similarity measure
        if common_faces > 0:
            return common_faces / max(len(faces1), len(faces2))
        return 0
    
    # Function to split large clusters using K-means
    def split_large_cluster(cluster_id, items):
        # Extract descriptions and faces for embedding
        texts = []
        for item in items:
            desc = item.get('description', '')
            faces = ' '.join(item.get('faces', []))
            texts.append(f"{faces} {desc}")
        
        # Create a simple embedding
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        try:
            X = vectorizer.fit_transform(texts)
            
            # Apply K-means with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Split items into two new clusters
            cluster1 = [items[i] for i in range(len(items)) if labels[i] == 0]
            cluster2 = [items[i] for i in range(len(items)) if labels[i] == 1]
            
            return cluster1, cluster2
        except Exception as e:
            print(f"Error splitting cluster {cluster_id}: {str(e)}")
            # Fallback: split evenly
            mid = len(items) // 2
            return items[:mid], items[mid:]
    
    # Step 1: Split large clusters
    next_cluster_id = max(int(cid) for cid in clusters.keys()) + 1 if clusters else 0
    
    # First, identify clusters to split
    clusters_to_split = []
    for cluster_id, cluster_data in clusters.items():
        items = cluster_data.get('images', [])
        if len(items) > max_cluster_size:
            clusters_to_split.append(cluster_id)
    
    # Process regular-sized clusters first
    for cluster_id, cluster_data in clusters.items():
        if cluster_id not in clusters_to_split:
            adjusted_clusters[cluster_id] = cluster_data
    
    # Then split large clusters
    for cluster_id in clusters_to_split:
        items = clusters[cluster_id].get('images', [])
        cluster1, cluster2 = split_large_cluster(cluster_id, items)
        
        # Update first cluster
        adjusted_clusters[cluster_id] = {
            'size': len(cluster1),
            'topic': clusters[cluster_id].get('topic', ''),
            'images': cluster1
        }
        
        # Create new cluster for second part
        new_cluster_id = str(next_cluster_id)
        next_cluster_id += 1
        adjusted_clusters[new_cluster_id] = {
            'size': len(cluster2),
            'topic': f"{clusters[cluster_id].get('topic', '')} (Split)",
            'images': cluster2
        }
        
        print(f"Split cluster {cluster_id} ({len(items)} items) into clusters {cluster_id} ({len(cluster1)} items) and {new_cluster_id} ({len(cluster2)} items)")
    
    # Step 2: Check image vs video ratio and move short videos if needed
    # First, analyze each cluster to identify image-dominant clusters
    cluster_info = {}
    for cluster_id, cluster_data in adjusted_clusters.items():
        items = cluster_data.get('images', [])
        
        # Count images vs videos and identify short videos
        image_count = 0
        video_count = 0
        short_videos = []
        
        for i, item in enumerate(items):
            file_path = item.get('file_path', '')
            
            if is_video_file(file_path):
                video_count += 1
                duration = get_video_duration(file_path)
                if duration < min_video_duration:
                    short_videos.append((i, item, duration))
            else:
                image_count += 1
        
        total_items = image_count + video_count
        image_ratio = image_count / total_items if total_items > 0 else 0
        
        cluster_info[cluster_id] = {
            'image_count': image_count,
            'video_count': video_count,
            'image_ratio': image_ratio,
            'is_image_dominant': image_ratio >= image_threshold,
            'short_videos': short_videos
        }
        
        print(f"Cluster {cluster_id}: {image_count} images, {video_count} videos, {image_ratio:.2f} image ratio, {len(short_videos)} short videos")
    
    # Now move short videos from image-dominant clusters to appropriate clusters
    videos_to_move = []
    
    for cluster_id, info in cluster_info.items():
        if info['is_image_dominant'] and info['short_videos']:
            for video_idx, video_data, duration in info['short_videos']:
                videos_to_move.append((cluster_id, video_idx, video_data, duration))
    
    # Sort videos to move in reverse order of index (to avoid index shifts when removing)
    videos_to_move.sort(key=lambda x: x[1], reverse=True)
    
    for source_cluster_id, video_index, video_data, duration in videos_to_move:
        # Find the best target cluster that is not image-dominant
        best_target = None
        best_score = -1
        
        for target_id, target_info in cluster_info.items():
            if target_id == source_cluster_id or target_info['is_image_dominant']:
                continue
            
            # Calculate similarity score
            target_items = adjusted_clusters[target_id]['images']
            similarity = calculate_similarity([video_data], target_items)
            
            # Prefer clusters with more videos and higher similarity
            video_factor = target_info['video_count'] / (target_info['image_count'] + target_info['video_count'])
            combined_score = similarity * (1 + video_factor)
            
            if combined_score > best_score:
                best_score = combined_score
                best_target = target_id
        
        # If no suitable non-image-dominant cluster found, try any cluster
        if best_target is None:
            for target_id, target_info in sorted(cluster_info.items(), key=lambda x: x[1]['image_ratio']):
                if target_id == source_cluster_id:
                    continue
                
                target_items = adjusted_clusters[target_id]['images']
                similarity = calculate_similarity([video_data], target_items)
                
                if similarity > best_score:
                    best_score = similarity
                    best_target = target_id
        
        # Move the video if a suitable target was found
        if best_target:
            # Remove from source
            source_items = adjusted_clusters[source_cluster_id]['images']
            del source_items[video_index]
            
            # Add to target
            target_items = adjusted_clusters[best_target]['images']
            target_items.append(video_data)
            
            # Update cluster info
            cluster_info[source_cluster_id]['video_count'] -= 1
            cluster_info[best_target]['video_count'] += 1
            
            # Recalculate image ratios
            source_total = cluster_info[source_cluster_id]['image_count'] + cluster_info[source_cluster_id]['video_count']
            target_total = cluster_info[best_target]['image_count'] + cluster_info[best_target]['video_count']
            
            cluster_info[source_cluster_id]['image_ratio'] = cluster_info[source_cluster_id]['image_count'] / source_total if source_total > 0 else 0
            cluster_info[best_target]['image_ratio'] = cluster_info[best_target]['image_count'] / target_total if target_total > 0 else 0
            
            # Update is_image_dominant flag
            cluster_info[source_cluster_id]['is_image_dominant'] = cluster_info[source_cluster_id]['image_ratio'] >= image_threshold
            cluster_info[best_target]['is_image_dominant'] = cluster_info[best_target]['image_ratio'] >= image_threshold
            
            print(f"Moved video (duration: {duration:.1f}s) from cluster {source_cluster_id} to {best_target}")
        else:
            print(f"Could not find suitable target cluster for video in {source_cluster_id}")
    
    # Update all cluster sizes to ensure consistency
    for cluster_id, cluster_data in adjusted_clusters.items():
        cluster_data['size'] = len(cluster_data.get('images', []))
    
    # Step 3: Split video-dominant clusters based on total duration
    # def split_video_cluster_by_duration(cluster_id, items, target_splits):
    #     """Split a video-dominant cluster into multiple clusters based on total duration"""
    #     if target_splits <= 1:
    #         return {cluster_id: items}
        
    #     # Sort items by duration (longest first for more even distribution)
    #     items_with_duration = []
    #     for item in items:
    #         file_path = item.get('file_path', '')
    #         if is_video_file(file_path):
    #             duration = get_video_duration(file_path)
    #         else:
    #             # Images count as 1 second
    #             duration = 1.0
    #         items_with_duration.append((item, duration))
        
    #     # Sort by duration (descending)
    #     items_with_duration.sort(key=lambda x: x[1], reverse=True)
        
    #     # Create new clusters with balanced duration
    #     new_clusters = [[] for _ in range(target_splits)]
    #     cluster_durations = [0] * target_splits
        
    #     # Distribute items using a greedy approach (place in least filled cluster)
    #     for item, duration in items_with_duration:
    #         # Find the cluster with minimal duration so far
    #         min_idx = cluster_durations.index(min(cluster_durations))
    #         new_clusters[min_idx].append(item)
    #         cluster_durations[min_idx] += duration
        
    #     # Return the balanced clusters
    #     result = {}
    #     result[cluster_id] = new_clusters[0]  # Keep first part in original cluster
        
    #     # Create new clusters for remaining parts
    #     for i in range(1, target_splits):
    #         new_cluster_id = str(next_cluster_id + i - 1)
    #         result[new_cluster_id] = new_clusters[i]
        
    #     return result

    def split_video_cluster_by_duration(cluster_id, items, target_splits):
        """Split a video-dominant cluster using a hybrid of context and duration"""
        if target_splits <= 1:
            return {cluster_id: items}
        
        # First, get duration for each item
        items_with_data = []
        for item in items:
            file_path = item.get('file_path', '')
            if is_video_file(file_path):
                duration = get_video_duration(file_path)
            else:
                duration = 1.0  # Images count as 1 second
                
            # Extract text features
            desc = item.get('description', '')
            faces = ' '.join(item.get('faces', []))
            text_data = f"{faces} {desc}"
            
            items_with_data.append((item, duration, text_data))
        
        try:
            # Create text embeddings for context
            texts = [item[2] for item in items_with_data]
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(texts)
            
            # Apply K-means with more clusters than target_splits
            # This creates "content groups" that we'll try to keep together
            content_group_count = min(len(items), target_splits * 2)  # 2x more content groups than target clusters
            kmeans = KMeans(n_clusters=content_group_count, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Group items by content cluster
            content_groups = [[] for _ in range(content_group_count)]
            content_group_durations = [0.0] * content_group_count
            
            for i, (item, duration, _) in enumerate(items_with_data):
                group_idx = labels[i]
                content_groups[group_idx].append(item)
                content_group_durations[group_idx] += duration
            
            # Now distribute content groups to target clusters using duration-balancing
            result = {cluster_id: []}
            new_clusters = {str(next_duration_cluster_id + i): [] for i in range(target_splits - 1)}
            cluster_durations = {cid: 0.0 for cid in list(new_clusters.keys()) + [cluster_id]}
            
            # Sort content groups by duration (largest first)
            sorted_groups = sorted(zip(content_groups, content_group_durations), 
                                key=lambda x: x[1], reverse=True)
            
            # Distribute content groups to minimize duration variance
            for group, group_duration in sorted_groups:
                # Find target cluster with minimal duration
                target_id = min(cluster_durations, key=cluster_durations.get)
                if target_id == cluster_id:
                    result[cluster_id].extend(group)
                else:
                    new_clusters[target_id].extend(group)
                cluster_durations[target_id] += group_duration
            
            # Add all new clusters to result
            for new_id, new_items in new_clusters.items():
                if new_items:  # Only add non-empty clusters
                    result[new_id] = new_items
            
            return result
        
        except Exception as e:
            print(f"Error in hybrid splitting for cluster {cluster_id}: {str(e)}")
            # Fall back to simple duration-based distribution
            
            # Sort items by duration (longest first for more even distribution)
            items_with_duration = []
            for item in items:
                file_path = item.get('file_path', '')
                if is_video_file(file_path):
                    duration = get_video_duration(file_path)
                else:
                    # Images count as 1 second
                    duration = 1.0
                items_with_duration.append((item, duration))
            
            # Sort by duration (descending)
            items_with_duration.sort(key=lambda x: x[1], reverse=True)
            
            # Create new clusters with balanced duration
            new_clusters = [[] for _ in range(target_splits)]
            cluster_durations = [0] * target_splits
            
            # Distribute items using a greedy approach (place in least filled cluster)
            for item, duration in items_with_duration:
                # Find the cluster with minimal duration so far
                min_idx = cluster_durations.index(min(cluster_durations))
                new_clusters[min_idx].append(item)
                cluster_durations[min_idx] += duration
            
            # Return the balanced clusters
            result = {}
            result[cluster_id] = new_clusters[0]  # Keep first part in original cluster
            
            # Create new clusters for remaining parts
            for i in range(1, target_splits):
                new_cluster_id = str(next_duration_cluster_id + i - 1)
                result[new_cluster_id] = new_clusters[i]
            
            return result

    # Add to adjust_clusters_core after the code for Step 2
    print("Step 3: Splitting video-dominant clusters based on duration...")

    # Track clusters after Step 3
    final_clusters = {}
    next_duration_cluster_id = next_cluster_id  # Use next available ID

    # Process each cluster
    for cluster_id, cluster_data in adjusted_clusters.items():
        items = cluster_data.get('images', [])
        
        # Skip if this is an image-dominant cluster
        if cluster_info[cluster_id]['is_image_dominant']:
            final_clusters[cluster_id] = cluster_data
            continue
        
        # Calculate total duration of the cluster
        total_duration = 0
        for item in items:
            file_path = item.get('file_path', '')
            if is_video_file(file_path):
                duration = get_video_duration(file_path)
                total_duration += duration
            else:
                # Images count as 1 second
                total_duration += 1.0
        
        # Calculate number of target clusters based on total minutes (ceiling)
        total_minutes = total_duration / 60.0
        target_splits = math.ceil(total_minutes)
        
        if target_splits <= 1:
            # No need to split
            final_clusters[cluster_id] = cluster_data
            continue
        
        print(f"Splitting video cluster {cluster_id} with duration {total_duration:.1f}s ({total_minutes:.2f} min) into {target_splits} clusters")
        
        # Split the cluster
        split_result = split_video_cluster_by_duration(cluster_id, items, target_splits)
        
        # First part keeps original cluster ID and topic
        final_clusters[cluster_id] = {
            'size': len(split_result[cluster_id]),
            'topic': cluster_data.get('topic', ''),
            'images': split_result[cluster_id]
        }
        
        # Add remaining parts as new clusters
        for new_id, new_items in split_result.items():
            if new_id != cluster_id:
                final_clusters[new_id] = {
                    'size': len(new_items),
                    'topic': f"{cluster_data.get('topic', '')} (Duration Split {new_id})",
                    'images': new_items
                }
        
        # Update the next available cluster ID
        next_duration_cluster_id = max(int(cid) for cid in final_clusters.keys()) + 1

    # Replace adjusted_clusters with final_clusters
    adjusted_clusters = final_clusters

    # Add after "adjusted_clusters = final_clusters" and before the return statement:

    # Rebalance clusters to ensure more even duration distribution
    print("Performing final duration rebalancing...")

    # Gather all video-dominant clusters
    video_clusters = {}
    cluster_durations = {}

    for cluster_id, cluster_data in adjusted_clusters.items():
        items = cluster_data.get('images', [])
        
        # Skip empty clusters
        if not items:
            continue
        
        # Check if this is a video-dominant cluster from Step 3
        if "Duration Split" in cluster_data.get('topic', '') or not cluster_info.get(cluster_id, {}).get('is_image_dominant', True):
            # Calculate cluster duration
            total_duration = 0
            for item in items:
                file_path = item.get('file_path', '')
                if is_video_file(file_path):
                    duration = get_video_duration(file_path)
                    total_duration += duration
                else:
                    total_duration += 1.0  # Images count as 1 second
            
            video_clusters[cluster_id] = items
            cluster_durations[cluster_id] = total_duration
            print(f"Cluster {cluster_id}: duration = {total_duration:.1f}s")

    # Only rebalance if we have multiple video clusters
    if len(video_clusters) >= 2:
        # Check for either condition:
        # 1. Any cluster is longer than 90 seconds (1:30)
        # 2. One cluster is more than 3x longer than another
        max_duration = max(cluster_durations.values())
        min_duration = min(cluster_durations.values())
        
        needs_rebalance = False
        
        # Check condition 1: Any cluster > 90 seconds
        if max_duration > 90.0:
            print(f"Rebalancing needed: Cluster with {max_duration:.1f}s exceeds 90s limit")
            needs_rebalance = True
        
        # Check condition 2: Max/min ratio > 3
        elif max_duration > min_duration * 3.0:
            print(f"Rebalancing needed: Max ({max_duration:.1f}s) is more than 3x min ({min_duration:.1f}s)")
            needs_rebalance = True
        
        if needs_rebalance:
            # Create a flat list of all items with their durations
            all_items = []
            for cluster_id, items in video_clusters.items():
                for item in items:
                    file_path = item.get('file_path', '')
                    if is_video_file(file_path):
                        duration = get_video_duration(file_path)
                    else:
                        duration = 1.0
                    all_items.append((item, duration))
            
            # Sort by duration (descending)
            all_items.sort(key=lambda x: x[1], reverse=True)
            
            # Get the cluster IDs
            cluster_ids = list(video_clusters.keys())
            
            # Distribute items to balance durations
            rebalanced_clusters = {cid: [] for cid in cluster_ids}
            new_durations = {cid: 0.0 for cid in cluster_ids}
            
            # Use greedy algorithm to distribute items
            for item, duration in all_items:
                # Find cluster with minimum duration
                target_id = min(new_durations, key=new_durations.get)
                rebalanced_clusters[target_id].append(item)
                new_durations[target_id] += duration
            
            # Update the actual clusters
            for cluster_id, items in rebalanced_clusters.items():
                adjusted_clusters[cluster_id]['images'] = items
                adjusted_clusters[cluster_id]['size'] = len(items)
                print(f"Rebalanced cluster {cluster_id}: {len(items)} items, {new_durations[cluster_id]:.1f}s")

    # Clean up - remove any empty clusters
    return {cid: data for cid, data in adjusted_clusters.items() if data['size'] > 0}
    

# Example usage
# adjust_clusters()  # Process all tier files
# adjust_clusters("very_good_reel_clusters.json", "adjusted_very_good_reel_clusters.json")  # Process specific file


import os
import shutil
from decorators import log_execution, handle_errors

@log_execution
@handle_errors
@sync_with_drive(
    inputs=[
        # 'reel_clusters.json',
        # 'all_reel_clusters.json',
        # 'average_reel_clusters.json',
        # 'good_reel_clusters.json',
        # 'very_good_reel_clusters.json',
        # 'best_reel_clusters.json'
    ],
    outputs=[
        'unadjusted_all_reel_clusters.json',
        'unadjusted_average_reel_clusters.json',
        'unadjusted_good_reel_clusters.json',
        'unadjusted_very_good_reel_clusters.json',
        'unadjusted_best_reel_clusters.json'
    ]
)
def rename_cluster_files():
    """
    Renames cluster files after adjustment:
    1. Renames original files from 'name.json' to 'unadjusted_name.json'
    2. Renames adjusted files from 'adjusted_name.json' to 'name.json'
    """
    base_path = os.path.join(config.User_ID, config.Chat_ID)
    
    # Define file pairs (original, adjusted)
    file_pairs = [
        # ('reel_clusters.json', 'adjusted_reel_clusters.json'),
        ('all_reel_clusters.json', 'adjusted_all_reel_clusters.json'),
        ('average_reel_clusters.json', 'adjusted_average_reel_clusters.json'),
        ('good_reel_clusters.json', 'adjusted_good_reel_clusters.json'),
        ('very_good_reel_clusters.json', 'adjusted_very_good_reel_clusters.json'),
        ('best_reel_clusters.json', 'adjusted_best_reel_clusters.json')
    ]
    
    for original_file, adjusted_file in file_pairs:
        original_path = os.path.join(base_path, original_file)
        adjusted_path = os.path.join(base_path, adjusted_file)
        unadjusted_path = os.path.join(base_path, f"unadjusted_{original_file}")
        
        # Check if both files exist
        if os.path.exists(original_path) and os.path.exists(adjusted_path):
            print(f"Processing {original_file} and {adjusted_file}...")
            
            # Rename original to unadjusted
            try:
                shutil.copy2(original_path, unadjusted_path)
                print(f"Copied {original_file} to unadjusted_{original_file}")
                
                # Rename adjusted to original
                shutil.copy2(adjusted_path, original_path)
                print(f"Copied {adjusted_file} to {original_file}")
                
                # Optionally remove the adjusted file if no longer needed
                os.remove(adjusted_path)
                print(f"Removed {adjusted_file}")
            except Exception as e:
                print(f"Error renaming files: {str(e)}")
        else:
            if not os.path.exists(original_path):
                print(f"Skipping {original_file} - file not found")
            if not os.path.exists(adjusted_path):
                print(f"Skipping {adjusted_file} - file not found")

# Example usage
# Run this after adjust_clusters() to rename the files
# rename_cluster_files()