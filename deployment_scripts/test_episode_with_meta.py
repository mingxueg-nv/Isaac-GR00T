#!/usr/bin/env python3
"""
Test TensorRT accuracy on single episode frames using meta information
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
import time
import pandas as pd

from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.dataset import LeRobotSingleDataset
from deployment_scripts.trt_model_forward import setup_tensorrt_engines


def load_episode_info(dataset_path):
    """Load episode information"""
    episodes_file = Path(dataset_path) / "meta" / "episodes.jsonl"
    episodes = []
    
    with open(episodes_file, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))
    
    return episodes


def calculate_episode_frame_indices(episodes):
    """Calculate frame index ranges for each episode"""
    episode_ranges = []
    current_index = 0
    
    for episode in episodes:
        episode_index = episode['episode_index']
        length = episode['length']
        
        episode_ranges.append({
            'episode_index': episode_index,
            'start_index': current_index,
            'end_index': current_index + length - 1,
            'length': length
        })
        
        current_index += length
    
    return episode_ranges


def get_episode_frames_from_indices(dataset, episode_ranges, target_episode):
    """Get frames for specified episode based on index ranges"""
    # Find target episode index range
    target_range = None
    for episode_range in episode_ranges:
        if episode_range['episode_index'] == target_episode:
            target_range = episode_range
            break
    
    if target_range is None:
        print(f"❌ Episode {target_episode} not found")
        return []
    
    print(f"Episode {target_episode}: frames {target_range['start_index']} to {target_range['end_index']} (length: {target_range['length']})")
    
    # Get all frames for this episode
    episode_frames = []
    for i in range(target_range['start_index'], target_range['end_index'] + 1):
        try:
            frame_data = dataset[i]
            episode_frames.append(frame_data)
            if len(episode_frames) % 50 == 0:
                print(f"Loaded {len(episode_frames)} frames...")
        except Exception as e:
            print(f"Error loading frame {i}: {e}")
            continue
    
    print(f"Successfully loaded {len(episode_frames)} frames for episode {target_episode}")
    return episode_frames


def test_single_episode_accuracy(policy, dataset, episode_ranges, target_episode=0, max_frames=50):
    """Test accuracy of different frames in a single episode"""
    
    print(f"\n=== Testing Single Episode Accuracy (Episode {target_episode}) ===")
    
    # Get all frames for specified episode
    episode_frames = get_episode_frames_from_indices(dataset, episode_ranges, target_episode)
    
    if len(episode_frames) == 0:
        print(f"❌ No frames found for episode {target_episode}")
        return
    
    # Limit number of frames
    num_frames = min(len(episode_frames), max_frames)
    episode_frames = episode_frames[:num_frames]
    
    print(f"Testing {num_frames} frames from episode {target_episode}")
    
    # Set random seed for reproducible comparisons
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Store results for comparison
    pytorch_results = []
    tensorrt_results = []
    
    # Test PyTorch inference on episode frames
    print("Running PyTorch inference on episode frames...")
    for i, frame_data in enumerate(episode_frames):
        action = policy.get_action(frame_data)
        pytorch_results.append(action)
        print(f"PyTorch frame {i+1}/{num_frames}: action.single_arm shape = {action['action.single_arm'].shape}")
    
    # Setup TensorRT engines
    print("\nSetting up TensorRT engines...")
    setup_tensorrt_engines(policy, "gr00t_engine")
    
    # Test TensorRT inference on same episode frames
    print("Running TensorRT inference on same episode frames...")
    for i, frame_data in enumerate(episode_frames):
        action = policy.get_action(frame_data)
        tensorrt_results.append(action)
        print(f"TensorRT frame {i+1}/{num_frames}: action.single_arm shape = {action['action.single_arm'].shape}")
    
    # Compare results
    print(f"\n=== Episode {target_episode} Frame Comparison ===")
    
    # Calculate sequence differences (consistent with original test_simulation_accuracy.py)
    pytorch_sequence = pytorch_results[0]['action.single_arm']  # Use first frame as reference
    tensorrt_sequence = tensorrt_results[0]['action.single_arm']
    
    # Calculate sequence differences
    sequence_diff = np.abs(pytorch_sequence - tensorrt_sequence)
    mean_sequence_diff = np.mean(sequence_diff)
    max_sequence_diff = np.max(sequence_diff)
    
    # Calculate sequence similarity (using correlation coefficient)
    pytorch_flat = pytorch_sequence.flatten()
    tensorrt_flat = tensorrt_sequence.flatten()
    sequence_similarity = np.corrcoef(pytorch_flat, tensorrt_flat)[0, 1]
    
    print(f"Sequence Comparison:")
    print(f"  Mean sequence difference: {mean_sequence_diff:.6f}")
    print(f"  Max sequence difference: {max_sequence_diff:.6f}")
    print(f"  Sequence similarity: {sequence_similarity:.6f}")
    
    # Calculate detailed metrics for each frame
    print(f"\n=== Per-Frame Analysis ===")
    
    total_cosine_sim = 0.0
    total_l1_mean = 0.0
    total_l1_max = 0.0
    total_l2_dist = 0.0
    total_relative_error = 0.0
    total_rmse = 0.0
    frame_count = 0
    
    for frame_idx in range(num_frames):
        pytorch_action = pytorch_results[frame_idx]
        tensorrt_action = tensorrt_results[frame_idx]
        
        print(f"\nFrame {frame_idx + 1}/{num_frames}:")
        
        frame_cosine_sim = 0.0
        frame_l1_mean = 0.0
        frame_l1_max = 0.0
        frame_l2_dist = 0.0
        frame_relative_error = 0.0
        frame_rmse = 0.0
        key_count = 0
        
        for key in pytorch_action.keys():
            pytorch_tensor = torch.from_numpy(pytorch_action[key]).to(torch.float32)
            tensorrt_tensor = torch.from_numpy(tensorrt_action[key]).to(torch.float32)
            
            # Calculate metrics
            flat_pytorch = pytorch_tensor.flatten()
            flat_tensorrt = tensorrt_tensor.flatten()
            
            # Cosine similarity (direction similarity)
            dot_product = torch.dot(flat_pytorch, flat_tensorrt)
            norm_pytorch = torch.norm(flat_pytorch)
            norm_tensorrt = torch.norm(flat_tensorrt)
            cos_sim = dot_product / (norm_pytorch * norm_tensorrt)
            
            # L1 distance (absolute error)
            l1_dist = torch.abs(flat_pytorch - flat_tensorrt)
            l1_mean = l1_dist.mean().item()
            l1_max = l1_dist.max().item()
            
            # L2 distance (Euclidean distance)
            l2_dist = torch.norm(flat_pytorch - flat_tensorrt).item()
            
            # Relative error
            relative_error = l1_dist / (torch.abs(flat_pytorch) + 1e-8)
            mean_relative_error = relative_error.mean().item()
            
            # Root Mean Square Error (RMSE)
            mse = torch.mean((flat_pytorch - flat_tensorrt) ** 2)
            rmse = torch.sqrt(mse).item()
            
            print(f"  {key}:")
            print(f"    Cosine Similarity: {cos_sim.item():.6f}")
            print(f"    L1 Mean/Max: {l1_mean:.4f}/{l1_max:.4f}")
            print(f"    L2 Distance: {l2_dist:.4f}")
            print(f"    Relative Error: {mean_relative_error:.6f}")
            print(f"    RMSE: {rmse:.4f}")
            
            # Accumulate for frame average
            frame_cosine_sim += cos_sim.item()
            frame_l1_mean += l1_mean
            frame_l1_max += l1_max
            frame_l2_dist += l2_dist
            frame_relative_error += mean_relative_error
            frame_rmse += rmse
            key_count += 1
            
        
        # Frame averages
        if key_count > 0:
            frame_cosine_sim /= key_count
            frame_l1_mean /= key_count
            frame_l1_max /= key_count
            frame_l2_dist /= key_count
            frame_relative_error /= key_count
            frame_rmse /= key_count
            
            print(f"  Frame Average:")
            print(f"    Cosine Similarity: {frame_cosine_sim:.6f}")
            print(f"    L1 Mean/Max: {frame_l1_mean:.4f}/{frame_l1_max:.4f}")
            print(f"    L2 Distance: {frame_l2_dist:.4f}")
            print(f"    Relative Error: {frame_relative_error:.6f}")
            print(f"    RMSE: {frame_rmse:.4f}")
            
            # Accumulate for overall average
            total_cosine_sim += frame_cosine_sim
            total_l1_mean += frame_l1_mean
            total_l1_max += frame_l1_max
            total_l2_dist += frame_l2_dist
            total_relative_error += frame_relative_error
            total_rmse += frame_rmse
            frame_count += 1
    
    # Overall statistics
    if frame_count > 0:
        avg_cosine_sim = total_cosine_sim / frame_count
        avg_l1_mean = total_l1_mean / frame_count
        avg_l1_max = total_l1_max / frame_count
        avg_l2_dist = total_l2_dist / frame_count
        avg_relative_error = total_relative_error / frame_count
        avg_rmse = total_rmse / frame_count
        
        print(f"\n{'='*60}")
        print(f"OVERALL EPISODE {target_episode} STATISTICS")
        print(f"{'='*60}")
        print(f"Total frames tested: {frame_count}")
        print(f"Average Cosine Similarity: {avg_cosine_sim:.6f}")
        print(f"Average L1 Mean: {avg_l1_mean:.4f}")
        print(f"Average L1 Max: {avg_l1_max:.4f}")
        print(f"Average L2 Distance: {avg_l2_dist:.4f}")
        print(f"Average Relative Error: {avg_relative_error:.6f}")
        print(f"Average RMSE: {avg_rmse:.4f}")
        


def main():
    parser = argparse.ArgumentParser(description="Test TensorRT accuracy on single episode frames using meta info")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="checkpoints/mixed_sim_real/checkpoint-30000",
        help="Path to the model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/sim/sim_enhance_camera_70",
        help="Path to the dataset"
    )
    parser.add_argument(
        "--target_episode",
        type=int,
        default=0,
        help="Target episode to test"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=50,
        help="Maximum number of frames to test per episode"
    )
    
    args = parser.parse_args()
    
    # Load episode information
    print("Loading episode information...")
    episodes = load_episode_info(args.dataset_path)
    episode_ranges = calculate_episode_frame_indices(episodes)
    
    print(f"Found {len(episodes)} episodes:")
    for i, episode in enumerate(episodes[:5]):  # Show first 5
        print(f"  Episode {episode['episode_index']}: {episode['length']} frames")
    if len(episodes) > 5:
        print(f"  ... and {len(episodes) - 5} more episodes")
    
    # Load policy and dataset
    print("\nLoading policy and dataset...")
    data_config = DATA_CONFIG_MAP["so100_dualcam"]
    data_config.video_keys = ["video.room", "video.wrist"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag="new_embodiment",
        modality_config=modality_config,
        modality_transform=modality_transform,
        device="cuda",
    )
    
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend="torchvision_av",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag="new_embodiment",
    )
    
    # Run test
    test_single_episode_accuracy(policy, dataset, episode_ranges, args.target_episode, args.max_frames)


if __name__ == "__main__":
    main()
