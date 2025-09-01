# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import torch
import time
import numpy as np
from action_head_utils import action_head_pytorch_forward
from trt_model_forward import setup_tensorrt_engines
from PIL import Image

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

# Global variables to collect timing data
timing_data = {
    'VLM - ViT': [],
    'VLM - LLM': [],
    'Action_Head - process_backbone_output': [],
    'Action_Head - state_encoder': [],
    'Action_Head - action_encoder': [],
    'Action_Head - DiT': [],
    'Action_Head - action_decoder': []
}

# PyTorch timing data - only real measurable components
pytorch_timing_data = {
    'VLM - ViT': [],  # This will be the actual backbone time
    'VLM - LLM': [],  # This will be the actual backbone time (same as ViT since they're integrated)
    'Backbone Total': [],  # Total backbone execution time
    'Action_Head - process_backbone_output': [],
    'Action_Head - state_encoder': [],
    'Action_Head - action_encoder': [],
    'Action_Head - DiT': [],
    'Action_Head - action_decoder': []
}

# Custom print function to capture timing data
original_print = print
def timing_print(*args, **kwargs):
    """Custom print function that captures timing data"""
    # Call original print
    original_print(*args, **kwargs)
    
    # Check if this is a timing line
    if args and isinstance(args[0], str) and 'ms FP16' in args[0]:
        collect_timing_data(args[0])

def collect_timing_data(timing_str):
    """Extract timing data from the printed timing strings"""
    try:
        # Parse timing string like "VLM - ViT: 11.96 ms FP16"
        parts = timing_str.split(':')
        if len(parts) == 2:
            component = parts[0].strip()
            time_str = parts[1].split()[0]  # Extract "11.96"
            time_ms = float(time_str)
            
            if component in timing_data:
                timing_data[component].append(time_ms)
    except Exception as e:
        pass  # Ignore parsing errors

def collect_pytorch_timing_data(component, time_ms):
    """Collect timing data for PyTorch components"""
    if component in pytorch_timing_data:
        pytorch_timing_data[component].append(time_ms)

def print_timing_statistics():
    """Print average timing statistics for each component"""
    print("\n=== Detailed Component Timing Statistics (100 runs average) ===")
    print(f"{'Component':<40} {'Avg (ms)':<10} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10}")
    print("-" * 80)
    
    for component, times in timing_data.items():
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            print(f"{component:<40} {avg_time:<10.2f} {std_time:<10.2f} {min_time:<10.2f} {max_time:<10.2f}")
        else:
            print(f"{component:<40} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

def print_pytorch_timing_statistics():
    """Print average timing statistics for PyTorch components"""
    print("\n=== PyTorch Component Timing Statistics 100 runs average) ===")
    print(f"{'Component':<40} {'Avg (ms)':<10} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10}")
    print("-" * 80)
    
    for component, times in pytorch_timing_data.items():
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            print(f"{component:<40} {avg_time:<10.2f} {std_time:<10.2f} {min_time:<10.2f} {max_time:<10.2f}")
        else:
            print(f"{component:<40} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")


def compare_predictions(pred_tensorrt, pred_torch):
    """
    Compare the similarity between TensorRT and PyTorch predictions

    Args:
        pred_tensorrt: TensorRT prediction results (numpy array)
        pred_torch: PyTorch prediction results (numpy array)
    """
    print("\n=== Prediction Comparison ===")

    # Ensure both predictions contain the same keys
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"

    # Calculate max label width for alignment
    max_label_width = max(
        len("Cosine Similarity (PyTorch/TensorRT):"),
        len("L1 Mean/Max Distance (PyTorch/TensorRT):"),
        len("Max Output Values (PyTorch/TensorRT):"),
        len("Mean Output Values (PyTorch/TensorRT):"),
        len("Min Output Values (PyTorch/TensorRT):"),
    )

    for key in pred_tensorrt.keys():
        tensorrt_array = pred_tensorrt[key]
        torch_array = pred_torch[key]

        # Convert to PyTorch tensors
        tensorrt_tensor = torch.from_numpy(tensorrt_array).to(torch.float32)
        torch_tensor = torch.from_numpy(torch_array).to(torch.float32)

        # Ensure tensor shapes are the same
        assert (
            tensorrt_tensor.shape == torch_tensor.shape
        ), f"{key} shapes do not match: {tensorrt_tensor.shape} vs {torch_tensor.shape}"

        # Calculate cosine similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        # Manually calculate cosine similarity
        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = dot_product / (norm_tensorrt * norm_torch)

        # Calculate L1 distance
        l1_dist = torch.abs(flat_tensorrt - flat_torch)

        print(f"\n{key}:")
        print(f'{"Cosine Similarity (PyTorch/TensorRT):".ljust(max_label_width)} {cos_sim.item()}')
        print(
            f'{"L1 Mean/Max Distance (PyTorch/TensorRT):".ljust(max_label_width)} {l1_dist.mean().item():.4f}/{l1_dist.max().item():.4f}'
        )
        print(
            f'{"Max Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.max().item():.4f}/{tensorrt_tensor.max().item():.4f}'
        )
        print(
            f'{"Mean Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.mean().item():.4f}/{tensorrt_tensor.mean().item():.4f}'
        )
        print(
            f'{"Min Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.min().item():.4f}/{tensorrt_tensor.min().item():.4f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GR00T inference")
    parser.add_argument(
        "--model_path", type=str, default="nvidia/GR00T-N1.5-3B", help="Path to the GR00T model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default="",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["pytorch", "tensorrt", "compare"],
        default="pytorch",
        help="Inference mode: 'pytorch' for PyTorch inference, 'tensorrt' for TensorRT inference, "
             "'compare' for comparing PyTorch and TensorRT outputs similarity",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps",
        default=4,
    )
    parser.add_argument(
        "--trt_engine_path",
        type=str,
        help="Path to the TensorRT engine",
        default="gr00t_engine",
    )
    parser.add_argument(
        "--num_warmup_runs",
        type=int,
        help="Number of warmup runs for TensorRT inference",
        default=3,
    )
    parser.add_argument(
        "--num_profile_runs",
        type=int,
        help="Number of profiling runs for TensorRT inference",
        default=100,
    )

    args = parser.parse_args()

    MODEL_PATH = args.model_path
    EMBODIMENT_TAG = "new_embodiment"
    dataset_path = args.dataset_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = DATA_CONFIG_MAP["so100_dualcam"]
    data_config.video_keys = ["video.room", "video.wrist"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device=device,
    )

    modality_config = policy.modality_config

    if dataset_path == "":
        room_img_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        room_img = Image.fromarray(room_img_array, "RGB")
        wrist_img_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        wrist_img = Image.fromarray(wrist_img_array, "RGB")
        arm = np.random.randn(5)
        gripper = np.random.randn(1)

        step_data = {
            "video.room": np.expand_dims(room_img, axis=0),
            "video.wrist": np.expand_dims(wrist_img, axis=0),
            "state.single_arm": np.expand_dims(np.array(arm), axis=0),
            "state.gripper": np.expand_dims(np.array(gripper), axis=0),
            "annotation.human.task_description": "Grip the scissors and put it into the tray",
        }
    else:
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            video_backend="torchvision_av",
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=EMBODIMENT_TAG,
        )

        step_data = dataset[0]


    if args.inference_mode == "pytorch":
        # Clear previous PyTorch timing data
        for component in pytorch_timing_data:
            pytorch_timing_data[component].clear()

        # Warmup runs
        print(f"Performing {args.num_warmup_runs} warmup runs...")
        for i in range(args.num_warmup_runs):
            _ = policy.get_action(step_data)

        # Timed inference runs
        num_runs = args.num_profile_runs
        inference_times = []

        print(f"\nRunning {num_runs} timed inference runs...")
        print("Collecting detailed PyTorch component timing data...")
        
        # Create hooks to measure individual component times
        component_times = {}
        component_start_times = {}
        
        def create_pre_forward_hook(component_name):
            def hook(module, input):
                component_start_times[component_name] = time.perf_counter()
            return hook
        
        def create_forward_hook(component_name):
            def hook(module, input, output):
                if component_name in component_start_times:
                    execution_time = (time.perf_counter() - component_start_times[component_name]) * 1000
                    if component_name not in component_times:
                        component_times[component_name] = []
                    component_times[component_name].append(execution_time)
            return hook
        
        # Register hooks for timing measurement
        hooks = []
        
        # Hook for backbone (EAGLE model)
        backbone_pre_hook = create_pre_forward_hook('backbone')
        backbone_hook = create_forward_hook('backbone')
        hooks.append(policy.model.backbone.register_forward_pre_hook(backbone_pre_hook))
        hooks.append(policy.model.backbone.register_forward_hook(backbone_hook))
        
        # Hook for EAGLE model internal components
        if hasattr(policy.model.backbone, 'eagle_model'):
            # Hook for vision model (ViT)
            if hasattr(policy.model.backbone.eagle_model, 'vision_model'):
                vit_pre_hook = create_pre_forward_hook('vision_model')
                vit_hook = create_forward_hook('vision_model')
                hooks.append(policy.model.backbone.eagle_model.vision_model.register_forward_pre_hook(vit_pre_hook))
                hooks.append(policy.model.backbone.eagle_model.vision_model.register_forward_hook(vit_hook))
            
            # Hook for language model (LLM)
            if hasattr(policy.model.backbone.eagle_model, 'language_model'):
                llm_pre_hook = create_pre_forward_hook('language_model')
                llm_hook = create_forward_hook('language_model')
                hooks.append(policy.model.backbone.eagle_model.language_model.register_forward_pre_hook(llm_pre_hook))
                hooks.append(policy.model.backbone.eagle_model.language_model.register_forward_hook(llm_hook))
        
        # Hook for action head components
        if hasattr(policy.model.action_head, 'state_encoder'):
            state_pre_hook = create_pre_forward_hook('state_encoder')
            state_hook = create_forward_hook('state_encoder')
            hooks.append(policy.model.action_head.state_encoder.register_forward_pre_hook(state_pre_hook))
            hooks.append(policy.model.action_head.state_encoder.register_forward_hook(state_hook))
        
        # Hook for process_backbone_output (vlln + vl_self_attention)
        if hasattr(policy.model.action_head, 'vlln'):
            print("Found vlln component, registering hooks...")
            vlln_pre_hook = create_pre_forward_hook('vlln')
            vlln_hook = create_forward_hook('vlln')
            hooks.append(policy.model.action_head.vlln.register_forward_pre_hook(vlln_pre_hook))
            hooks.append(policy.model.action_head.vlln.register_forward_hook(vlln_hook))
        else:
            print("vlln component not found!")
        
        if hasattr(policy.model.action_head, 'vl_self_attention'):
            print("Found vl_self_attention component, registering hooks...")
            vl_attn_pre_hook = create_pre_forward_hook('vl_self_attention')
            vl_attn_hook = create_forward_hook('vl_self_attention')
            hooks.append(policy.model.action_head.vl_self_attention.register_forward_pre_hook(vl_attn_pre_hook))
            hooks.append(policy.model.action_head.vl_self_attention.register_forward_hook(vl_attn_hook))
        else:
            print("vl_self_attention component not found!")
        
        # Debug: show what components are available
        print(f"Available action_head components: {[name for name, _ in policy.model.action_head.named_children()]}")
        
        if hasattr(policy.model.action_head, 'action_encoder'):
            action_enc_pre_hook = create_pre_forward_hook('action_encoder')
            action_enc_hook = create_forward_hook('action_encoder')
            hooks.append(policy.model.action_head.action_encoder.register_forward_pre_hook(action_enc_pre_hook))
            hooks.append(policy.model.action_head.action_encoder.register_forward_hook(action_enc_hook))
        
        if hasattr(policy.model.action_head, 'model'):  # DiT model
            dit_pre_hook = create_pre_forward_hook('DiT')
            dit_hook = create_forward_hook('DiT')
            hooks.append(policy.model.action_head.model.register_forward_pre_hook(dit_pre_hook))
            hooks.append(policy.model.action_head.model.register_forward_hook(dit_hook))
        
        if hasattr(policy.model.action_head, 'action_decoder'):
            decoder_pre_hook = create_pre_forward_hook('action_decoder')
            decoder_hook = create_forward_hook('action_decoder')
            hooks.append(policy.model.action_head.action_decoder.register_forward_pre_hook(decoder_pre_hook))
            hooks.append(policy.model.action_head.action_decoder.register_forward_hook(decoder_hook))
        
        for i in range(num_runs):
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            start_time = time.perf_counter()
            
            # Clear component times for this run
            component_times.clear()
            component_start_times.clear() # Clear start times for this run
            
            # Run full inference
            predicted_action = policy.get_action(step_data)
            
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            end_time = time.perf_counter()

            # Calculate component times from hooks - NO ESTIMATION, only real measurements
            if 'backbone' in component_times:
                backbone_time = component_times['backbone'][0]  # Already in ms
                collect_pytorch_timing_data('Backbone Total', backbone_time)
            
            # Calculate process_backbone_output time using the accurate formula
            process_time = 0
            if 'backbone' in component_times and 'vision_model' in component_times and 'language_model' in component_times:
                # Ideal Process time calculation
                process_time = (
                    component_times['backbone'][0] - 
                    (component_times['vision_model'][0] + component_times['language_model'][0]) +  # 多模态融合+投影层+数据处理
                    (component_times.get('vlln', [0])[0] + component_times.get('vl_self_attention', [0])[0])  # vlln + vl_self_attention
                )
                collect_pytorch_timing_data('Action_Head - process_backbone_output', process_time)
                print(f"DEBUG: Accurate process_backbone_output time: {process_time:.2f} ms")
                print(f"DEBUG: Breakdown - backbone: {component_times['backbone'][0]:.2f}, vit: {component_times['vision_model'][0]:.2f}, llm: {component_times['language_model'][0]:.2f}")
            else:
                print("DEBUG: Cannot calculate accurate process time - missing components!")
            
            # Debug: show all captured components
            print(f"DEBUG: Captured components: {list(component_times.keys())}")
            
            # Calculate ViT time (vision model)
            if 'vision_model' in component_times:
                vit_time = component_times['vision_model'][0]  # Already in ms
                collect_pytorch_timing_data('VLM - ViT', vit_time)
            
            # Calculate LLM time (language model)
            if 'language_model' in component_times:
                llm_time = component_times['language_model'][0]  # Already in ms
                collect_pytorch_timing_data('VLM - LLM', llm_time)
            
            if 'state_encoder' in component_times:
                state_time = component_times['state_encoder'][0]  # Already in ms
                collect_pytorch_timing_data('Action_Head - state_encoder', state_time)
            
            if 'action_encoder' in component_times:
                action_enc_time = component_times['action_encoder'][0]  # Already in ms
                collect_pytorch_timing_data('Action_Head - action_encoder', action_enc_time)
            
            if 'DiT' in component_times:
                dit_time = component_times['DiT'][0]  # Already in ms
                collect_pytorch_timing_data('Action_Head - DiT', dit_time)
            
            if 'action_decoder' in component_times:
                decoder_time = component_times['action_decoder'][0]  # Already in ms
                collect_pytorch_timing_data('Action_Head - action_decoder', decoder_time)

            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time)
            
            if (i + 1) % 10 == 0:  # Print progress every 10 runs
                print(f"Completed {i+1}/{num_runs} runs...")

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Calculate statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)

        print("\n=== PyTorch Overall Inference Performance ===")
        print(f"Number of runs: {num_runs}")
        print(f"Mean inference time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"Min inference time: {min_time:.2f} ms")
        print(f"Max inference time: {max_time:.2f} ms")
        print(f"Throughput: {1000/mean_time:.2f} inferences/second")

        # Print detailed PyTorch component timing statistics
        print_pytorch_timing_statistics()

        print("\n=== PyTorch Inference Results ===")
        for key, value in predicted_action.items():
            print(key, value.shape)

    elif args.inference_mode == "tensorrt":
        # Setup TensorRT engines
        setup_tensorrt_engines(policy, args.trt_engine_path)

        # Clear previous timing data
        for component in timing_data:
            timing_data[component].clear()

        # Warmup runs (TensorRT engines may need warmup for optimal performance)
        print(f"Performing {args.num_warmup_runs} warmup runs...")
        for i in range(args.num_warmup_runs):
            _ = policy.get_action(step_data)

        # Timed inference runs
        num_runs = args.num_profile_runs
        inference_times = []

        print(f"\nRunning {num_runs} timed inference runs...")
        print("Collecting detailed component timing data...")
        
        # Temporarily replace print with timing_print to capture timing data
        import builtins
        builtins.print = timing_print
        
        for i in range(num_runs):
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            start_time = time.perf_counter()

            predicted_action = policy.get_action(step_data)
            
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            end_time = time.perf_counter()

            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time)
            
            if (i + 1) % 10 == 0:  # Print progress every 10 runs
                print(f"Completed {i+1}/{num_runs} runs...")

        # Restore original print function
        builtins.print = original_print

        # Calculate overall inference statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)

        print("\n=== TensorRT Overall Inference Performance ===")
        print(f"Number of runs: {num_runs}")
        print(f"Mean inference time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"Min inference time: {min_time:.2f} ms")
        print(f"Max inference time: {max_time:.2f} ms")
        print(f"Throughput: {1000/mean_time:.2f} inferences/second")

        # Print detailed component timing statistics
        print_timing_statistics()

        print("\n=== TensorRT Inference Results ===")
        for key, value in predicted_action.items():
            print(key, value.shape)


    else:
        # ensure PyTorch and TensorRT have the same init_actions
        if not hasattr(policy.model.action_head, "init_actions"):
            policy.model.action_head.init_actions = torch.randn(
                (1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
                dtype=torch.float16,
                device=device,
            )
        
        # PyTorch inference
        policy.model.action_head.get_action = partial(
            action_head_pytorch_forward, policy.model.action_head
        )
        predicted_action_torch = policy.get_action(step_data)

        # Setup TensorRT engines and run inference
        setup_tensorrt_engines(policy, args.trt_engine_path)
        predicted_action_tensorrt = policy.get_action(step_data)

        # Compare predictions
        compare_predictions(predicted_action_tensorrt, predicted_action_torch)
