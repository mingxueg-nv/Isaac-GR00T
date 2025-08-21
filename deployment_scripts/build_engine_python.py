#!/usr/bin/env python3

import os
import tensorrt as trt
import numpy as np

def build_engine_from_onnx(onnx_path, engine_path, min_shape, opt_shape, max_shape):
    """Build TensorRT engine from ONNX model using Python API"""
    
    logger = trt.Logger(trt.Logger.INFO)  # More verbose logging
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    config.set_flag(trt.BuilderFlag.FP16) 
    
    # Parse ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Set optimization profile
    profile = builder.create_optimization_profile()
    
    for input_name in min_shape.keys():
        profile.set_shape(input_name, min_shape[input_name], opt_shape[input_name], max_shape[input_name])
    
    config.add_optimization_profile(profile)
    
    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print(f"Failed to build engine for {onnx_path}")
        return False
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"Successfully built engine: {engine_path}")
    return True

def main():
    # Define length variables
    MIN_LEN = 100
    OPT_LEN = 600
    MAX_LEN = 1200
    
    os.makedirs("gr00t_engine", exist_ok=True)
    
    # VLLN-VLSelfAttention (use smaller sequence lengths due to memory constraints)
    print("Building vlln_vl_self_attention Model...")
    build_engine_from_onnx(
        "gr00t_onnx/action_head/vlln_vl_self_attention.onnx",
        "gr00t_engine/vlln_vl_self_attention.engine",
        {"backbone_features": (1, MIN_LEN, 2048)},
        {"backbone_features": (2, 600, 2048)},  # Use 600 as optimal length to match actual sequence length
        {"backbone_features": (8, 1200, 2048)}   # Use 1200 as max length for safety
    )
    
    # DiT Model
    print("Building DiT Model...")
    build_engine_from_onnx(
        "gr00t_onnx/action_head/DiT.onnx",
        "gr00t_engine/DiT.engine",
        {"sa_embs": (1, 17, 1536), "vl_embs": (1, MIN_LEN, 2048), "timesteps_tensor": (1,)},
        {"sa_embs": (2, 17, 1536), "vl_embs": (2, OPT_LEN, 2048), "timesteps_tensor": (2,)},
        {"sa_embs": (8, 17, 1536), "vl_embs": (8, MAX_LEN, 2048), "timesteps_tensor": (8,)}
    )
    
    # State Encoder
    print("Building State Encoder...")
    build_engine_from_onnx(
        "gr00t_onnx/action_head/state_encoder.onnx",
        "gr00t_engine/state_encoder.engine",
        {"state": (1, 1, 64), "embodiment_id": (1,)},
        {"state": (2, 1, 64), "embodiment_id": (2,)},
        {"state": (8, 1, 64), "embodiment_id": (8,)}
    )
    
    # Action Encoder
    print("Building Action Encoder...")
    build_engine_from_onnx(
        "gr00t_onnx/action_head/action_encoder.onnx",
        "gr00t_engine/action_encoder.engine",
        {"actions": (1, 16, 32), "timesteps_tensor": (1,), "embodiment_id": (1,)},
        {"actions": (2, 16, 32), "timesteps_tensor": (2,), "embodiment_id": (2,)},
        {"actions": (8, 16, 32), "timesteps_tensor": (8,), "embodiment_id": (8,)}
    )
    
    # Action Decoder
    print("Building Action Decoder...")
    build_engine_from_onnx(
        "gr00t_onnx/action_head/action_decoder.onnx",
        "gr00t_engine/action_decoder.engine",
        {"model_output": (1, 17, 1024), "embodiment_id": (1,)},
        {"model_output": (2, 17, 1024), "embodiment_id": (2,)},
        {"model_output": (8, 17, 1024), "embodiment_id": (8,)}
    )
    
    # VLM-ViT
    print("Building VLM-ViT...")
    build_engine_from_onnx(
        "gr00t_onnx/eagle2/vit.onnx",
        "gr00t_engine/vit.engine",
        {"pixel_values": (1, 3, 224, 224), "position_ids": (1, 256)},
        {"pixel_values": (2, 3, 224, 224), "position_ids": (2, 256)},
        {"pixel_values": (8, 3, 224, 224), "position_ids": (8, 256)}
    )
    
    # VLM-LLM
    print("Building VLM-LLM...")
    build_engine_from_onnx(
        "gr00t_onnx/eagle2/llm.onnx",
        "gr00t_engine/llm.engine",
        {"input_ids": (1, MIN_LEN), "vit_embeds": (1, 512, 1152), "attention_mask": (1, MIN_LEN)},  # vit_embeds uses 512 (2*256) for dual camera
        {"input_ids": (1, OPT_LEN), "vit_embeds": (1, 512, 1152), "attention_mask": (1, OPT_LEN)},  # vit_embeds uses 512 (2*256) for dual camera
        {"input_ids": (8, MAX_LEN), "vit_embeds": (8, 512, 1152), "attention_mask": (8, MAX_LEN)}  # vit_embeds max uses 512 for dual camera
    )
    
    print("All engines built successfully!")

if __name__ == "__main__":
    main()
