import torch

def identify_vit_arch(checkpoint_path):
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    #state_dict = ckpt.get("model", ckpt)["teacher"]# needed for orig ver
    state_dict = ckpt.get("model", ckpt)

    #print(state_dict.keys())
    print("Checkpoint keys:")
    for k in sorted(state_dict.keys()):
        print(" ", k)

    # Determine hidden dim
    class_embed = state_dict.get("visual.class_embedding", None)
    if class_embed is None:
        print("❌ Could not find 'visual.class_embedding'. Not a ViT-style model.")
        return

    embed_dim = class_embed.shape[-1]
    
    # Determine number of transformer layers
    num_blocks = sum(1 for k in state_dict if k.startswith("visual.transformer.resblocks.") and ".attn" in k) // 3

    # Determine patch size
    proj_w_shape = state_dict.get("visual.patch_embed.proj.weight", None)
    if proj_w_shape is not None:
        _, _, patch_H, patch_W = proj_w_shape.shape
    else:
        patch_H = patch_W = -1

    # Positional embedding length
    pos_embed = state_dict.get("visual.positional_embedding", None)
    if pos_embed is not None:
        pos_len = pos_embed.shape[0]
    else:
        pos_len = -1

    print(f"📐 Detected:")
    print(f"  - Embedding dim: {embed_dim}")
    print(f"  - Transformer layers: {num_blocks}")
    print(f"  - Patch size: {patch_H}x{patch_W}")
    print(f"  - Positional embeddings: {pos_len} tokens")

    # Identify model type
    if embed_dim == 768 and num_blocks == 12:
        if patch_H == 14:
            model_name = "ViT-B-14"
        elif patch_H == 16:
            model_name = "ViT-B-16"
        else:
            model_name = "ViT-B (unknown patch size)"
    elif embed_dim == 1024 and num_blocks == 24 and patch_H == 14:
        model_name = "ViT-L-14"
    elif embed_dim == 1280 and num_blocks == 32 and patch_H == 14:
        model_name = "ViT-H-14"
    else:
        model_name = "Unknown ViT configuration"

    print(f"\n✅ Best matching OpenCLIP model: {model_name}")


# Example usage
#identify_vit_arch("/projects/chimera/nobackup/wliu25/ckpts/pretrain/ckpt_lshort_bs128/eval/training_99999/teacher_checkpoint.pth")
identify_vit_arch("./openclip_vit_l14_from_dino.pth")


# check

import torch
import math

def inspect_vit_config(ckpt_path, image_res=224):
    print(f"📦 Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("teacher", ckpt)

    # Infer patch size from positional embeddings
    pos_embed = state_dict.get("backbone.pos_embed", None)
    if pos_embed is None:
        print("❌ No 'backbone.pos_embed' found in checkpoint.")
        return

    pos_embed = pos_embed.squeeze(0)  # shape: [1, N+1, C] -> [N+1, C]
    num_patches = pos_embed.shape[0] - 1  # subtract CLS token
    patches_per_dim = int(math.sqrt(num_patches))
    patch_size = image_res // patches_per_dim

    # Infer number of layers
    block_keys = [k for k in state_dict.keys() if "backbone.blocks" in k]
    layer_indices = set()
    for key in block_keys:
        parts = key.split(".")
        if "blocks" in parts:
            block_idx = parts[parts.index("blocks") + 1]
            layer_indices.add(int(block_idx))
    num_layers = max(layer_indices) + 1 if layer_indices else None

    # Infer width from first norm layer
    norm_weight = state_dict.get("backbone.blocks.0.norm1.weight", None)
    width = norm_weight.shape[0] if norm_weight is not None else None

    # Infer head width from qkv weight
    qkv_weight = state_dict.get("backbone.blocks.0.attn.qkv.weight", None)
    if qkv_weight is not None and width is not None:
        qkv_dim = qkv_weight.shape[0]
        head_width = qkv_dim // (3 * width)
    else:
        head_width = None

    print("\n🧠 Inferred vision_cfg:")
    print(f"  image_size  : {image_res}")
    print(f"  patch_size  : {patch_size}")
    print(f"  layers      : {num_layers}")
    print(f"  width       : {width}")
    print(f"  head_width  : {head_width}")

# Example usage
inspect_vit_config("/projects/chimera/nobackup/wliu25/ckpts/pretrain/ckpt_lshort_bs128/eval/training_99999/teacher_checkpoint.pth")




    
