import torch

"""

python convert_dino_to_openclip.py \
  --input /projects/chimera/nobackup/wliu25/ckpts/pretrain/ckpt_lshort_bs128/eval/training_99999/teacher_checkpoint.pth \
  --output ./openclip_vit_l14_from_dino.pth

"""

def remap_dino_teacher_to_openclip(dino_ckpt_path, output_ckpt_path):
    dino_ckpt = torch.load(dino_ckpt_path, map_location="cpu")
    dino_state = dino_ckpt.get("teacher", dino_ckpt)

    new_state = {}

    # Top-level mappings
    mapping = {
        "backbone.cls_token": "visual.class_embedding",
        "backbone.pos_embed": "visual.positional_embedding",
        "backbone.patch_embed.proj.weight": "visual.conv1.weight",
        "backbone.patch_embed.proj.bias": "visual.conv1.bias",
        "backbone.norm.weight": "visual.ln_post.weight",
        "backbone.norm.bias": "visual.ln_post.bias",
    }
    for k, v in mapping.items():
        if k in dino_state:
            new_state[v] = dino_state[k]

    # Transformer blocks: flatten stage.block into resblock index
    for key, val in dino_state.items():
        if not key.startswith("backbone.blocks."):
            continue
        parts = key.split(".")
        stage = int(parts[2])
        block = int(parts[3])
        subkey = ".".join(parts[4:])
        block_idx = stage * 12 + block

        if subkey.startswith("attn.qkv."):
            proj = subkey.split(".")[-1]
            new_key = f"visual.transformer.resblocks.{block_idx}.attn.in_proj_{proj}"
        elif subkey.startswith("attn.proj."):
            proj = subkey.split(".")[-1]
            new_key = f"visual.transformer.resblocks.{block_idx}.attn.out_proj.{proj}"
        elif subkey.startswith("mlp.fc1."):
            proj = subkey.split(".")[-1]
            new_key = f"visual.transformer.resblocks.{block_idx}.mlp.c_fc.{proj}"
        elif subkey.startswith("mlp.fc2."):
            proj = subkey.split(".")[-1]
            new_key = f"visual.transformer.resblocks.{block_idx}.mlp.c_proj.{proj}"
        elif subkey.startswith("norm1."):
            proj = subkey.split(".")[-1]
            new_key = f"visual.transformer.resblocks.{block_idx}.ln_1.{proj}"
        elif subkey.startswith("norm2."):
            proj = subkey.split(".")[-1]
            new_key = f"visual.transformer.resblocks.{block_idx}.ln_2.{proj}"
        else:
            continue

        new_state[new_key] = val

    # Save in OpenCLIP format
    torch.save({"state_dict": new_state}, output_ckpt_path)
    print(f"✅ Saved converted model to: {output_ckpt_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to DINOv2 teacher checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save OpenCLIP-compatible checkpoint")
    args = parser.parse_args()

    remap_dino_teacher_to_openclip(args.input, args.output)
