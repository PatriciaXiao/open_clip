import torch

ckpt = torch.load("/projects/chimera/nobackup/wliu25/ckpts/pretrain/ckpt_lshort_bs128/eval/training_99999/teacher_checkpoint.pth", map_location="cpu")
state_dict = ckpt.get("model", ckpt)

print("Checkpoint keys:")
for k in sorted(state_dict.keys()):
    print(" ", k)