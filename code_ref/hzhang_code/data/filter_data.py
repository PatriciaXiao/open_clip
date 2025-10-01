import os
import json

# Input/output paths
input_dir = "/homes/gws/patxiao/open_clip/code_ref/hzhang_code/data"
output_dir = "/homes/gws/patxiao/files/hzhang_data_filtered"
old_prefix = "/home/hzhanguw/research-projects/data/tensors/"
new_prefix = "/projects/chimera/nobackup/patxiao/tensors/"

# Ensure output folder exists
os.makedirs(output_dir, exist_ok=True)

# Process both train and val files
for fname in ["t1_train_data.json", "t1_val_data.json"]:
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)

    with open(input_path, "r") as f:
        data = json.load(f)

    new_data = {}
    for k, v in data.items():
        old_file = v.get("file", "")
        if old_file.startswith(old_prefix):
            new_file = old_file.replace(old_prefix, new_prefix, 1)
        else:
            new_file = old_file  # if it doesn't match prefix, keep as is

        # Keep entry only if file exists
        if os.path.exists(new_file):
            v["file"] = new_file
            new_data[k] = v

    # Save filtered JSON
    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=4)

    print(f"{fname}: kept {len(new_data)} / {len(data)} entries → {output_path}")
