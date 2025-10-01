import os
import json

# Paths
json_dir = "/homes/gws/patxiao/files/hzhang_data_filtered"
tensor_dir = "/projects/chimera/nobackup/patxiao/tensors"

# Step 1: Collect all referenced files from both JSONs
referenced_files = set()

for fname in ["t1_train_data.json", "t1_val_data.json"]:
    fpath = os.path.join(json_dir, fname)
    with open(fpath, "r") as f:
        data = json.load(f)
    for v in data.values():
        file_path = v.get("file")
        if file_path and file_path.startswith(tensor_dir):
            referenced_files.add(os.path.basename(file_path))  # keep just filename

print(f"Found {len(referenced_files)} referenced tensor files.")

# Step 2: Delete unreferenced files in tensor_dir
deleted_count = 0
for fname in os.listdir(tensor_dir):
    if fname not in referenced_files:
        fpath = os.path.join(tensor_dir, fname)
        os.remove(fpath)
        deleted_count += 1
        print(f"Deleted unreferenced: {fpath}")

print(f"Finished cleanup. Deleted {deleted_count} files.")
