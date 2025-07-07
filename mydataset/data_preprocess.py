import os
import tarfile
import json

def create_webdataset_tar(image_dir, metadata_path, output_path):
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
    image_dir = os.path.abspath(image_dir) # use absolute directory
    with tarfile.open(output_path, "w") as tar:
        for idx, line in enumerate(lines):
            item = json.loads(line)
            image_path = os.path.join(image_dir, item["image"])
            caption = item["text"]
            basename = f"{idx:06d}"

            # Add image
            tar.add(image_path, arcname=f"{basename}.png")

            # Add caption
            caption_bytes = caption.encode("utf-8")
            info = tarfile.TarInfo(name=f"{basename}.txt")
            info.size = len(caption_bytes)
            tar.addfile(tarinfo=info, fileobj=io.BytesIO(caption_bytes))


create_webdataset_tar("./sample_data/images", "./sample_data/metadata.jsonl", "./sample_data/my_sample.tar")
