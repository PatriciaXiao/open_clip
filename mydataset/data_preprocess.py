import os
import tarfile
import json
import io

def create_webdataset_tar(image_dir, metadata_path, output_path):
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
    with tarfile.open(output_path, "w") as tar:
        for idx, line in enumerate(lines):
            item = json.loads(line)
            image_path = os.path.join(image_dir, item["image"])
            caption = item["text"]
            basename = f"{idx:06d}"

            # Add image
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
            img_info = tarfile.TarInfo(name=f"{basename}.png")
            img_info.size = len(img_data)
            tar.addfile(tarinfo=img_info, fileobj=io.BytesIO(img_data))

            # Add caption
            txt_data = caption.encode("utf-8")
            txt_info = tarfile.TarInfo(name=f"{basename}.txt")
            txt_info.size = len(txt_data)
            tar.addfile(tarinfo=txt_info, fileobj=io.BytesIO(txt_data))



#create_webdataset_tar("./sample_data/images", "./sample_data/metadata.jsonl", "./sample_data/my_sample.tar")



create_webdataset_tar("/projects/chimeranb/PanosDeidentified", "./mydata.jsonl", "/projects/chimeranb/patxiao/mydata.tar")