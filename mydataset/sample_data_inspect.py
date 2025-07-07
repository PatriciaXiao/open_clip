import webdataset as wds
import io

def inspect_webdataset_tar(tar_path):
    print(f"📦 Inspecting WebDataset TAR: {tar_path}")
    
    dataset = wds.WebDataset(tar_path).decode().to_tuple("png", "txt")

    for idx, (image, caption) in enumerate(dataset):
        print(f"\n----- Sample {idx} -----")

        # Process text (caption)
        if isinstance(caption, bytes):
            try:
                text = caption.decode("utf-8")
            except UnicodeDecodeError:
                print("  ⚠️ Could not decode caption text.")
                continue
        else:
            text = str(caption)

        lines = text.splitlines()

        if not lines:
            print("  [empty line]")
        else:
            for i, line in enumerate(lines):
                if line.strip() == "":
                    print(f"  Line {i+1}: [empty line]")
                else:
                    print(f"  Line {i+1}: {line}")

# Example usage
inspect_webdataset_tar("/homes/gws/patxiao/open_clip/mydataset/sample_data/my_sample.tar")
