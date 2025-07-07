import os
import json
import pickle
import argparse
import pandas as pd

def generate_jsonl_from_pkl(pkl_path, png_dir, output_jsonl_path):
    # Load the .pkl file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    pano_entry = data.pop("Pano", None)  # Remove 'Pano' if present

    with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
        for key, content in data.items():
            png_path = os.path.join(png_dir, f"{key}.png")
            has_image = os.path.exists(png_path)

            if not has_image:
                continue  # Skip entries with no corresponding image

            case_id_list = content.get("Case ID", [])
            text_list = content.get("Text", [])

            for i, text in enumerate(text_list):
                if pd.isna(text):
                    continue  # Skip NaN

                entry = {
                    "image": f"{key}.png",
                    "text": str(text),
                    "case_id": case_id_list[i] if i < len(case_id_list) else None
                }
                out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ JSONL written to: {output_jsonl_path}")

# Example usage:
"""
python generate_jsonl.py --pkl sample.pkl --png_dir ./images --output mydata.jsonl

python generate_jsonl.py --pkl /homes/gws/patxiao/Dental/Junwei_preprocessed/content.pkl --png_dir /projects/chimeranb/PanosDeidentified --output mydata.jsonl

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, required=True, help="Path to .pkl file")
    parser.add_argument("--png_dir", type=str, required=True, help="Path to folder containing .png images")
    parser.add_argument("--output", type=str, default="output.jsonl", help="Output .jsonl file path")
    args = parser.parse_args()

    generate_jsonl_from_pkl(args.pkl, args.png_dir, args.output)








