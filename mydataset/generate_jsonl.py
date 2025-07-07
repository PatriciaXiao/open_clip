import os
import json
import pickle
import argparse
import pandas as pd

import os
import json
import pickle
import argparse
import pandas as pd
import math

def process_reports(entry_content):
    """
    Given a dictionary for a patient record (with keys like 'Pano', 'xxx', etc),
    extract all non-empty texts under the sub-keys and format them.
    """
    processed_texts = []

    for subkey, value in entry_content.items():
        if subkey == "Pano":
            continue  # Skip pano key

        if isinstance(value, dict) and "Text" in value:
            text_list = value.get("Text", [])
            cleaned_texts = [
                str(txt).strip() for txt in text_list
                if isinstance(txt, str) and txt.strip()
            ]
            if cleaned_texts:
                formatted_text = f"{subkey} Text: " + ";\n\n".join(cleaned_texts)
                processed_texts.append(formatted_text)

    return "\n\n".join(processed_texts) if processed_texts else None

def generate_jsonl_from_pkl(pkl_path, png_dir, output_jsonl_path):
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)

    with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
        for patient_key, entry_content in all_data.items():
            if not isinstance(entry_content, dict):
                continue

            image_path = os.path.join(png_dir, f"{patient_key}.png")
            if not os.path.exists(image_path):
                continue  # Skip if no image

            text = process_reports(entry_content)
            if not text:
                continue  # Skip if no valid text

            out_file.write(json.dumps({
                "image": f"{patient_key}.png",
                "text": text
            }, ensure_ascii=False) + "\n")

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








