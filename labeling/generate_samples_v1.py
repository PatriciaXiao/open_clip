import json
import csv

selected_ids_path = 'selected_case_ids.json'
# scp patxiao@kraken.cs.washington.edu:~/open_clip/mydataset/mydata.jsonl mydata.jsonl
jsonl_path = 'mydata.jsonl'
output_csv_path = 'selected_100_pano_v1.csv'

# Load selected case IDs
with open(selected_ids_path, 'r') as f:
    selected_ids = set(json.load(f))

# Prepare output rows
rows = []

# Read JSONL file line by line
with open(jsonl_path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            image_name = data.get("image", "")
            case_id = image_name.replace(".png", "")

            if case_id in selected_ids:
                text = data.get("text", "").replace('\n', ' ').replace('\r', ' ')
                rows.append((case_id, text))
        except json.JSONDecodeError:
            continue  # Skip malformed lines

# Write to CSV
with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['case_id', 'text'])  # Header
    writer.writerows(rows)

print(f"Saved {len(rows)} records to {output_csv_path}")
