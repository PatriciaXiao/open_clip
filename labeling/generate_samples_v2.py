import json
import csv
import random

selected_ids_path = 'selected_case_ids.json'
jsonl_path = 'mydata.jsonl'
output_prefix = 'selected_200_pano_v1_part'

# Load selected case IDs
with open(selected_ids_path, 'r') as f:
    selected_ids = set(json.load(f))

# Store unique (case_id, text) pairs
records = {}

# Read JSONL file
with open(jsonl_path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            image_name = data.get("image", "")
            case_id = image_name.replace(".png", "")
            if case_id in selected_ids and case_id not in records:
                text = data.get("text", "").replace('\n', ' ').replace('\r', ' ')
                records[case_id] = text
        except json.JSONDecodeError:
            continue

# Ensure we only take up to 600 unique records
case_items = list(records.items())
if len(case_items) < 600:
    print(f"Warning: only {len(case_items)} unique cases found.")
sampled = random.sample(case_items, min(600, len(case_items)))


# Shuffle and split into 3 equal parts
chunk_len = 150
random.shuffle(sampled)
chunks = [sampled[i:i + chunk_len] for i in range(0, len(sampled), chunk_len)]

# Write each chunk to separate CSV file
for i, chunk in enumerate(chunks):
    output_csv_path = f"{output_prefix}{i+1}.csv"
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'text'])
        writer.writerows(chunk)
    print(f"Saved {len(chunk)} records to {output_csv_path}")
