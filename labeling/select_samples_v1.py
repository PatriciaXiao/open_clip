import os
import json
import random
from collections import defaultdict, Counter

def load_json_labels(folder):
    """Load all {case_id: label} dicts from JSON files in a folder"""
    all_labels = {}
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            with open(os.path.join(folder, fname), 'r') as f:
                data = json.load(f)
                all_labels[fname] = data
    return all_labels

def invert_label_map(all_labels):
    """Build case_id → list of (filename, label)"""
    case_map = defaultdict(list)
    for fname, case_dict in all_labels.items():
        for case_id, label in case_dict.items():
            case_map[case_id].append((fname, label))
    return case_map

def satisfies_min_positives(selected_ids, all_labels, min_pos_per_file=10):
    """Check that each file has at least `min_pos_per_file` positive cases"""
    for fname, label_map in all_labels.items():
        count = sum(1 for cid in selected_ids if label_map.get(cid) == "True")
        if count < min_pos_per_file:
            return False
    return True

def sample_cases(all_labels, total_cases=100, min_pos_per_file=10, max_trials=10000):
    case_map = invert_label_map(all_labels)

    # Rank cases by number of files they appear in (overlap)
    case_scores = [(cid, len(files)) for cid, files in case_map.items()]
    case_scores.sort(key=lambda x: -x[1])  # descending by overlap

    for _ in range(max_trials):
        # Candidate pool is top cases by overlap
        top_cases = [cid for cid, _ in case_scores[:500]]
        random.shuffle(top_cases)
        candidate_set = top_cases[:total_cases]

        if satisfies_min_positives(candidate_set, all_labels, min_pos_per_file):
            return candidate_set

    raise ValueError("Failed to find a valid sample satisfying all constraints.")

def main():
    input_folder = "dental label"  # <- replace this with your path
    output_path = "selected_case_ids.json"

    all_labels = load_json_labels(input_folder)
    selected_ids = sample_cases(all_labels, total_cases=100, min_pos_per_file=10)

    print(f"Selected {len(selected_ids)} cases satisfying all constraints.")
    with open(output_path, 'w') as f:
        json.dump(selected_ids, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

