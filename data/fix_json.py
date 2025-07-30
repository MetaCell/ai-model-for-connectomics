#!/usr/bin/env python3
"""
Convert JSONL (JSON Lines) file to proper JSON array format
"""
import json


def convert_jsonl_to_json(input_file, output_file):
    """Convert JSONL file to JSON array format"""
    data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line[:100]}...")
                    print(f"Error: {e}")
                    continue

    # Write as proper JSON array
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Converted {len(data)} records from {input_file} to {output_file}")


if __name__ == "__main__":
    convert_jsonl_to_json("skeleton_test_t1_p1.json", "skeleton_test_t1_p1_fixed.json")
