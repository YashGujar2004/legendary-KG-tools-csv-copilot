
import os
import csv
import json
from collections import defaultdict
# Define skip list


input_file = os.environ.get("CHUNK_LINK_FRAME")
output_file = os.environ.get("FEATURE_SUBFEATURE_GROUP")
chunks_file = os.environ.get("SPEC_CHUNKS_TRIM")

def load_chunks_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    # return a dictionary keyed by chunkid
    return {str(entry["chunkid"]): entry for entry in data}

def get_prefix(section_number):
    """Return the section number prefix by dropping the last part"""
    parts = section_number.split(".")
    if len(parts) > 1:
        return ".".join(parts[:-1])   # remove last digit
    else:
        return section_number  # if only one part, return as is

groups = defaultdict(list)

skip_prefix_feature = os.environ.get("SKIP_PREFIXES_FOR_FEATURES", " ")
skip_section_feature = os.environ.get("SKIP_SECTIONS_FOR_FEATURES", " ")

# Convert to lists (comma-separated in env)
skip_prefix_feature = [s.strip() for s in skip_prefix_feature.split(",") if s.strip()]
skip_section_feature = [s.strip() for s in skip_section_feature.split(",") if s.strip()]

def skip_section(sec_num: str) -> bool:
    """Return True if section should be skipped."""
    if sec_num in skip_section_feature:
        return True
    return any(sec_num.startswith(prefix) for prefix in skip_prefix_feature)

# Read input CSV
with open(input_file, "r", newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        sec_num = row["section_number"].strip()
        if sec_num and not skip_section(sec_num):  # skip empty or unwanted
            prefix = get_prefix(sec_num)
            groups[prefix].append(sec_num)

# Write output CSV
with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    
    chunks_lookup = load_chunks_json(chunks_file)

    # Write header
    writer.writerow(["Group ID", "Parent Section", "Parent Title", "Section Numbers"])
    

    for idx, (prefix, sec_nums) in enumerate(groups.items(), start=1):
      if len(sec_nums) == 1:
        parent = sec_nums[0]   # parent = the section itself
      else:
        parent = prefix        # parent = common prefix

      parent_title = "N/A"
      for chunk_id, chunk_meta in chunks_lookup.items():
        if chunk_meta.get("section_number") == parent:
            parent_title = chunk_meta.get("section_title", "N/A")
            break

    # Parent section is its own column, not merged into sec_nums
     # row = [f"Group{idx}", parent, parent_title] + list(sec_nums)
      row = [f"Group{idx}", parent, parent_title, ",".join(sec_nums)]

      writer.writerow(row)

print(f"Groups written to {output_file}")

