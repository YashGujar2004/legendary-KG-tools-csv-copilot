import json
import os
import csv

input_file = os.getenv("FIELDS_LIST")
output_file = os.getenv("FIELDS_LIST_NAMES")

# Read JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Open CSV for writing
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(["name", "chunkid", "section_number"])
    
    # Extract section_title from each entry
    for entry in data:
        section_title = entry.get("section_title", "").strip()
        chunkid = entry.get("chunkid", "").strip()
        section_number = entry.get("section_number", "").strip()
        if section_title:  # only write if not empty
            writer.writerow([section_title, chunkid, section_number])
print(f"✅ Extracted section titles written to {output_file}")

