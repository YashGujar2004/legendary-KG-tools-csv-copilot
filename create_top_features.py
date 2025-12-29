
import os
import json
import csv
import re

output_file = os.environ.get("KG_TOP_FEATURES")
input_file = os.environ.get("SPEC_CHUNKS_TRIM")

# Regex to match section_number of format X.Y (digits dot digits)
pattern = re.compile(r'^\d+\.\d+$')

with open(input_file, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["chunkid", "section_number", "section_title"])  # header row

    for chunk in chunks_data:
        section_number = chunk.get("section_number")
        section_title = chunk.get("section_title")
        chunkid = chunk.get("chunkid")

        if section_number and pattern.match(section_number):
            writer.writerow([chunkid, section_number, section_title])

print(f"CSV file written to {output_file}")

