import os
import json
import csv

# Read environment variables
input_file = os.getenv("SPEC_CHUNKS_TRIM")
output_file = os.getenv("CHUNKS_FOR_NEO4J")

if not input_file or not output_file:
    raise ValueError("Please set INPUT_FILE and OUTPUT_FILE environment variables.")

# Load the JSON data
with open(input_file, "r") as f:
    data = json.load(f)

# Write to CSV
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["chunkid","section_number", "section_title"])  # header row

    for item in data:
        chunk_id_str = item.get("chunkid")
        section_number_str = item.get("section_number")
        section_title_str = item.get("section_title")
        if chunk_id_str is not None:
            try:
                chunk_id_int = int(chunk_id_str)  # convert "001" → 1
                writer.writerow([chunk_id_int, section_number_str, section_title_str])
            except ValueError:
                writer.writerow([chunk_id_str])
