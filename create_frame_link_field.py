
import os
import json
import csv

import json
import csv
import sys

def find_fields_in_content(input_json_file, fields_json_file, output_csv_file):
    """
    Reads a JSON file of field entries, searches for their titles in the content
    of a second JSON file, and writes the results to a CSV with frequency counts.

    Args:
        input_json_file (str): The path to the main JSON file with content chunks.
        fields_json_file (str): The path to the JSON file with field entries.
        output_csv_file (str): The path to the output CSV file.
    """
    try:
        # Load main JSON data (with content)
        with open(input_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load field entries from the second JSON file
        with open(fields_json_file, "r", encoding="utf-8") as f:
            fields_data = json.load(f)

    except FileNotFoundError as e:
        print(f"Error: A file was not found - {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in one of the files - {e}")
        return

    # Extract section titles from the fields JSON file
    field_titles = []
    for entry in fields_data:
        section_title = entry.get("section_title")
        if isinstance(section_title, str):
            field_titles.append(section_title.strip())

    if not field_titles:
        print("No valid field titles found in the field entries JSON file.")
        return

    # Open CSV for writing
    with open(output_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunkid", "frame name", "section_number", "field name", "frequency_count"])  # header

        # Search each field title in the content of the main JSON file
        for field_name in field_titles:
            search_term = field_name.lower()
            
            for chunk in data:
                content = chunk.get("content", "").lower()
                freq = content.count(search_term)  # count occurrences
                
                if freq > 0:
                    writer.writerow([
                        chunk.get("chunkid", ""),
                        chunk.get("section_title", ""),
                        chunk.get("section_number", ""),
                        field_name,
                        freq
                    ])

    print(f"✅ Results written to {output_csv_file}")

    
input_chunk_file = os.environ.get("FRAMES_LIST")
input_field_list = os.environ.get("FIELDS_LIST")
chunk_link_field = os.environ.get("FRAME_LINK_FIELD")

find_fields_in_content(input_chunk_file, input_field_list, chunk_link_field)

# find_frames_in_json_with_frequency(in"80211-2020-sections-chunks-trim.json", "frames_list.txt", "80211-2020-frames-link-chunks.csv")

