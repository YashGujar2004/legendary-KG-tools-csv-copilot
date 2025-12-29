
from collections import Counter
import os
import json
import sys

# Original list
exception_frame_names = [
    ("sector sweep feedback (ssw-feedback) frame", "SSW-Feedback frame"),
    ("sector sweep ack (ssw-ack) frame", "SSW-Ack frame"),
    ("sector sweep (ssw) frame", "SSW frame"),
    ("bdt variant of the ps-poll frame", "PS-Poll+BDT frame"),
    ("non-bdt variant of the ps-poll frame", "PS-Poll frame"),
    ("service period request (spr) frame", "SPR frame"),
    ("basic addts request frame", "ADDTS Request frame"),
    ("basic addts response frame", "ADDTS Response frame"),
]

# Create dictionary (case-insensitive keys)

def find_entries_ending_with_frame(input_file_path, output_file_path):
    """
    Reads a JSON file and identifies entries where the 'section_title'
    ends with the word "frame" (case-insensitive), then saves them to a new JSON file.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_file_path (str): The path to the output JSON file.
    """
    section_counts = {} 
    exception_dict = {orig.lower(): new for orig, new in exception_frame_names}
    print(input_file_path)
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The input file '{input_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The input file '{input_file_path}' is not a valid JSON file.")
        return

    if not isinstance(data, list):
        print("Error: The JSON file does not contain a list of entries.")
        return

    matching_entries = []
    
    # Iterate through each entry in the list
    for entry in data:
        # Get the section_title, handling potential missing keys or None values
        section_title = str(entry.get("section_title"))

        if isinstance(section_title, str):
            st_lower = section_title.strip()

        if (
            st_lower.endswith(" frame format")
            or st_lower.endswith(" frame")
            or st_lower.endswith(" action frame details")
            or st_lower.endswith(" subframe format")
            or st_lower.endswith(" frame variant")
            or st_lower.endswith(" frame format")
        ):
        # Remove unwanted suffixes
            cleaned_title = (
                st_lower.removesuffix("format")
                .removesuffix("details")
                .removesuffix("variant")
                .strip()
            )
        # Update the entry with cleaned title
            cleaned_title = exception_dict.get(cleaned_title.lower(), cleaned_title)

            entry["section_title"] = cleaned_title


            if section_title:  # Skip None or empty
              if section_title in section_counts:
                 section_counts[section_title] += 1
              else:
                section_counts[section_title] = 1

            matching_entries.append(entry)

    duplicates = {title: count for title, count in section_counts.items() if count > 1}

    if duplicates:
      print("Duplicate section titles with counts:", duplicates)

    # Save the results to the new JSON file
    if matching_entries:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(matching_entries, f, indent=4)
            print(f"Successfully saved {len(matching_entries)} entries to '{output_file_path}'.")
        except IOError:
            print(f"Error: Could not write to the output file '{output_file_path}'.")
    else:
        print("No entries found with 'section_title' ending in 'frame'. No output file was created.")

if __name__ == "__main__":
    input_json_file_path = os.environ.get("SPEC_CHUNKS_TRIM")
    output_json_file_path = os.environ.get("FRAMES_LIST")
    find_entries_ending_with_frame(input_json_file_path, output_json_file_path)

