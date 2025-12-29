import os
import json
import sys

def check_size(input_file_path):
    """
    Reads a JSON file and identifies entries where the 'section_title'
    ends with the word "element" (case-insensitive), then saves them to a new JSON file.

    Args:
        input_file_path (str): The path to the input JSON file.
    """
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

    # Iterate through each entry in the list
    for entry in data:
        # Get the section_title, handling potential missing keys or None values
        size = str(entry.get("section_title"))
        print(size) 

if __name__ == "__main__":
    input_json_file_path = os.environ.get("SPEC_CHUNKS")
    check_size("80211-2020-chunks-trim.json")

