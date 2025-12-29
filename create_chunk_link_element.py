
import os
import json
import csv
from collections import defaultdict

import json
import csv
import sys

def find_elements_in_content(input_json_file, elements_json_file, output_csv_file):
    """
    Reads a JSON file of element entries, searches for their titles in the content
    of a second JSON file, and writes the results to a CSV with frequency counts.

    Args:
        input_json_file (str): The path to the main JSON file with content chunks.
        elements_json_file (str): The path to the JSON file with element entries.
        output_csv_file (str): The path to the output CSV file.
    """
    try:
        # Load main JSON data (with content)
        with open(input_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load element entries from the second JSON file
        with open(elements_json_file, "r", encoding="utf-8") as f:
            elements_data = json.load(f)

    except FileNotFoundError as e:
        print(f"Error: A file was not found - {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in one of the files - {e}")
        return

    # Extract section titles from the elements JSON file
    element_titles = []
    for entry in elements_data:
        section_title = entry.get("section_title")
        if isinstance(section_title, str):
            element_titles.append(section_title.strip())

    if not element_titles:
        print("No valid element titles found in the element entries JSON file.")
        return

    element_dict = defaultdict(int)  # auto-initialize counts to 0

    # Open CSV for writing
    with open(output_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunkid", "section_number", "name", "frequency_count"])  # header

        # Search each element title in the content of the main JSON file
        for element_name in element_titles:
            search_term = element_name.lower()
            
            for chunk in data:
                content = chunk.get("content", "").lower()
                freq = content.count(search_term)  # count occurrences              

                element_dict[search_term] += freq

                if freq > 0:
                    writer.writerow([
                        chunk.get("chunkid", ""),
                        chunk.get("section_number", ""),
                        element_name,
                        freq
                    ])
        f.flush()
        os.fsync(f.fileno())

    print(f"✅ Results written to {output_csv_file}")

    with open(elements_json_file, "r+", encoding="utf-8") as f:
       data = json.load(f)
       f.seek(0)
       f.truncate()

# Convert all section_titles to lowercase for the frequency count
       low_freq_titles = {key.lower() for key, freq in element_dict.items() if freq < 5}
       
       filtered_data = [entry for entry in data if entry.get("section_title", "").lower() not in low_freq_titles]

       print(low_freq_titles)
# Save back to JSON
       json.dump(filtered_data, f, indent=2, ensure_ascii=False)
       f.flush()
       os.fsync(f.fileno())
       element_dict.clear()

       print(f"✅ Filtered JSON written to {elements_json_file}")

    
input_chunk_file = os.environ.get("SPEC_CHUNKS_TRIM")
input_element_list = os.environ.get("ELEMENTS_LIST")
chunk_link_element = os.environ.get("CHUNK_LINK_ELEMENT")


# In first iteration, we remove the elements that does not have links with chunks. reference count < 2 
print("Ist Iteration...")

find_elements_in_content(input_chunk_file, input_element_list, chunk_link_element)


# In second iteration, we reform the chunks-to-frame link using the new cleaned element list
print("IInd Iteration...")

find_elements_in_content(input_chunk_file, input_element_list, chunk_link_element)


