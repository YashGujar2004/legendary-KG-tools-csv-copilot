
import os
import json
import csv
from collections import defaultdict

import json
import csv
import sys

def find_frames_in_content(input_json_file, frames_json_file, output_csv_file):
    """
    Reads a JSON file of frame entries, searches for their titles in the content
    of a second JSON file, and writes the results to a CSV with frequency counts.

    Args:
        input_json_file (str): The path to the main JSON file with content chunks.
        frame_json_file (str): The path to the JSON file with frame entries.
        output_csv_file (str): The path to the output CSV file.
    """
    try:
        # Load main JSON data (with content)
        with open(input_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load frame entries from the second JSON file
        with open(frames_json_file, "r", encoding="utf-8") as f:
            frames_data = json.load(f)

    except FileNotFoundError as e:
        print(f"Error: A file was not found - {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in one of the files - {e}")
        return

    # Extract section titles from the frames JSON file
    frame_titles = []
    for entry in frames_data:
        section_title = entry.get("section_title")
        if isinstance(section_title, str):
            frame_titles.append(section_title.strip())

    if not frame_titles:
        print("No valid frame titles found in the frame entries JSON file.")
        return


    frame_dict = defaultdict(int)  # auto-initialize counts to 0

    # Open CSV for writing
    with open(output_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunkid", "section_number", "name", "frequency_count"])  # header

        # Search each frame title in the content of the main JSON file
        for frame_name in frame_titles:
            search_term = frame_name.lower()
            
            for chunk in data:
                content = chunk.get("content", "").lower()
                freq = content.count(search_term)  # count occurrences

                frame_dict[search_term] += freq

                if freq > 0:
                    writer.writerow([
                        chunk.get("chunkid", ""),
                        chunk.get("section_number", ""),
                        frame_name,
                        freq
                    ])

        f.flush()
        os.fsync(f.fileno())


    print(f"✅ Results written to {output_csv_file}")

    with open(frames_json_file, "r+", encoding="utf-8") as f:
       data = json.load(f)
       f.seek(0)
       f.truncate()

# Convert all section_titles to lowercase for the frequency count
       low_freq_titles = {key.lower() for key, freq in frame_dict.items() if freq < 5}
       
       filtered_data = [entry for entry in data if entry.get("section_title", "").lower() not in low_freq_titles]

       print(low_freq_titles)
# Save back to JSON
       json.dump(filtered_data, f, indent=2, ensure_ascii=False)
       f.flush()
       os.fsync(f.fileno())
       frame_dict.clear()

       print(f"✅ Filtered JSON written to {frames_json_file}")


input_chunk_file = os.environ.get("SPEC_CHUNKS_TRIM")
input_frame_list = os.environ.get("FRAMES_LIST")
chunk_link_frame = os.environ.get("CHUNK_LINK_FRAME")

# In first iteration, we remove the frames that does not have links with chunks. reference count < 2 
print("Ist Iteration...")
find_frames_in_content(input_chunk_file, input_frame_list, chunk_link_frame)

# In second iteration, we reform the chunks-to-frame link using the new cleaned frame list
print("IInd Iteration...")
find_frames_in_content(input_chunk_file, input_frame_list, chunk_link_frame)


