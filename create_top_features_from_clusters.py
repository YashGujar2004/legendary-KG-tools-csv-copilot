
import os
import json
import csv
import re


CLUSTER_INPUT_FILE = os.environ.get("KG_CLUSTERS")
CHUNK_INPUT_FILE = os.environ.get("SPEC_CHUNKS_TRIM")

OUTPUT_FILE = os.environ.get("KG_TOP_FEATURES")

def generate_cluster_features():
    """
    Processes cluster data and chunk data to create a CSV mapping 
    chunks to the clusters they belong to.
    """
    print(f"Starting feature generation...")

    # 1. Load Cluster Data and Map Section Numbers to Cluster IDs
    target_sections = {}
    
    try:
        with open(CLUSTER_INPUT_FILE, "r", encoding="utf-8") as f:
            clusters_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Cluster data file '{CLUSTER_INPUT_FILE}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from '{CLUSTER_INPUT_FILE}'.")
        return

    # Create a fast lookup map: { "section_number": cluster_id }
    for cluster in clusters_data:
        cluster_id = cluster["cluster_id"]
        for section_entry in cluster["sections"]:
          if isinstance(section_entry, (list, tuple)) and len(section_entry) > 0:
            section_number = section_entry[0]
            # Map the section number to its cluster ID
            target_sections[section_number] = cluster_id
            
    print(f"Identified {len(target_sections)} unique sections belonging to clusters.")


    # 2. Load Chunk Metadata
    try:
        with open(CHUNK_INPUT_FILE, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Chunk data file '{CHUNK_INPUT_FILE}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from '{CHUNK_INPUT_FILE}'.")
        return
        
    
    # 3. Process and Write to CSV
    matched_count = 0
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        # Header row (Note: Added cluster_id as the final column)
        writer.writerow(["chunkid", "section_number", "section_title", "cluster_id"])
        
        for chunk in chunks_data:
            section_number = chunk.get("section_number")
            
            # Check if this section number is part of any cluster
            if section_number in target_sections:
                cluster_id = target_sections[section_number]
                chunkid = chunk.get("chunkid")
                section_title = chunk.get("section_title")
                
                # Write the row to the CSV
                writer.writerow([chunkid, section_number, section_title, cluster_id])
                matched_count += 1

    print(f"\nSuccessfully generated feature file: '{OUTPUT_FILE}'")
    print(f"Total matching sections written: {matched_count}")

generate_cluster_features()
