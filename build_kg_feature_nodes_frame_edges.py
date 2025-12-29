
import os
import csv
import json
from collections import defaultdict


skip_prefix_feature = os.environ.get("SKIP_PREFIXES_FOR_FEATURES", " ")
skip_section_feature = os.environ.get("SKIP_SECTIONS_FOR_FEATURES", " ")

# Convert to lists (comma-separated in env)

skip_prefix_feature = [s.strip() for s in skip_prefix_feature.split(",") if s.strip()]
skip_section_feature = [s.strip() for s in skip_section_feature.split(",") if s.strip()]

input_csv_file = os.environ.get("CHUNK_LINK_FRAME")
output_file = os.environ.get("KG_TOPICS_FEATURE_NODES_FRAME_EDGES")
chunks_file = os.environ.get("SPEC_CHUNKS_TRIM")


# ---------- FUNCTIONS ----------
def skip_section(sec_num: str) -> bool:
  #"""Return True if section should be skipped."""

  if sec_num in skip_section_feature: 
    return True
  return any(sec_num.startswith(prefix) for prefix in skip_prefix_feature)


def parse_section_hierarchy(section_number):
    """
    Given a section_number string like '11.19.2.1':
    Returns a dictionary mapping hierarchy levels to section ids.
    Level mapping:
        - Topic: first part
        - Feature: first two parts
        - SubFeature: first three parts
        - Component: all remaining parts (4+)
    """
    if not section_number or section_number.upper() == 'N/A':
        return {}

    parts = section_number.split('.')
    hierarchy = {}

    if len(parts) >= 1:
        hierarchy['Topic'] = parts[0]
    if len(parts) >= 2:
        hierarchy['Feature'] = '.'.join(parts[:2])
    if len(parts) >= 3:
        hierarchy['SubFeature'] = '.'.join(parts[:3])
    if len(parts) >= 4:
        hierarchy['Component'] = '.'.join(parts)

    return hierarchy

# ---------- MAIN PROCESS ----------

# 1. Read CSV with frame frequency data
# Dictionaries to store nodes and edges
nodes = {}  # node_id -> {id, label, name}
edges = defaultdict(lambda: 0)  # (frame_node_id, hierarchy_node_id) -> cumulative score

with open(input_csv_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    row_count = 0
    for row in reader:
        row_count += 1

        frame_name = row.get("name", "").strip()
        if not frame_name:
            print(f"Warning: Row {row_count} missing Frame name. Skipping.")
            continue

        # Convert Frequency Count to integer safely
        freq_raw = row.get("frequency_count", "0").strip()
        try:
            freq = int(freq_raw)
        except ValueError:
            print(f"Warning: Row {row_count} has invalid Frequency Count '{freq_raw}'. Skipping row.")
            continue

        section_number = row.get("section_number", "").strip()
        if not section_number or skip_section(section_number) or section_number.upper() == 'N/A':
#            print(f"Info: Row {row_count} frame '{frame_name}' has no section number. Skipping hierarchy scoring.")
             #if skip_section(section_number): 
             #  print(f"Info: skip section {section_number}")

             continue

        # ---------- Create Frame node ----------
        frame_node_id = f"Frame_{frame_name}"
        if frame_node_id not in nodes:
            nodes[frame_node_id] = {"id": frame_node_id, "label": "Frame", "name": frame_name}

        # ---------- Parse section hierarchy ----------
        hierarchy = parse_section_hierarchy(section_number)

        # ---------- For each hierarchy level, create nodes and accumulate link scores ----------
        for level, sec_id in hierarchy.items():
            hierarchy_node_id = f"{level}_{sec_id}"
            if hierarchy_node_id not in nodes:
                nodes[hierarchy_node_id] = {"id": hierarchy_node_id, "label": level, "name": sec_id}

            # Edge from frame to hierarchy node, cumulative frequency
            edges[(frame_node_id, hierarchy_node_id)] += freq

        # ---------- Intermediate milestone prints ----------
        #print(f"Row {row_count}: Frame '{frame_name}' -> section '{section_number}' -> freq={freq}")
        #print(f"  Hierarchy nodes: {hierarchy}")
        #print(f"  Current edges count: {len(edges)}")

print(f"Total rows processed: {row_count}")
print(f"Total nodes created: {len(nodes)}")
print(f"Total edges created: {len(edges)}")

# ---------- Prepare JSON for Neo4j ----------
output_json = {
    "nodes": list(nodes.values()),
    "edges": [
        {"from": f_id, "to": h_id, "score": score} for (f_id, h_id), score in edges.items()
    ]
}

# ---------- Write JSON to file ----------


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=4)

print(f"Output JSON written to '{output_file}'")

