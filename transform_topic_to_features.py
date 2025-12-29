
import json
import os
import re

def transform_id(value: str) -> str:
    """Replace id/edge references starting with SubFeature, Topic, Component → Feature"""
    return re.sub(r'^(SubFeature|Topic|Component)', "Feature", value)

def transform_json(input_file: str, output_file: str):
    with open(input_file, "r") as f:
        data = json.load(f)

    seen_ids = set()
    duplicate_ids = []

    # Transform nodes
    for node in data.get("nodes", []):
        if "id" in node:
            new_id = transform_id(node["id"])
            if new_id in seen_ids:
                duplicate_ids.append(new_id)
            else:
                seen_ids.add(new_id)
            node["id"] = new_id

        if "label" in node and node["label"] in ["SubFeature", "Topic", "Component"]:
            node["label"] = "Feature"

    # Transform edges
    for edge in data.get("edges", []):
        if "from" in edge:
            edge["from"] = transform_id(edge["from"])
        if "to" in edge:
            edge["to"] = transform_id(edge["to"])

    # Write output JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Transformed JSON written to {output_file}")

    # Report duplicates
    if duplicate_ids:
        print("⚠️ Duplicate IDs detected after transformation:")
        for dup in duplicate_ids:
            print(f"   - {dup}")
    else:
        print("✅ No duplicate IDs found after transformation.")


#1. Read CSV with frame frequency data
input_file = os.environ.get("KG_TOPICS_FEATURE_NODES_FRAME_EDGES")

output_file = os.environ.get("KG_TRANSFORM_FEATURE_NODES_FRAME_EDGES")

# Example usage:
transform_json(input_file, output_file)

