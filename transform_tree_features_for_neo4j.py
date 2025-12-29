import os
import json

# Get filenames from environment variables
INPUT_JSON = os.environ.get("KG_TRANSFORM_FEATURE_NODES_FRAME_EDGES")
OUTPUT_JSON = os.environ.get("KG_TRANSFORM_FEATURE_FOR_NEO4J")

# Load your file
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# Function to strip Frame_/Feature_ prefix
def strip_prefix(value):
    if isinstance(value, str) and value.startswith("Frame_"):
        return value.replace("Frame_", "", 1)
    if isinstance(value, str) and value.startswith("Feature_"):
        return value.replace("Feature_", "", 1)
    return value

# Update nodes
for node in data.get("nodes", []):
    if "id" in node:
        node["id"] = strip_prefix(node["id"])

# Update edges
for edge in data.get("edges", []):
    if "from" in edge:
        edge["from"] = strip_prefix(edge["from"])
    if "to" in edge:
        edge["to"] = strip_prefix(edge["to"])

# Remove "id" from nodes
for node in data.get("nodes", []):
    node.pop("id", None)  # safely remove if it exists

# Save cleaned file
with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=4)

print(f"✅ Cleaned JSON saved to {OUTPUT_JSON}")
