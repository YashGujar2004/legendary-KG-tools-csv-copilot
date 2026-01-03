import json
import os
import re
from collections import OrderedDict

INPUT = os.environ.get("VAR_SECTION_LIST")
OUTPUT = os.environ.get("VAR_ALL_SPEC")

# INPUT = "merged_output_V2.json"

def normalize_caption(caption: str) -> str:
    """
    Normalize captions:
    - If it starts with 'Figure ' or 'Table ' and ends with '(continued)', strip the continuation marker.
    - Otherwise return as-is.
    """
    # if caption.strip().startswith(("Figure ", "Table ")):
    #     return re.sub(r"\s*\(continued\)\s*$", "", caption.strip())
    # return caption.strip()
    
                                                    # only strip (continued) for captions starting with Table or Figure
    
    # caption = caption.strip()

    # # Fix missing space after "Table" or "Figure" (e.g., "Table9-" -> "Table 9-")
    # caption = re.sub(r"^(Table|Figure)(\d+)", r"\1 \2", caption)

    # # Remove duplicate prefix like "Table 9-297." or "Figure 9-297."
    # caption = re.sub(r"^(Table|Figure)\s*\d+\s*[-–]\d+\.\s*", "", caption)

    # # Remove (continued) at the end
    # caption = re.sub(r"\s*\(continued\)\s*$", "", caption)

    # return caption.strip()
    
                                                      # STRIP continue and the random text
                                                      
def normalize_caption(caption: str) -> str:
    caption = caption.strip()

    # Fix missing space after "Table" or "Figure" (e.g., "Table9-" → "Table 9-")
    caption = re.sub(r"^(Table|Figure)(\d+)", r"\1 \2", caption)

    # Find the last valid "Table N—..." or "Figure N—..." caption
    match = re.findall(r"(?:Table|Figure)\s*\d+\s*[-–]\s*\d+—.*", caption)
    if match:
        caption = match[-1]  # keep the last one

    # Remove (continued) at the end
    caption = re.sub(r"\s*\(continued\)\s*$", "", caption)

    return caption.strip()


with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

groups = {}

for entry in data:
    caption = entry["caption"].strip()
    base_caption = normalize_caption(caption)

    # Extract table_data for this entry
    table_data = entry.get("table_data", [])

    if base_caption not in groups:
        # Initialize group with first occurrence
        groups[base_caption] = {
            "page_number": entry.get("page_number"),
            "caption": base_caption,
            "section_title": entry.get("section_title"),
            "section_number": entry.get("section_number"),
            "chunkid": entry.get("chunkid"),
            "figure_cells": [],    # for flattened Figure merge
            "table_rows": [],      # for Table merge
            "is_table": base_caption.startswith("Table "),
            "merged_pages": [],
            "merged_chunkIDs": []
        }

        if groups[base_caption]["is_table"]:
            # For base tables, keep header + data
            groups[base_caption]["table_rows"].extend(table_data)
        else:
            # For figures, flatten rows
            flat = [cell for row in table_data for cell in row]
            groups[base_caption]["figure_cells"].extend(flat)

    else:
        # Handle continuations
        if groups[base_caption]["is_table"]:
            # For tables, skip header row in continuations
            if len(table_data) > 1:
                groups[base_caption]["table_rows"].extend(table_data[1:])
        else:
            # For figures, flatten rows
            flat = [cell for row in table_data for cell in row]
            groups[base_caption]["figure_cells"].extend(flat)

        # Track continuation info
        if caption != base_caption:
            groups[base_caption]["merged_pages"].append(entry.get("page_number"))
            groups[base_caption]["merged_chunkIDs"].append(entry.get("chunkid"))

# Build final output
out = []
for g in groups.values():
    ordered = OrderedDict()
    ordered["page_number"] = g["page_number"]
    ordered["caption"] = g["caption"]
    ordered["section_title"] = g["section_title"]
    ordered["section_number"] = g["section_number"]
    ordered["chunkid"] = g["chunkid"]

    if g["is_table"]:
        ordered["table_data"] = g["table_rows"]
    else:
        ordered["table_data"] = [g["figure_cells"]]

    if g["merged_pages"]:
        ordered["merged_pages"] = g["merged_pages"]
    if g["merged_chunkIDs"]:
        ordered["merged_chunkIDs"] = g["merged_chunkIDs"]

    out.append(ordered)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=4, ensure_ascii=False)

print(f"Merged JSON saved to {OUTPUT}")
