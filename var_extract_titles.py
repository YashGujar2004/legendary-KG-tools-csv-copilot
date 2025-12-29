                        # chunkID --> chunkid

import json
from difflib import SequenceMatcher
from collections import OrderedDict

def similarity(a, b):
    """Return similarity ratio between two strings (0–1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

INPUT_JSON_FILE1 = os.environ.get("VAR_TABLE_LIST")
INPUT_CHUNKS_FILE2 = os.environ.get("SPEC_CHUNKS_TRIM")
OUTPUT_FILE = os.environ.get("VAR_SECTION_LIST")

with open(INPUT_JSON_FILE1, "r", encoding="utf-8") as f1, open(INPUT_CHUNKS_FILE2, "r", encoding="utf-8") as f2:
    file1 = json.load(f1)
    file2 = json.load(f2)

merged_entries = []

for entry1 in file1:
    page_num = entry1["page_number"]
    caption = entry1.get("caption", "")

    # Find all candidate sections whose range covers this page
    candidates = [
        e for e in file2 if e["page_start"] <= page_num <= e["page_end"]
    ]

    if not candidates:
        section_title = None
        section_number = None
        chunkID = None
    elif len(candidates) == 1:
        best_match = candidates[0]
        section_title = best_match.get("section_title")
        section_number = best_match.get("section_number")
        chunkID = best_match.get("chunkid")
    else:
        # Multiple matches → pick by best caption similarity
        best_match = max(
            candidates,
            key=lambda e: similarity(caption, e.get("section_title") or "")
        )
        section_title = best_match.get("section_title")
        section_number = best_match.get("section_number")
        chunkID = best_match.get("chunkid")

    # Build entry in desired key order
    ordered_entry = OrderedDict()
    ordered_entry["page_number"] = entry1.get("page_number")
    ordered_entry["caption"] = entry1.get("caption")
    ordered_entry["section_title"] = section_title
    ordered_entry["section_number"] = section_number
    ordered_entry["chunkid"] = chunkID
    ordered_entry["table_data"] = entry1.get("table_data")

    merged_entries.append(ordered_entry)

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged_entries, f, indent=4, ensure_ascii=False)


print("Merged JSON")
