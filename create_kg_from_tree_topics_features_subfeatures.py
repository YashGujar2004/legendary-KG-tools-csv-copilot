
#!/usr/bin/env python3
import csv
import os
import json
import re
from pathlib import Path
from typing import Dict, Any

def find_col(fieldnames, *candidates):
    """Return the first matching column name from fieldnames matching any candidate (case-insensitive)."""
    if not fieldnames:
        return None
    low_to_orig = {fn.strip().lower(): fn for fn in fieldnames}
    for cand in candidates:
        c = cand.strip().lower()
        if c in low_to_orig:
            return low_to_orig[c]
    # try startswith or contains fallback
    for cand in candidates:
        lcand = cand.strip().lower()
        for k, orig in low_to_orig.items():
            if k == lcand or k.startswith(lcand) or lcand in k:
                return orig
    return None

def split_list_cell(cell: str):
    """Split a CSV cell containing a comma-separated list into clean parts."""
    if cell is None:
        return []
    # remove wrapping quotes if any (csv reader normally already removed them)
    s = cell.strip().strip('"').strip("'")
    if s == "":
        return []
    # Split on commas but trim whitespace
    parts = [p.strip() for p in re.split(r"\s*,\s*", s) if p.strip()]
    return parts

def load_chunks_map(json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load chunks.json and return a mapping: section_number -> chunk_entry.
    Ignores entries with null/empty section_number.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = {}
    for entry in data:
        sec = entry.get("section_number")
        if sec is None:
            continue
        sec_key = str(sec).strip()
        if sec_key:
            m[sec_key] = entry
    return m

# ---------- Main processing ----------

def map_csv_to_chunks(input_csv: str, chunks_json: str, output_json: str):
    # load chunks lookup
    chunk_map = load_chunks_map(chunks_json)

    output_list = []
    seen = set()  # to dedupe (type, section_number) pairs in order

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # detect column names robustly
        fieldnames = reader.fieldnames or []
        group_col = find_col(fieldnames, "Group ID", "GroupID", "Group")  # optional, not required
        topic_col = find_col(fieldnames, "Topic")
        feature_col = find_col(fieldnames, "Feature")
        subfeature_col = find_col(fieldnames, "Sub-Feature", "SubFeature", "Sub Feature")
        component_col = find_col(fieldnames, "Component")

        if topic_col is None and feature_col is None and subfeature_col is None and component_col is None:
            raise ValueError("Could not find Topic/Feature/Sub-Feature/Component columns in CSV.")

        # mapping of CSV column -> output type string
        col_to_type = []
        if topic_col:
            col_to_type.append((topic_col, "topic"))
        if feature_col:
            col_to_type.append((feature_col, "feature"))
        if subfeature_col:
            col_to_type.append((subfeature_col, "sub-feature"))
        if component_col:
            col_to_type.append((component_col, "component"))

        # iterate rows
        for row in reader:
            # group_id may be useful to debug but not required in final object per your earlier spec
            group_id = row.get(group_col) if group_col else None

            for col_name, type_label in col_to_type:
                raw = row.get(col_name, "")
                parts = split_list_cell(raw)
                for sec in parts:
                    # normalize section string
                    sec_key = sec.strip()
                    if not sec_key:
                        continue

                    key = (type_label, sec_key)
                    if key in seen:
                        continue
                    seen.add(key)

                    chunk_entry = chunk_map.get(sec_key)
                    if chunk_entry:
                        chunk_id = chunk_entry.get("chunkid") or chunk_entry.get("chunkid") or chunk_entry.get("chunkid")
                        section_title = chunk_entry.get("section_title") or chunk_entry.get("sectionTitle") or chunk_entry.get("title")
                    else:
                        chunk_id = None
                        section_title = None

                    out_obj = {
                        "type": type_label,
                        "section_number": sec_key,
                        "section_title": section_title if section_title is not None else None,
                        "chunkid": chunk_id if chunk_id is not None else None
                    }
                    output_list.append(out_obj)

    # write JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote {len(output_list)} entries to {output_json}")

# ---------- Run when invoked ----------

if __name__ == "__main__":
    # change filenames if needed

    input_csv_path = os.environ.get("TREE_TOPICS_FEATURES_SUBFEATURES")
    chunks_json_path = os.environ.get("SPEC_CHUNKS_TRIM")
    output_json_path = os.environ.get("KG_TOPICS_FEATURES_SUBFEATURES")

    # simple sanity checks
    if not Path(input_csv_path).exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")
    if not Path(chunks_json_path).exists():
        raise FileNotFoundError(f"Chunks JSON not found: {chunks_json_path}")

    map_csv_to_chunks(input_csv_path, chunks_json_path, output_json_path)

