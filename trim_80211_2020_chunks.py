
import os
import json

def remove_section_chunks(input_file, output_file):
    # Load JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Section prefixes to exclude
    exclude_prefixes = tuple(
        os.environ.get("EXCLUDE_PREFIXES", "").split(",")
    )

    # Keep only chunks where section_number does NOT start with excluded prefixes
    filtered_data = [
        chunk for chunk in data
        if not any(str(chunk.get("section_number", "")).startswith(prefix) for prefix in exclude_prefixes)
    ]

    # Save filtered JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"Filtered JSON written to {output_file}, kept {len(filtered_data)} chunks.")

 # input_chunk_file = sys.argv[1]

input_chunk_file = os.environ.get("SPEC_CHUNKS")

trim_chunk_file = os.environ.get("SPEC_CHUNKS_TRIM")

remove_section_chunks(input_chunk_file, trim_chunk_file)


