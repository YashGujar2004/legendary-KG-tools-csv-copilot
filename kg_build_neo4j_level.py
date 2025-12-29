import os
import json
import pandas as pd

def generate_csvs_from_edges(input_json, output_file, output_dir="."):
    with open(input_json, "r") as f:
        data = json.load(f)

    edges = data.get("edges", [])

    # Prepare CSV buckets
    csv_data = {i: [] for i in range(1, 6)}

    for edge in edges:
        feature = edge["to"]
        frame = edge["from"]
        score = edge["score"]

        parts = feature.split(".")

        # up to 5 levels (v, v.w, v.w.x, v.w.x.y, v.w.x.y.z)
        for level in range(1, min(len(parts), 5) + 1):
            pattern = ".".join(parts[:level])
            parent = ".".join(parts[:level-1]) if level > 1 else None

            row = {
                "feature": pattern,
                "score": score,
                "frame": frame
            }
            if level > 1:
                row["parent"] = parent

            csv_data[level].append(row)

    # Save CSVs
    for level, rows in csv_data.items():
        df = pd.DataFrame(rows)
        if level == 1:
            df = df[["feature", "score", "frame"]]
        else:
            df = df[["feature", "score", "frame", "parent"]]

        df.to_csv(f"{output_dir}/{output_file}_{level}.csv", index=False)
        print(f"Saved {output_dir}/{output_file}_{level}.csv")

if __name__ == "__main__":
    # Get JSON file from environment variable
    input_file = os.getenv("KG_TRANSFORM_FEATURE_FOR_NEO4J")
    output_file = os.getenv("KG_BUILD_NEO4J_LEVEL")
    if not input_file:
        raise ValueError("Environment variable JSON_FILE not set!")

    generate_csvs_from_edges(input_file, output_file)
