
import os
import pandas as pd
import csv

chunks_to_frames_file = os.getenv("CHUNK_LINK_FIELD")
features_file = os.getenv("KG_TOP_FEATURES")
output_file = os.getenv("KG_FEATURE_FIELD_SCORE")


features_df = pd.read_csv(features_file)
frames_df = pd.read_csv(chunks_to_frames_file)

# Ensure section_number is string
features_df["section_number"] = features_df["section_number"].astype(str)
frames_df["section_number"] = frames_df["section_number"].astype(str)

# Output storage
rows = []

# Step 1: For each feature, calculate local scores
for _, feature in features_df.iterrows():
    feature_sec = feature["section_number"]

    # Pick all frame entries whose section_number matches or starts with this feature
#    matched_frames = frames_df[frames_df["section_number"].str.startswith(feature_sec)]

    matched_frames = frames_df[
        (frames_df["section_number"] == feature_sec) |
        (frames_df["section_number"].str.startswith(feature_sec + "."))
    ]

    if matched_frames.empty:
        continue

    total_feature_freq = matched_frames["frequency_count"].sum()

    for frame_name, group in matched_frames.groupby("name"):
        frame_freq = group["frequency_count"].sum()

        # Local score for this feature-frame pair
        local_score = frame_freq / total_feature_freq if total_feature_freq > 0 else 0

        rows.append({
            "feature_section_number": feature_sec,
            "frame_name": frame_name,
            "frame_frequency": frame_freq,
            "local_score": local_score
        })

# Convert rows to DataFrame
output_df = pd.DataFrame(rows)

# Step 2: Compute global scores
total_chunks = frames_df["chunkid"].nunique()

# For each frame, count unique chunks it appears in
field_chunk_counts = frames_df.groupby("name")["chunkid"].nunique()

# Compute global scores
global_scores = (total_chunks / field_chunk_counts).to_dict()

# Add global score to output_df
output_df["global_score"] = output_df["frame_name"].map(global_scores)

# Step 3: Compute td-idf score (local_score * global_score)
output_df["tf_idf"] = output_df["local_score"] * output_df["global_score"]

# Step 4: Sort by feature_section_number for easier analysis
output_df = output_df.sort_values(by="feature_section_number")

# Save output
output_df.to_csv(output_file, index=False)

print(f"Output written to {output_file}")

