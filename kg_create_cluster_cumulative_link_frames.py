
import json
import pandas as pd
import os
from collections import defaultdict
import math


chunks_to_frames_file = os.getenv("CHUNK_LINK_FRAME")
kg_clusters_file = os.getenv("KG_CLUSTERS")
kg_cluster_primary_frame_file = os.getenv("KG_CLUSTER_PRIMARY_FRAME")


# ============================================================
# LOAD INPUT FILES
# ============================================================

print("\n=== LOADING FILES ===")

df = pd.read_csv(chunks_to_frames_file)

with open(kg_clusters_file, "r") as f:
    clusters = json.load(f)

print("Loaded sections_to_frames rows:", len(df))
print("Loaded clusters:", len(clusters))

# ============================================================
# PREPARE FRAMES_BY_SECTION (EXCLUDE SECTION 9.*)
# ============================================================

frames_by_section = defaultdict(dict)

for _, row in df.iterrows():
    sec = str(row["section_number"]).strip()
    frame = row["name"].strip()
    freq = int(row["frequency_count"])

    if not sec or sec == "nan":
        continue

    # exclude section 9 or 9.x
    if sec.startswith("9"):
        continue

    frames_by_section[sec][frame] = freq

print("\n=== UNIQUE VALID SECTIONS (excluding 9.*) ===")
print(len(frames_by_section))


# ============================================================
# PROCESS CLUSTERS
# ============================================================

frame_to_clusters = defaultdict(set)       # frame → set of cluster_ids
local_scores = defaultdict(dict)           # frame → cluster_id → local_score
frame_freq_cluster = defaultdict(dict)     # frame → cluster_id → frequency_sum

valid_cluster_ids = []  # track only clusters that are allowed

print("\n=== PROCESSING CLUSTERS ===")

for cluster in clusters:

    cluster_id = cluster["cluster_id"]
    cluster_sections = [str(sec[0]) for sec in cluster["sections"]]

    print(f"\nProcessing cluster_id={cluster_id}")
    print("  Cluster top-level sections:", cluster_sections)

    # ==== NEW LOGIC: Skip cluster if ALL sections start with 9 ====
    all_excluded = all(sec.startswith("9") for sec in cluster_sections)
    if all_excluded:
        print(f"  SKIPPING cluster {cluster_id} because all sections begin with 9")
        continue

    valid_cluster_ids.append(cluster_id)

    # Aggregate frequencies for this cluster
    freq_map = defaultdict(int)
    total_freq = 0

    # Find matching descendant sections
    for csec in cluster_sections:

        # skip cluster-level section 9
        if csec.startswith("9"):
            continue

        for sec in frames_by_section:

            # skip any section 9
            if sec.startswith("9"):
                continue

            # prefix match: sec is descendant of csec
            if sec == csec or sec.startswith(csec + "."):

                # accumulate frequencies
                for frame, freq in frames_by_section[sec].items():
                    freq_map[frame] += freq
                    total_freq += freq

    print("  Frames found in this cluster:", dict(freq_map))
    print("  Total frame frequency in cluster:", total_freq)

    if total_freq == 0:
        print("  WARNING: no frames survive filtering for this cluster")
        continue

    # Compute local scores
    for frame, fsum in freq_map.items():
        local = fsum / total_freq
        frame_freq_cluster[frame][cluster_id] = fsum
        local_scores[frame][cluster_id] = local
        frame_to_clusters[frame].add(cluster_id)
        print(f"    Frame={frame} freq={fsum}, local_score={local:.4f}")


# ============================================================
# COMPUTE GLOBAL SCORES
# ============================================================

print("\n=== COMPUTING GLOBAL SCORES ===")

total_valid_clusters = len(valid_cluster_ids)
print("Valid clusters count:", total_valid_clusters)

global_scores = {}

for frame, clusterset in frame_to_clusters.items():
    global_scores[frame] = total_valid_clusters / len(clusterset)
    print(f"Frame={frame}, clusters={list(clusterset)}, global_score={global_scores[frame]:.4f}")


# ============================================================
# BUILD OUTPUT ROWS (frame_cluster_scores.csv)
# ============================================================

print("\n=== BUILDING OUTPUT ROWS ===")

rows = []

for frame in frame_to_clusters:
    gscore = global_scores[frame]

    for cluster_id in frame_to_clusters[frame]:
        lscore = local_scores[frame][cluster_id]
#        tfidf = gscore * lscore
        tfidf = lscore * math.log(1 + gscore)


        rows.append([
            frame,
            gscore,
            cluster_id,
            lscore,
            tfidf
        ])

        print(f"Output row: frame={frame}, cluster={cluster_id}, "
              f"local={lscore:.4f}, global={gscore:.4f}, tfidf={tfidf:.4f}")

frame_cluster_score_df = pd.DataFrame(rows, columns=[
    "frame_name", "global_score", "cluster_id", "local_score", "tf_idf"
])

# ============================================================
# GLOBAL SCORE DISTRIBUTION CSV
# ============================================================

print("\n=== BUILDING GLOBAL SCORE DISTRIBUTION CSV ===")

dist_rows = [[frame, global_scores[frame]] for frame in global_scores]

df = pd.DataFrame(dist_rows, columns=["frame_name", "global_score"])

print(df.head())

# ============================================================
# BUILD THRESHOLD TO IDENTIFY PRIMARY FRAMES MINIMUM GLOBAL SCORE
# ============================================================

common_frames = [
    "probe request frame",
    "probe response frame",
    "beacon frame",
    "cts frame",
    "ack frame",
    "action frame",
    "association request frame",
    "association response frame",
    "reassociation response frame",
    "reassociation request frame",
    "CF-End frame",
    "RTS frame",
    "S1G beacon frame"
]


df["normalized"] = df["frame_name"].str.lower()

scores = []

for fr in common_frames:
    row = df[df["normalized"] == fr.lower()]
    if len(row) > 0:
        score = row["global_score"].iloc[0]
        scores.append(score)
        print(f"{fr:30} -> {score}")
    else:
        print(f"{fr:30} -> NOT FOUND")

PRIMARY_FRAME_MINIMUM_GLOBAL_SCORE = max(scores)


# ============================================================
# BUILD CLUSTER PRIMARY FRAME LIST
# ============================================================

# Normalize frame names for safety
frame_cluster_score_df["frame_name"] = frame_cluster_score_df["frame_name"].astype(str)

# Dictionary: cluster_id → list of primary frames
cluster_to_primary = {}

for cluster_id, subdf in frame_cluster_score_df.groupby("cluster_id"):

    # Filter frames below threshold
    filtered = subdf[subdf["global_score"] > PRIMARY_FRAME_MINIMUM_GLOBAL_SCORE]

    # Sort by global_score DESCENDING (high → low)
    #filtered = filtered.sort_values(by="global_score", ascending=False)
    filtered = filtered.sort_values(by="tf_idf", ascending=False)

    # Build list of "frame:score"
    primary_list = [
        f"{row['frame_name']}:{row['global_score']}:{row['tf_idf']}"
        for _, row in filtered.iterrows()
    ]

    # Join into comma-separated string
    primary_frames_str = ", ".join(primary_list)

    cluster_to_primary[cluster_id] = primary_frames_str


# final dataframe
out_df = pd.DataFrame([
    {"cluster_id": cid, "primary_frames": frames}
    for cid, frames in cluster_to_primary.items()
])

out_df.to_csv(kg_cluster_primary_frame_file, index=False)

print(f"\nSaved:{kg_cluster_primary_frame_file}" )
print(out_df.head())



# ============================================================
# VALIDATION LOGIC
# ============================================================

def validate_frame(frame_name):
    print("\n\n=== VALIDATION FOR FRAME:", frame_name, "===")

    if frame_name not in frame_to_clusters:
        print("Frame NOT found in any VALID cluster")
        return

    print("Appears in clusters:", frame_to_clusters[frame_name])
    print("Global Score:", global_scores[frame_name])

    print("\nCluster-wise details:")
    for cid in frame_to_clusters[frame_name]:
        freq = frame_freq_cluster[frame_name][cid]
        loc = local_scores[frame_name][cid]
        tfidf = loc * global_scores[frame_name]

        print(
            f"  Cluster {cid}: freq={freq}, "
            f"local={loc:.4f}, tfidf={tfidf:.4f}"
        )


# Example: validate_frame("RTS frame")
##########################################################################

