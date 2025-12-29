
import json
import numpy as np
from collections import defaultdict
from openai import OpenAI

# --- Config ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_CACHE_FILE = "embeddings_cache.json"
MAX_TOKENS = 8000  # Max tokens per chunk

# Initialize OpenAI client
client = OpenAI(api_key="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")

# ============================================================
#  🧠  CLUSTERING SECTION: Identify groups of similar features
# ============================================================
import re
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
#  🧠  CLUSTERING SECTION: Identify groups of similar top features
# ============================================================
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from itertools import combinations

def cosine_similarity_internal(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

import numpy as np
import itertools

#Approach 4: clustering child embeddings

def is_parent_cohesive_child_cluster(child_embeddings, distance_threshold=0.275, cluster_fraction_threshold=0.65):
    """
    Determine if a parent section is cohesive by clustering its children's embeddings.
    """

    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    # --- Step 1: Extract embedding matrix ---
    keys, embs = zip(*child_embeddings)
    X = np.vstack(embs)

    # --- Step 2: Compute distance matrix ---
    sim_matrix = np.dot(X, X.T) / (np.linalg.norm(X, axis=1, keepdims=True) * np.linalg.norm(X, axis=1, keepdims=True).T)
    dist_matrix = 1 - sim_matrix

    # --- Step 3: Cluster embeddings ---
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold
    )
    labels = clustering.fit_predict(dist_matrix)

    # --- Step 4: Analyze cluster structure ---
    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique)
    largest_cluster_frac = np.max(counts) / len(keys)

    cohesive = False
    reason = None

    if n_clusters == 1:
        cohesive = True
        reason = "COHESIVE single cluster"
    elif largest_cluster_frac >= cluster_fraction_threshold:
        cohesive = True
        reason = f"COHESIVE dominant cluster covers {largest_cluster_frac:.2f} fraction"
    else:
        cohesive = False
        reason = f"NON COHESIVE multiple clusters ({n_clusters}), dominant only {largest_cluster_frac:.2f}"

    # --- Step 5: Return structured result ---
    stats = {
        "n_clusters": n_clusters,
        "cluster_sizes": dict(zip(unique, counts)),
        "largest_cluster_frac": largest_cluster_frac,
        "reason": reason
    }

    print(f"[ClusterCohesion] clusters={n_clusters}, sizes={counts.tolist()}, "
          f"largest_frac={largest_cluster_frac:.2f}, Cohesive={cohesive} ({reason})")

    return cohesive, stats

#Approach 3: We do adaptive adjustment of IQR

def is_parent_cohesive_adaptive_iqr(child_embeddings, inter_child_threshold=0.7):
    """
    Evaluate if a parent node is cohesive based on its children's embeddings.
    Tightened + variance-aware version:
      - Uses IQR to prune outliers.
      - Adaptive threshold with higher strictness.
      - Requires stronger statistical evidence for cohesion.
    """

    import numpy as np
    from itertools import combinations

    # --- Step 1: compute pairwise similarities ---
    sims = []
    pair_labels = []
    for (k1, e1), (k2, e2) in combinations(child_embeddings, 2):
        sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))
        sims.append(sim)
        pair_labels.append((k1, k2))

    if not sims:
        return False, {"error": "no_pairs"}

    sims = np.array(sims)

    # --- Step 2: compute mean similarity per child (for outlier pruning) ---
    mean_per_child = {}
    for k, emb in child_embeddings:
        child_sims = [
            float(np.dot(emb, e2) / (np.linalg.norm(emb) * np.linalg.norm(e2) + 1e-10))
            for (k2, e2) in child_embeddings if k2 != k
        ]
        mean_per_child[k] = np.mean(child_sims)

    means = np.array(list(mean_per_child.values()))
    median = np.median(means)
    q1, q3 = np.percentile(means, [25, 75])
    iqr = q3 - q1
    lower_bound = median - 1.0 * iqr  # tighter cutoff for outliers

    # --- Step 3: filter out child outliers ---
    filtered_children = [k for k, m in mean_per_child.items() if m >= lower_bound]
    outliers = [k for k, m in mean_per_child.items() if m < lower_bound]

    filtered_pairs = [
        (p, s) for p, s in zip(pair_labels, sims)
        if p[0] in filtered_children and p[1] in filtered_children
    ]
    filtered_sims = np.array([s for _, s in filtered_pairs]) if filtered_pairs else sims

    # --- Step 4: compute cohesion metrics ---
    mean_sim = np.mean(filtered_sims)
    tp75 = np.percentile(filtered_sims, 75)
    std_sim = np.std(filtered_sims)
    frac_high = np.mean(filtered_sims >= 0.7)
    adaptive_thr = np.median(filtered_sims) - 0.02  # stricter adaptive baseline

    # --- Step 5: tightened cohesion rules ---
    cohesive = False
    reason = None

    if (mean_sim >= adaptive_thr + 0.02 and tp75 >= 0.75):
        cohesive = True
        reason = f"mean_sim ({mean_sim:.3f}) ≥ adaptive_thr+0.02 ({adaptive_thr+0.02:.3f}) and tp75 ({tp75:.3f}) ≥ 0.75"
    elif frac_high >= 0.45:
        cohesive = True
        reason = f"frac_high ({frac_high:.2f}) ≥ 0.45"
    elif std_sim < 0.06 and mean_sim >= 0.72:
        cohesive = True
        reason = f"std_sim ({std_sim:.3f}) < 0.06 and mean_sim ({mean_sim:.3f}) ≥ 0.72"

    # --- Step 6: build stats dictionary ---
    stats = {
        "mean": mean_sim,
        "tp75": tp75,
        "frac_high": frac_high,
        "std": std_sim,
        "adaptive_thr": adaptive_thr,
        "num_pairs": len(filtered_sims),
        "outliers": outliers,
        "filtered": len(outliers) > 0,
        "reason": reason if cohesive else "not cohesive",
    }

    # --- Step 7: print diagnostics ---
    for k, m in mean_per_child.items():
        flag = "✅ ok" if m >= lower_bound else "❌ outlier"
        print(f"   {k}: mean={m:.3f} {flag}")

    print(f"   Median={median:.3f}, IQR={iqr:.3f}, LowerBound={lower_bound:.3f}")
    print(
        f"[ParentCohesion] Mean={mean_sim:.3f}, TP75={tp75:.3f}, "
        f"Std={std_sim:.3f}, Frac≥0.7={frac_high:.2f}, "
        f"AdaptiveThr={adaptive_thr:.3f}, Cohesive={cohesive}"
    )

    if cohesive:
        print(f"✅ [CohesionReason] Parent is cohesive because {reason}\n")
    else:
        print(f"❌ [CohesionReason] Parent is NOT cohesive — no rule satisfied.\n")

    return cohesive, stats

#Approach 2: IQR + fraction-limit version to remove the outlier problem found in approach 1:

def is_parent_cohesive_iqr(
    child_embeddings,
    inter_child_threshold=0.7,     # cosine similarity threshold for "strong relation"
    iqr_factor=1.5,                # controls IQR sensitivity (Tukey fence)
    max_outlier_fraction=0.2,      # allow pruning up to 20% of children
    tp75_bonus=0.05,               # allow +0.05 margin for TP75 rule
    frac_high_limit=0.5            # require at least 50% pairs above threshold
):
    """
    Determine if a parent section is cohesive based on pairwise cosine similarities
    among its child embeddings, using adaptive (IQR-based) outlier suppression.

    Steps:
      1. Compute all pairwise cosine similarities.
      2. Compute mean similarity per child.
      3. Detect and remove outlier children using median ± iqr_factor * IQR rule.
      4. Limit removals to <= max_outlier_fraction of children.
      5. Recompute cohesion metrics on filtered pairs.
      6. Decide final cohesion using mean, TP75, and fraction-high rules.
    """

    # --- Guard ---
    if len(child_embeddings) < 2:
        return False, {"status": "insufficient_children"}

    # --- Step 1: pairwise similarities ---
    child_keys = [k for k, _ in child_embeddings]
    child_vecs = [v for _, v in child_embeddings]
    sims = []
    for (i, a), (j, b) in itertools.combinations(enumerate(child_vecs), 2):
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        sims.append((child_keys[i], child_keys[j], sim))

    if not sims:
        return False, {"status": "no_pairs"}

    # --- Step 2: mean similarity per child ---
    child_means = {k: [] for k in child_keys}
    for c1, c2, s in sims:
        child_means[c1].append(s)
        child_means[c2].append(s)
    mean_per_child = {k: np.mean(v) if v else 0 for k, v in child_means.items()}

    # --- Step 3: compute IQR and identify outliers ---
    means = np.array(list(mean_per_child.values()))
    median = np.median(means)
    q1, q3 = np.percentile(means, [25, 75])
    iqr = q3 - q1
    lower_bound = median - iqr_factor * iqr

    outliers = [k for k, mean in mean_per_child.items() if mean < lower_bound]

    # --- Step 4: enforce max outlier limit ---
    max_allowed = int(len(child_keys) * max_outlier_fraction)
    if len(outliers) > max_allowed:
        print(f"[OutlierCheck] Too many ({len(outliers)}) below threshold; ignoring outlier removal.")
        outliers = []

    inliers = [k for k in child_keys if k not in outliers]

    print(f"\n[OutlierCheck] Children: {child_keys}")
    for k, m in mean_per_child.items():
        tag = "❌ outlier" if k in outliers else "✅ ok"
        print(f"  {k}: mean={m:.3f} {tag}")
    print(f"  Median={median:.3f}, IQR={iqr:.3f}, LowerBound={lower_bound:.3f}")

    # --- Step 5: filter sims to exclude outlier pairs ---
    filtered_sims = [s for c1, c2, s in sims if c1 in inliers and c2 in inliers]
    if not filtered_sims:
        return False, {
            "mean": np.mean([s for _, _, s in sims]),
            "tp75": np.percentile([s for _, _, s in sims], 75),
            "frac_high": 0,
            "outliers": outliers,
            "filtered": False
        }

    # --- Step 6: compute metrics ---
    mean_sim = np.mean(filtered_sims)
    tp75 = np.percentile(filtered_sims, 75)
    frac_high = np.mean(np.array(filtered_sims) > inter_child_threshold)

    # --- Step 7: final decision logic ---
    cohesive = (
        (mean_sim >= inter_child_threshold) or
        (tp75 >= inter_child_threshold + tp75_bonus) or
        (frac_high >= frac_high_limit)
    )

    print(f"[ParentCohesion] Mean={mean_sim:.3f}, TP75={tp75:.3f}, "
          f"Frac≥{inter_child_threshold}={frac_high:.2f}, Cohesive={cohesive}")

    # --- Step 8: return detailed stats ---
    stats = {
        "mean": mean_sim,
        "tp75": tp75,
        "frac_high": frac_high,
        "num_pairs": len(filtered_sims),
        "num_high": int(sum(s > inter_child_threshold for s in filtered_sims)),
        "outliers": outliers,
        "filtered": True
    }

    return cohesive, stats

#Approach 1: is parent cohesive method uses mean, std, p75 to identify cohesiveness
# we found an issue that outliers disturbs the outcome

def is_parent_cohesive_mean(children_embeddings, inter_child_threshold=0.7, min_fraction_high=0.6):
    """
    Determine whether a parent node is cohesive based on its children's embeddings.
    Considers outliers and sub-clusters intelligently.
    """
    if len(children_embeddings) < 2:
        return False, {"reason": "Insufficient children"}

    sims = []
    for (key_i, emb_i), (key_j, emb_j) in combinations(children_embeddings, 2):
        sim = cosine_similarity_internal(emb_i, emb_j)
        sims.append(sim)

    sims = np.array(sims)
    mean_sim = np.mean(sims)
    std_sim = np.std(sims)
    p75_sim = np.percentile(sims, 75)
    fraction_high = np.mean(sims > inter_child_threshold)
    top_half_mean = np.mean(np.sort(sims)[len(sims)//2:])  # captures cohesive core

    # Improved decision rule
    cohesive = (
        (mean_sim >= 0.67 and p75_sim >= 0.73 and std_sim <= 0.10)
        or (fraction_high >= min_fraction_high)
        or (top_half_mean > 0.75 and std_sim <= 0.12)
    )

    summary = {
        "mean": float(mean_sim),
        "std": float(std_sim),
        "p75": float(p75_sim),
        "fraction_high": float(fraction_high),
        "top_half_mean": float(top_half_mean),
        "n_pairs": len(sims),
        "decision": "Cohesive" if cohesive else "Non-Cohesive"
    }

    return cohesive, summary


# --- Step 1: Build similarity matrix directly from embeddings ---
def build_similarity_matrix_from_embeddings(top_feature_embeddings):
    """
    Compute pairwise cosine similarity among top feature embeddings.

    Args:
        top_feature_embeddings (dict): {section_number: embedding_vector}

    Returns:
        (sections, sim_matrix)
    """
    if not top_feature_embeddings:
        raise ValueError("[Error] No embeddings provided for clustering.")

    sections = sorted(top_feature_embeddings.keys())
    embeddings = np.array([top_feature_embeddings[s] for s in sections])

    # Compute cosine similarity
    sim_matrix = cosine_similarity(embeddings)
    print(f"[Info] Built similarity matrix directly from {len(sections)} embeddings")

    return sections, sim_matrix


# --- Step 2: Perform clustering ---
def cluster_sections_from_similarity(sections, sim_matrix, threshold=0.20):
    """
    Perform agglomerative clustering using precomputed cosine similarity.
    Distance = 1 - similarity.
    """
    dist_matrix = 1 - sim_matrix

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold
    )
    labels = clustering.fit_predict(dist_matrix)

    clusters = pd.DataFrame({"Section": sections, "Cluster": labels}).sort_values("Cluster")
    print(f"[Info] Formed {clusters['Cluster'].nunique()} clusters.")
    return clusters



# --- Step 3: Integration wrapper ---
def cluster_top_features_from_embeddings(top_feature_list, top_feature_embeddings, threshold):
    """
    Full workflow:
      1. Compute cosine similarity matrix from embeddings
      2. Cluster based on similarity
    """
    # Step 1: Build similarity matrix
    sections, sim_matrix = build_similarity_matrix_from_embeddings(top_feature_embeddings)

    # Step 2: Cluster
    clusters = cluster_sections_from_similarity(sections, sim_matrix, threshold)

    # Step 3: Create embedding map per cluster
    cluster_embeddings = {
        sec: top_feature_embeddings.get(sec)
        for sec in sections
        if sec in top_feature_embeddings
    }

    return clusters, cluster_embeddings

def cluster_all_sections_from_embeddings(all_section_embeddings, threshold):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np, pandas as pd

    sections = sorted(all_section_embeddings.keys())
    embeddings = np.array([all_section_embeddings[s] for s in sections])

    print(f"[Info] Clustering {len(sections)} sections...")

    # Compute cosine similarity and convert to distance
    sim_matrix = cosine_similarity(embeddings, embeddings)
    dist_matrix = 1 - sim_matrix

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold
    )
    labels = clustering.fit_predict(dist_matrix)

    clusters = pd.DataFrame({"Section": sections, "Cluster": labels}).sort_values("Cluster")
    print(f"[Info] Formed {clusters['Cluster'].nunique()} clusters.")
    return clusters


# ============================================================
#  🚀  CLUSTER EXECUTION — integrate after top_feature_list
# ============================================================

# Example:
# top_feature_list, promoted_map, top_feature_embeddings = get_top_feature_list(tree, features)

CLUSTER_THRESHOLD = 0.20   # tweak this value to control cluster granularity

# --- Cosine similarity ---
def x_cosine_similarity(a, b):
    try:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        print(f"[Error] Cosine similarity failed: {e}")
        return 0.0

# --- Chunk long text ---
def chunk_text(text, max_tokens=MAX_TOKENS):
    if not text:
        return []
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        current_chunk.append(word)
        current_tokens += 1
        if current_tokens >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# --- Embedding fetcher with chunking ---
def get_embedding(section, embeddings_cache):
    sec_id = section.get("section_number", None)
    if not sec_id:
        print("[Error] Section missing section_number")
        return None

    if "embedding" in section:
        return section["embedding"]
    if sec_id in embeddings_cache:
        section["embedding"] = embeddings_cache[sec_id]
        return section["embedding"]

    content = section.get("content", "")
    chunks = chunk_text(content)
    if not chunks:
        print(f"[Warning] Empty content for section {sec_id}")
        return None

    try:
        embeddings = []
        for chunk in chunks:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
            embeddings.append(resp.data[0].embedding)
        avg_emb = np.mean(embeddings, axis=0)
        section["embedding"] = avg_emb
        embeddings_cache[sec_id] = avg_emb
        print(f"[Info] Fetched embedding for section {sec_id} ({len(chunks)} chunks)")
        return avg_emb
    except Exception as e:
        print(f"[Embedding error] Section {sec_id}: {e}")
        return None

# --- Build hierarchy ---
def build_hierarchy(data):
    tree = {}
    nodes = {}
    for idx, entry in enumerate(data):
        sec_raw = entry.get("section_number", None)
        if not sec_raw:
            print(f"[Warning] Skipping entry {idx} with missing section_number")
            continue
        sec = str(sec_raw).strip(".")
        nodes[sec] = {"data": entry, "embedding": None, "children": {}}

    for sec, node in nodes.items():
        if "." in sec:
            parent = ".".join(sec.split(".")[:-1])
            if parent in nodes:
                nodes[parent]["children"][sec] = node
            else:
                tree[sec] = node
        else:
            tree[sec] = node

    print("[Info] Full hierarchy created successfully")
    return tree, nodes

import numpy as np

import numpy as np
from itertools import combinations

def detect_feature_sections_hybrid(node, embeddings_cache,
                                   inter_child_threshold=0.7,
                                   child_internal_threshold=0.5):
    """
    Hybrid feature detection:
    - Use inter-child similarity as primary signal for parent cohesion.
    - Fall back to child-internal cohesion if inter-child sims are insufficient.
    """

    children = node.get("children", {})
    feature_sections = []
    inter_child_stats = {}

    # --- Leaf case ---
    if not children:
        emb = get_embedding(node.get("data", {}), embeddings_cache)
        if emb is not None:
            node["embedding"] = np.array(emb)
        else:
            print(f"[Warn] Leaf {node.get('data', {}).get('section_number')} missing embedding")
        return node.get("embedding"), feature_sections, inter_child_stats

    # --- Recursive compute for children ---
    child_embeddings = []
    child_features = []
    for child_key, child_node in children.items():
        emb, feats, _ = detect_feature_sections_hybrid(
            child_node,
            embeddings_cache,
            inter_child_threshold,
            child_internal_threshold
        )
        if emb is not None:
            child_embeddings.append((child_key, emb))
        child_features.extend(feats)

    # --- Evaluate parent-level cohesion using statistical method ---
    parent_feature = False
    if len(child_embeddings) > 1:
        cohesive, stats = is_parent_cohesive_child_cluster(child_embeddings,distance_threshold=0.275, cluster_fraction_threshold=0.65)
        inter_child_stats[node["data"]["section_number"]] = stats

        print(f"[ParentCheck] {node['data']['section_number']}: {stats}")

        if cohesive:
            parent_feature = True
            print(f"[FeatureIdentified] {node['data']['section_number']} marked as feature (cohesive).")
    elif len(child_embeddings) == 1:
        print(f"[Info] {node['data']['section_number']} has only one child → skipping cohesion check.")
    else:
        print(f"[Info] {node['data']['section_number']} has no valid embeddings for children.")

    # --- Fallback: use child-internal cohesion ---
    if not parent_feature:
        cohesive_children = 0
        total_children = len(children)
        for child_node in children.values():
            child_stats = inter_child_stats.get(child_node["data"]["section_number"])
            if child_stats and child_stats.get("fraction_high", 0) > child_internal_threshold:
                cohesive_children += 1

        fraction_cohesive = cohesive_children / total_children if total_children > 0 else 0
        print(f"[Fallback] {node['data']['section_number']} - cohesive children fraction: {fraction_cohesive:.2f}")
        if fraction_cohesive > 0.5:
            parent_feature = True
            print(f"[FeatureIdentified] {node['data']['section_number']} marked as feature (fallback).")

    # --- Compute aggregated embedding for current node ---
    child_embs_only = [emb for _, emb in child_embeddings]
    self_emb = get_embedding(node.get("data", {}), embeddings_cache)
    combined_embeddings = child_embs_only + ([self_emb] if self_emb is not None else [])
    if combined_embeddings:
        node["embedding"] = np.mean(combined_embeddings, axis=0)

    # --- Mark this section as feature if applicable ---
    if parent_feature:
        feature_sections.append(node["data"]["section_number"])

    return node.get("embedding"), feature_sections + child_features, inter_child_stats

def propagate_features(parent, nodes, features):
    """
    Maintain feature consistency:
      - Keep only true bottom-up detected features.
      - Do NOT mark children just because a parent isn't a feature.
      - Preserve existing child features as independent.
    """
    new_features = set(features)

    for section, node in nodes.items():
        children = node.get("children", {})

        # Recurse first (depth-first)
        new_features |= propagate_features(section, children, new_features)

        # CASE 1: Node itself is already a feature
        if section in features:
            print(f"[Keep] Section {section} already marked as feature.")
            continue

        # CASE 2: Node not a feature but has feature children
        if any(child in features for child in children.keys()):
            print(f"[Info] Parent {section} not marked, but child features retained independently.")
            # Do not propagate anything downward
            continue

        # CASE 3: Node not a feature, and no child features
        # (No action needed)
        if not children:
            print(f"[Skip] Leaf {section} has no feature children and not a feature itself.")

    # If this is a root call, summarize
    if parent is None:
        print("\n[Summary] Propagation complete.")
        print(f"  → Total features retained: {len(new_features)}")

    return new_features

# --- Print hierarchy recursively ---
def print_hierarchy(node, level=0):
    indent = "    " * level
    sec_number = node.get("data", {}).get("section_number", "Unknown")
    children_keys = list(node.get("children", {}).keys())
    print(f"{indent}Section: {sec_number}, Children: {children_keys}")
    for child_node in node.get("children", {}).values():
        print_hierarchy(child_node, level + 1)
# Step 3: Extract top-level parents from features
def get_top_parents(parents_list):
    # Sort by hierarchical order
    parents_list = sorted(parents_list, key=lambda x: [int(p) for p in x.split('.') if p])
    parents_top_list = []

    for parent in parents_list:
        # Skip if parent is a child of any already added top parent
        if any(parent.startswith(tp + '.') for tp in parents_top_list):
            continue
        parents_top_list.append(parent)

    return parents_top_list

def make_json_safe(obj):
    """Recursively convert NumPy arrays and other non-serializable types into JSON-safe formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def get_top_feature_list(tree, features):
    """
    Identify top-level features following these exact rules:

    1. If a node's parent is a feature → none of its descendants are added.
    2. If a node's parent is not a feature:
         2.1 If the node is a feature → add it to top_features, stop descending.
         2.2 If the node is NOT a feature → for each immediate child:
              - find the most top (highest) descendant that is a feature → add it.
              - if no feature exists, take leaf sections instead.
    3. Prune any section whose ancestor is already in top_features.

    Returns:
        tuple: (pruned_top_features, promoted_map, top_feature_embeddings)
    """
    features = set(features)
    top_features = set()
    promoted_map = {}
    top_feature_embeddings = {}

    # --- Utility: gather top-most features in subtree ---
    def topmost_features_in_subtree(node):
        found = set()

        def dfs(n, ancestor_feature_found=False):
            sec = n.get("data", {}).get("section_number")
            is_feature = sec in features
            children = n.get("children", {})
            if is_feature and not ancestor_feature_found:
                found.add(sec)
                return
            for ch in children.values():
                dfs(ch, ancestor_feature_found or is_feature)

        dfs(node)
        return found

    # --- Utility: gather leaves in subtree ---
    def leaves_in_subtree(node):
        leaves = []
        def dfs(n):
            ch = n.get("children", {})
            sec = n.get("data", {}).get("section_number")
            if not ch:
                leaves.append(sec)
            else:
                for c in ch.values():
                    dfs(c)
        dfs(node)
        return leaves

    # --- Recursive traversal following your rules ---
    def traverse(node, parent_is_feature=False):
        sec = node.get("data", {}).get("section_number")
        is_feature = sec in features
        children = node.get("children", {})

        # Rule 1: parent is feature → skip subtree
        if parent_is_feature:
            return

        # Rule 2.1: parent not feature & node is feature
        if is_feature:
            top_features.add(sec)
            # ✅ Add embedding if available
            if node.get("embedding") is not None:
                top_feature_embeddings[sec] = node["embedding"].tolist()
            print(f"[TopFeature] Added {sec} (feature, parent not feature)")
            return

        # Rule 2.2 / 2.3: node not feature
        if children:
            for child in children.values():
                top_feats = topmost_features_in_subtree(child)
                if top_feats:
                    # ✅ Found at least one feature in subtree
                    for f in top_feats:
                        top_features.add(f)
                        if sec in features:
                            promoted_map.setdefault(sec, []).append(f)
                        # ✅ Add embedding if available
                        emb_node = find_node_by_section(tree, f)
                        if emb_node and emb_node.get("embedding") is not None:
                            top_feature_embeddings[f] = emb_node["embedding"].tolist()
                        print(f"[Promote] Under {sec}, promoting top feature {f}")
                else:
                    # ❗ No features found in child's subtree — could be a leaf or a small subtree
                    leaves = leaves_in_subtree(child)
                    if leaves:
                        for leaf in leaves:
                            top_features.add(leaf)
                            if sec in features:
                                promoted_map.setdefault(sec, []).append(leaf)
                            emb_node = find_node_by_section(tree, leaf)
                            if emb_node and emb_node.get("embedding") is not None:
                                top_feature_embeddings[leaf] = emb_node["embedding"].tolist()
                            print(f"[Promote-Leaf] Under {sec}, promoting leaf {leaf}")
                    else:
                        # Edge case: direct leaf child itself
                        child_sec = child.get("data", {}).get("section_number")
                        if child_sec:
                            top_features.add(child_sec)
                            if sec in features:
                                promoted_map.setdefault(sec, []).append(child_sec)
                            if child.get("embedding") is not None:
                                top_feature_embeddings[child_sec] = child["embedding"].tolist()
                            print(f"[Promote-LeafDirect] Under {sec}, promoting direct leaf {child_sec}")

        # Recurse further down to explore deeper non-feature nodes
        for ch in children.values():
            traverse(ch, parent_is_feature=is_feature)

    # --- Helper to find a node by section number ---
    def find_node_by_section(tree, sec_number):
        """Return the node dict corresponding to section_number."""
        for _, node in tree.items():
            if node.get("data", {}).get("section_number") == sec_number:
                return node
            res = find_node_by_section(node.get("children", {}), sec_number)
            if res:
                return res
        return None

    # --- Start traversal for each root node ---
    for root_key, root_node in tree.items():
        traverse(root_node, parent_is_feature=False)

    # --- Prune: remove any section whose ancestor is already in top_features ---
    def has_ancestor_in_top(sec, topset):
        parts = sec.split(".")
        for i in range(1, len(parts)):
            anc = ".".join(parts[:i])
            if anc in topset:
                return True
        return False

    pruned = []
    for sec in sorted(top_features):
        if not has_ancestor_in_top(sec, top_features - {sec}):
            pruned.append(sec)
        else:
            print(f"[Pruned] Removed {sec} (ancestor already top feature)")

    print(f"\n[Summary] Total top features after prune: {len(pruned)}")
    return pruned, promoted_map, top_feature_embeddings


# --- Usage ---
if __name__ == "__main__":
    try:
        with open("80211-2020-chunks-trim.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Info] Loaded {len(data)} sections from JSON")
    except Exception as e:
        print(f"[Error] Failed to load JSON: {e}")
        data = []

# Step 1: Build hierarchy
    tree, nodes = build_hierarchy(data)

# Step 1a: Print hierarchy
    #print("[Full Hierarchy]")
    #for root_key, root_node in tree.items():
      #print_hierarchy(root_node)

# Step 2: Hybrid feature detection
    print("[Info] Starting hybrid feature detection (inter-child + fallback)...")

    try:
      with open(EMBEDDING_CACHE_FILE, "r") as f:
        embeddings_cache = json.load(f)
    except:
      embeddings_cache = {}

    all_features = []
    all_stats = {}

    for root_key, root_node in tree.items():
      _, feats, stats = detect_feature_sections_hybrid(
        root_node,
        embeddings_cache,
        inter_child_threshold=0.8,
        child_internal_threshold=0.5
      )
      all_features.extend(feats)
      all_stats.update(stats)
    features = set(all_features)

# Save updated cache safely
    try:
      with open(EMBEDDING_CACHE_FILE, "w") as f:
        json.dump(make_json_safe(embeddings_cache), f)
        print(f"[Info] Embedding cache saved successfully to {EMBEDDING_CACHE_FILE}")
    except Exception as e:
      print(f"[Error] Failed to save embeddings cache: {e}")

    features = set(all_features)
    embeddings = {k: v["embedding"].tolist() for k, v in nodes.items() if v.get("embedding") is not None}

    print("\n[Summary] Hybrid detection complete.")
    print(f"[Summary] Total features detected: {len(features)}")
    print("[Output] Feature sections:", features)

# Step 3: Propagate features
    features = propagate_features(None, nodes, features)
    print("[Output] Final feature sections after propagation:", features)

# Step 4: List sections with embeddings
    print("[Output] Sections with embeddings:", list(embeddings.keys()))

# Step 5: Get top-level parents and features
    top_parent_list = get_top_parents(features)

    print("\n[Output] Top Parent Sections:", top_parent_list)

    top_feature_list, promoted_map, top_feature_embeddings = get_top_feature_list(tree, features)

# ✅ Filter out non-feature parents when printing
    filtered_map = {k: v for k, v in promoted_map.items() if k in features}

    print("\n[Output] Top Feature Sections:", sorted(top_feature_list))

    try:
      clusters, cluster_embeddings = cluster_top_features_from_embeddings(
        top_feature_list,
        top_feature_embeddings,
        threshold=0.275
      )

      print("\n[Output] 🧩 Clustered Top Features:")
      print(clusters)

    # Optional: Save clusters to CSV
      clusters.to_csv("top_feature_clusters.csv", index=False)
      print("[Saved] top_feature_clusters.csv")

    except Exception as e:
      print(f"[Error] Clustering failed: {e}")

# === Exhaustive Clustering of All Sections ===
    try:
      all_clusters = cluster_all_sections_from_embeddings(embeddings, threshold=0.275)
      all_clusters.to_csv("all_section_clusters.csv", index=False)
      print("[Saved] all_section_clusters.csv")
    except Exception as e:
      print(f"[Error] All-section clustering failed: {e}")

   
