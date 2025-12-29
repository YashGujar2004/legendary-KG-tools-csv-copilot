
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
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# --- Cosine similarity ---
def x_cosine_similarity(a, b):
    try:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        print(f"[Error] Cosine similarity failed: {e}")
        return 0.0

def chunk_text_safe(text, max_tokens=MAX_TOKENS, token_per_word=1.3):
    """
    Simple heuristic chunking:
    - Uses 1 word ≈ 1.3 tokens (safe estimate).
    - Ensures no chunk exceeds max_tokens (approx).
    """
    if not text:
        return []

    words = text.split()
    max_words = max(1, int(max_tokens / token_per_word))
    
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    
    return chunks

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
    print("embedding section number..", sec_id)
#   chunks = chunk_text(content)

    chunks = chunk_text_safe(content, max_tokens=MAX_TOKENS, token_per_word=1.6)

    if not chunks:
        print(f"[Warning] Empty content for section {sec_id}")
        return None

    print("num of chunk in chunks..")
    for i, chunk in enumerate(chunks):
      tokens = len(tokenizer.encode(chunk))
      print(f"Chunk {i}: {tokens} tokens")

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
        emb, feats, _ = detect_feature_sections_hybrid(child_node,
                                                       embeddings_cache,
                                                       inter_child_threshold,
                                                       child_internal_threshold)
        if emb is not None:
            child_embeddings.append((child_key, emb))
        child_features.extend(feats)

    # --- Inter-child similarity computation ---
    sims = []
    if len(child_embeddings) > 1:
        for (key_i, emb_i), (key_j, emb_j) in combinations(child_embeddings, 2):
            sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
            sims.append(sim)
            print(f"[InterChildSim] {key_i} <-> {key_j} = {sim:.3f}")

    # --- Evaluate parent-level cohesion ---
    parent_feature = False
    if sims:
        n_pairs = len(sims)
        n_high = sum(1 for s in sims if s > inter_child_threshold)
        fraction_high = n_high / n_pairs
        inter_child_stats[node["data"]["section_number"]] = {
            "num_pairs": n_pairs,
            "num_high": n_high,
            "fraction_high": fraction_high,
        }

        print(
            f"[ParentCheck] {node['data']['section_number']}: "
            f"{n_pairs} pairs, {n_high} > {inter_child_threshold}, "
            f"fraction={fraction_high:.2f}"
        )

        if fraction_high > 0.1:
            parent_feature = True
            print(f"[FeatureIdentified] {node['data']['section_number']} marked as feature (inter-child).")

    # --- Fallback: use child-internal cohesion ---
    elif len(child_embeddings) == 1:
        print(f"[Info] {node['data']['section_number']} has only one child → fallback not applicable.")
    else:
        cohesive_children = 0
        total_children = len(children)
        for child_node in children.values():
            child_stats = inter_child_stats.get(child_node["data"]["section_number"])
            if child_stats and child_stats["fraction_high"] > child_internal_threshold:
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
   # print("[Full Hierarchy]")
   # for root_key, root_node in tree.items():
   #   print_hierarchy(root_node)

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

   
