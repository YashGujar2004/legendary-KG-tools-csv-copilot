
import json
import os
import numpy as np
from collections import defaultdict
from openai import OpenAI

# --- Config ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
EMBEDDING_CACHE_FILE = os.environ.get("EMBEDDING_CACHE_FILE")
CHUNKS_FILE = os.environ.get("SPEC_CHUNKS_TRIM")

TREE_HIERARCHY_FILE = os.environ.get("SPEC_SECTIONS_TREE_HIERARCHY_FILE")

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

CLUSTER_THRESHOLD = 0.20   # tweak this value to control cluster granularity
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")



# ============================================================
#  🧠  CLUSTERING SECTION: Identify groups of similar top features
# ============================================================
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

import json
import numpy as np

TOLERANCE = 0.1                  # acceptable deviation threshold


# -----------------------------
# Helper functions
# -----------------------------
def normalize_section_number(s):
    return str(s).strip().rstrip(".")

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def verify_parent_embeddings(tree, embeddings_cache, tolerance=TOLERANCE):
    """
    Verify that each parent's embedding equals:
        E(parent) = normalize(mean(E_self + E_children))
    """

    def _verify_node(node):
        sec = normalize_section_number(node["data"].get("section_number"))
        children = node.get("children", {})
        if not children:
            return

        # collect child embeddings
        child_embs = []
        for child_key, child_node in children.items():
            child_sec = normalize_section_number(child_node["data"]["section_number"])
            emb = embeddings_cache.get(child_sec)
            if emb is not None:
                child_embs.append(emb)
            _verify_node(child_node)

        if not child_embs:
            return

        # self and parent embeddings
        E_self = embeddings_cache.get(sec)
        E_parent = embeddings_cache.get(sec)
        if E_parent is None:
            print(f"[Missing] Parent embedding missing for {sec}")
            return
        if E_self is not None:
            emb_list = [E_self] + child_embs
        else:
            emb_list = child_embs

        # expected parent embedding (same formula as builder)
        expected = normalize(np.mean(emb_list, axis=0))

        diff = np.linalg.norm(E_parent - expected)
        if diff > tolerance:
            print(f"[Mismatch] {sec}: deviation={diff:.6f}")
        else:
            print(f"[OK] {sec}: deviation={diff:.6f}")

        cosine = np.dot(E_parent, expected) / (np.linalg.norm(E_parent) * np.linalg.norm(expected))
        print(f"[Info] {sec}: cosine={cosine:.6f}, deviation={diff:.6f}")

    _verify_node(tree)
    print("[Info] Validation complete using mean-aggregation method.")

def save_tree_to_json(tree, filepath=TREE_HIERARCHY_FILE):
    """
    Recursively saves the hierarchy tree into JSON format.
    Converts numpy arrays (embeddings) to lists.
    """
    def serialize_node(node):
        serialized = {
            "data": node.get("data", {}),
            "embedding": node["embedding"].tolist() if isinstance(node.get("embedding"), np.ndarray) else None,
            "children": {k: serialize_node(v) for k, v in node.get("children", {}).items()}
        }
        return serialized

    serializable_tree = {k: serialize_node(v) for k, v in tree.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_tree, f, indent=2)
    print(f"[Info] Tree hierarchy saved to {filepath}")

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
    sec_id = str(section.get("section_number", "")).strip().rstrip(".")
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

import numpy as np

import numpy as np

def compute_hierarchical_embeddings(node, embeddings_cache):
    """
    Recursively computes hierarchical embeddings bottom-up.
    Each parent embedding = normalize(mean(E_self + E_children)).

    Ensures every parent is recomputed fresh, overwriting any stale cache entry.
    """

    section_number = str(node["data"].get("section_number", "")).strip().rstrip(".")
    children = node.get("children", {})
    collected_nodes = {}

    # --- (1) Leaf case ---
    if not children:
        emb = get_embedding(node["data"], embeddings_cache)   # uses your existing chunking + cache logic
        if emb is not None:
            emb = np.array(emb)
            emb /= np.linalg.norm(emb)
            node["embedding"] = emb
            embeddings_cache[section_number] = emb
            collected_nodes[section_number] = emb
        return node.get("embedding"), section_number, collected_nodes

    # --- (2) Recursive compute for children ---
    child_embeddings = []
    for child_key, child_node in children.items():
        child_emb, _, child_nodes = compute_hierarchical_embeddings(child_node, embeddings_cache)
        if child_emb is not None:
            child_embeddings.append(child_emb)
            collected_nodes.update(child_nodes)

    # --- (3) Self embedding ---
    self_emb = get_embedding(node["data"], embeddings_cache)
    if self_emb is not None:
        self_emb = np.array(self_emb)
        self_emb /= np.linalg.norm(self_emb)
        child_embeddings.append(self_emb)

    # --- (4) Aggregate parent = mean(E_self + E_children) ---
    if child_embeddings:
        mean_emb = np.mean(child_embeddings, axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        node["embedding"] = mean_emb
        embeddings_cache[section_number] = mean_emb
        collected_nodes[section_number] = mean_emb
        print(f"[Info] Recomputed hierarchical embedding for section {section_number}")

    return node.get("embedding"), section_number, collected_nodes

if __name__ == "__main__":
    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Info] Loaded {len(data)} sections from JSON")
    except Exception as e:
        print(f"[Error] Failed to load JSON: {e}")
        data = []

# Step 1: Build hierarchy
    tree, nodes = build_hierarchy(data)

# Step 1a: Print hierarchy
   # print("[Full Hierarchy]")
    for root_key, root_node in tree.items():
      print_hierarchy(root_node)

    save_tree_to_json(tree, TREE_HIERARCHY_FILE)

# Step 2: Hybrid feature detection
    print("[Info] Starting hybrid feature detection (inter-child + fallback)...")

    try:
      with open(EMBEDDING_CACHE_FILE, "r") as f:
        embeddings_cache = json.load(f)
    except:
      embeddings_cache = {}

    for root_key, root_node in tree.items():
      _, feats, stats = compute_hierarchical_embeddings(
        root_node,
        embeddings_cache,
      )

# Save updated cache safely
    try:
      with open(EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(embeddings_cache), f)
        print(f"[Info] Embedding cache saved successfully to {EMBEDDING_CACHE_FILE}")
    except Exception as e:
      print(f"[Error] Failed to save embeddings cache: {e}")


    verify_parent_embeddings(tree["11"], embeddings_cache)
    verify_parent_embeddings(tree["10"], embeddings_cache)
    verify_parent_embeddings(tree["14"], embeddings_cache)
    verify_parent_embeddings(tree["19"], embeddings_cache)

