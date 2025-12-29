
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import re
import os
import json
import numpy as np

import chromadb

# --- Config ---
KG_CLUSTERS = os.environ.get("KG_CLUSTERS")
KG_CLUSTER_PRIMARY_FRAME = os.environ.get("KG_CLUSTER_PRIMARY_FRAME")
CHROMA_PATH = os.environ.get("CHROMADB_FOR_CLUSTER_SECTION")

EMBEDDING_CACHE_FILE= os.environ.get("EMBEDDING_CACHE_FILE")
CHUNKS_FILE = os.environ.get("SPEC_CHUNKS_TRIM")

# Name of the collection (index) we are creating
COLLECTION_NAME = os.environ.get("CHROMA_CLUSTER_SECTION_COLLECTION_NAME")

# Ensure the Chroma directory exists
os.makedirs(CHROMA_PATH, exist_ok=True)

# --- Step 1: Load embeddings ---
print("[Info] Loading embeddings...")
try:
    with open(EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
        embedding_cache = json.load(f)
    print(f"[Info] Loaded {len(embedding_cache)} embeddings from {EMBEDDING_CACHE_FILE}")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Missing {EMBEDDING_CACHE_FILE}")
except json.JSONDecodeError:
    raise ValueError(f"❌ Invalid JSON format in {EMBEDDING_CACHE_FILE}")

# --- Step 2: Load section content ---
print("[Info] Loading section content...")
try:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    print(f"[Info] Loaded {len(chunks_data)} sections from {CHUNKS_FILE}")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Missing {CHUNKS_FILE}")
except json.JSONDecodeError:
    raise ValueError(f"❌ Invalid JSON format in {CHUNKS_FILE}")

# --- Step 3: Validate alignment between embeddings and content ---
print("\n[Validation] Checking alignment between embeddings and chunks...")

# Normalize section numbers
chunk_sections = {
    str(entry.get("section_number", "")).strip(".")
    for entry in chunks_data if entry.get("section_number")
}
embedding_sections = set(embedding_cache.keys())

# Find mismatches
missing_embeddings = chunk_sections - embedding_sections
missing_contents = embedding_sections - chunk_sections

if missing_embeddings:
    print(f"⚠️  {len(missing_embeddings)} sections have content but missing embeddings:")
    print("   ", sorted(list(missing_embeddings))[:10], "..." if len(missing_embeddings) > 10 else "")
else:
    print("✅ All content sections have embeddings.")

if missing_contents:
    print(f"⚠️  {len(missing_contents)} embeddings have no corresponding content:")
    print("   ", sorted(list(missing_contents))[:10], "..." if len(missing_contents) > 10 else "")
else:
    print("✅ All embeddings have matching content.")

# Optionally, skip sections without both content & embeddings
valid_sections = chunk_sections & embedding_sections
print(f"✅ {len(valid_sections)} sections have both embeddings and content.")

# --- Step 4: Initialize Chroma Persistent Client ---
print("\n[Info] Initializing Chroma vectorstore...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# --- Step 5: Prepare data for insertion ---
ids, embeddings, metadatas, documents = [], [], [], []

seen_ids = {}

for idx, entry in enumerate(chunks_data):
    sec_raw = entry.get("section_number", None)
    if not sec_raw:
        print(f"[Warning] Skipping entry {idx}: missing section_number")
        continue

    sec = str(sec_raw).strip(".")
    if sec not in valid_sections:
        continue

    sec_content = entry.get("content", None)
    if not sec_content:
        print(f"[Warning] Section {sec} has no content — skipped.")
        continue

    emb = embedding_cache.get(sec)
    if emb is None:
        print(f"[Error] Missing embedding for section {sec} — skipped.")
        continue

    emb = np.array(emb, dtype=np.float32)
    if not emb.any():
        print(f"[Error] Empty embedding for section {sec} — skipped.")
        continue

    base_id = sec
    if base_id in seen_ids:
        seen_ids[base_id] += 1
        unique_id = f"{base_id}_{seen_ids[base_id]}"
    else:
        seen_ids[base_id] = 0
        unique_id = base_id

    ids.append(unique_id)
    embeddings.append(emb.tolist())
    documents.append(sec_content)
    metadatas.append({
        "section_number": sec,
        "content_length": len(sec_content)
    })

print(f"\n[Summary] Prepared {len(embeddings)} valid sections for vectorstore.")

# --- Step 6: Add to ChromaDB ---
if embeddings:
    print(f"[Info] Adding {len(embeddings)} entries to Chroma collection...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    print(collection.peek()["embeddings"][0][:5])

    print(f"✅ Successfully saved vectorstore to {CHROMA_PATH}")
else:
    print("⚠️ No valid sections to add — nothing written to Chroma.")

# --- Step 7: Verify ---
# --- Step 7: Verify ---
print("\n[Verify] Fetching a sample from Chroma:")
sample = collection.get(limit=3, include=["embeddings", "metadatas", "documents"])

for i in range(len(sample["ids"])):
    sid = sample["ids"][i]
    meta = sample["metadatas"][i]
    doc = sample["documents"][i]
    emb = sample["embeddings"][i]

    if emb is None:
        emb_preview = "❌ Missing"
    else:
        emb_preview = emb[:5] if len(emb) > 0 else "❌ Empty"

    print(f" {i+1}. ID={sid} | Section={meta['section_number']} | "
          f"Content len={meta['content_length']} | Embedding preview={emb_preview}")

