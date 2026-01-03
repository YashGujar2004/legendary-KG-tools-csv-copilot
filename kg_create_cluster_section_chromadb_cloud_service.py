
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import re
import os
import json
import numpy as np
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import chromadb

# --- Config ---
KG_CLUSTERS = os.environ.get("KG_CLUSTERS")

CHROMADB_TENANT_ID = os.environ["CHROMADB_TENANT_ID"]# your tenant ID
CHROMADB_API_KEY = os.environ["CHROMADB_API_KEY"]  # your API key

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

EMBEDDING_CACHE_FILE= os.environ.get("EMBEDDING_CACHE_FILE")
CHUNKS_FILE = os.environ.get("SPEC_CHUNKS_TRIM")

COLLECTION_NAME = os.environ.get("CHROMADB_CLUSTER_SECTION_COLLECTION_NAME")
CHROMADB_CLOUD_DATABASE_NAME = os.environ.get("CHROMADB_CLOUD_DATABASE_NAME")

# adjust batch size as needed (KEEP 1 ONLY)
BATCH_SIZE = 1

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
print("Connecting to Chroma Cloud...")

try:
    client = chromadb.CloudClient(
        tenant=CHROMADB_TENANT_ID,
        database=CHROMADB_CLOUD_DATABASE_NAME,
        api_key=CHROMADB_API_KEY
        )
    client.heartbeat() 
    print("Connection successful (heartbeat OK).")
    
except Exception as e:
        print(f"❗ FAILED TO CONNECT: {e}")
        raise e
    
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    #     embedding_function=OpenAIEmbeddingFunction(
    #     api_key=os.environ.get("OPENAI_API_KEY"),
    #     model_name="text-embedding-3-large"
    # )
    # metadata={"hnsw:space": "cosine"}
)

# --- Step 5: Prepare data for insertion ---
ids, embeddings, metadatas, documents = [], [], [], []

# seen_ids = {}
# Sets are much faster for lookups than lists.
added_section_numbers = set()
duplicates_count = 0

for idx, entry in enumerate(chunks_data):
    sec_raw = entry.get("section_number", None)
    if sec_raw in missing_embeddings or missing_contents:
        print(f"[Warning] Skipping entry {idx}: missing embedding for section_number {sec_raw}")
        continue
    
    if not sec_raw:
        print(f"[Warning] Skipping entry {idx}: missing section_number")
        continue

    sec = str(sec_raw).strip(".")
    if sec not in valid_sections:
        continue
    
    if sec in added_section_numbers:
            duplicates_count += 1
            print(f"  [Duplicate] Skipping extra chunk for section {sec}")
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

    # base_id = sec_raw
    # if base_id in seen_ids:
    #     seen_ids[base_id] += 1
    #     unique_id = f"{base_id}_{seen_ids[base_id]}"
    # else: 
    #     seen_ids[base_id] = 0
    #     unique_id = base_id
    
# Mark this section as added so duplicates are skipped next time
    added_section_numbers.add(sec)    

    ids.append(sec)
    embeddings.append(emb.tolist())
    documents.append(sec_content)
    metadatas.append({
        "section_number": sec,
        "content_length": len(sec_content)
    })

print(f"\n[Summary] Prepared {len(embeddings)} valid sections for vectorstore.")
print(f"          Skipped {duplicates_count} duplicates.")
print(f"          (Expect match with: {len(valid_sections)})")
# --- Step 6: Add to ChromaDB ---
if embeddings:
    
    total = len(ids)    
    print(f"Uploading {total} cluster entries in batches of {BATCH_SIZE}...")
    
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        
        print(f"  Uploading batch {i} to {end-1}...")
        
        # Get the slice for the current batch
        batch_ids = ids[i:end]
        batch_embeddings = embeddings[i:end]
        batch_documents = documents[i:end]
        batch_metadatas = metadatas[i:end]

        try:
            # Upsert this batch
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )
        except Exception as e:
            print(f"  ❗ ERROR uploading batch {i}-{end-1}: {e}", "with ID: ", batch_ids[0])
            print("     Skipping this batch and continuing...")
            # Optionally, you could re-raise the error to stop the script
    
    print(f"\nSuccessfully uploaded all cluster data to Chroma Cloud collection: '{COLLECTION_NAME}'")
else:
    print("⚠️ No valid sections to add — nothing written to Chroma.")

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

