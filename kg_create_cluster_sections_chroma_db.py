
import json
import os
import chromadb

# === CONFIG ===
KG_CLUSTERS = "kg_clusters.json"
CHROMA_PATH = os.environ["CHROMADB_FOR_CLUSTER_SECTION"]         # path where Chroma data is saved
COLLECTION_NAME = os.environ["CHROMADB_CLUSTER_SECTION_COLLECTION_NAME"]       # name of Chroma collection

def create_chroma_db_for_sections(filename=KG_CLUSTERS):
    """Reads kg_clusters.json, creates a persistent ChromaDB collection for section embeddings."""

    print("--- Starting ChromaDB Indexing (Section Embeddings) ---")

    # 1. Load your cluster JSON
    try:
        with open(filename, 'r') as f:
            clusters_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: '{filename}' not found. Please ensure it exists.")
        return

    # 2. Extract all section-level embeddings
    embeddings = []
    metadatas = []
    ids = []

    for cluster in clusters_data:
        cluster_id = cluster["cluster_id"]
        sections = cluster.get("sections", [])

        for idx, (section_number, section_emb) in enumerate(sections):
            section_id = f"cluster{cluster_id}_section{section_number}"
            ids.append(section_id)
            embeddings.append(section_emb)
            metadatas.append({
                "cluster_id": cluster_id,
                "section_number": section_number
            })

    print(f"✅ Extracted {len(embeddings)} section embeddings across {len(clusters_data)} clusters.")

    if not embeddings:
        print("⚠️ No section embeddings found in the JSON file. Aborting.")
        return

    # 3. Initialize persistent Chroma client
    client_db = chromadb.PersistentClient(path=CHROMA_PATH)

    # 4. Create or get the section collection
    print(f"Creating/getting collection '{COLLECTION_NAME}' at {CHROMA_PATH}...")
    collection = client_db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # use cosine similarity
    )

    # 5. Insert section embeddings + metadata
    print(f"Adding {len(embeddings)} section embeddings...")
    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"\n✅ Successfully indexed all section embeddings.")
    print(f"📁 Saved persistent ChromaDB at: {CHROMA_PATH}")
    print("You can now query this DB to search for sections (filtered by cluster_id).")


if __name__ == "__main__":
    create_chroma_db_for_sections()

