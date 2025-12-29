
import os
import json
from langchain_community.vectorstores import Chroma

# === CONFIG ===
CLUSTERS_FILE = "kg_clusters.json"
VECTORSTORE_DIR = "./vectorstore_clusters"
VECTORSTORE_CLUSTER_DIR = os.path.join(VECTORSTORE_DIR, "cluster_embeddings")
VECTORSTORE_SECTION_DIR = os.path.join(VECTORSTORE_DIR, "section_embeddings")

def build_vectorstore_from_clusters_json(cluster_file: str, vectorstore_dir: str):
    if not os.path.exists(cluster_file):
        raise FileNotFoundError(
            f"{cluster_file} not found. Ensure your clustering pipeline has generated it."
        )

    with open(cluster_file, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    print(f"✅ Loaded {len(clusters_data)} clusters from {cluster_file}")

    # --- Ensure directories exist ---
    os.makedirs(VECTORSTORE_CLUSTER_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_SECTION_DIR, exist_ok=True)

    # --- Initialize Chroma vectorstores ---
    vectorstore_cluster = Chroma(
        persist_directory=VECTORSTORE_CLUSTER_DIR
    )
    vectorstore_section = Chroma(
        persist_directory=VECTORSTORE_SECTION_DIR
    )

    # --- Prepare cluster-level data ---
    cluster_embeddings = []
    cluster_metadatas = []
    cluster_texts = []

    section_embeddings = []
    section_metadatas = []
    section_texts = []

    for cluster in clusters_data:
        cid = cluster["cluster_id"]
        cluster_emb = cluster["cluster_embedding"]

        # Add cluster-level entry
        if cluster_emb:
            cluster_embeddings.append(cluster_emb)
            cluster_texts.append(f"Cluster {cid}")
            cluster_metadatas.append({"cluster_id": cid})

        # Add section-level entries
        for section_id, section_emb in cluster.get("sections", []):
            section_embeddings.append(section_emb)
            section_texts.append(f"Section {section_id} of cluster {cid}")
            section_metadatas.append({
                "cluster_id": cid,
                "section_id": section_id
            })

    print(f"📦 Prepared {len(cluster_embeddings)} cluster vectors and {len(section_embeddings)} section vectors")

    # --- Insert embeddings directly into Chroma ---
    # NOTE: add_embeddings() expects list of texts, list of embeddings, list of metadatas
    if cluster_embeddings:
        vectorstore_cluster.add_embeddings(
            texts=cluster_texts,
            embeddings=cluster_embeddings,
            metadatas=cluster_metadatas
        )
        print(f"✅ Added {len(cluster_embeddings)} cluster embeddings")

    if section_embeddings:
        vectorstore_section.add_embeddings(
            texts=section_texts,
            embeddings=section_embeddings,
            metadatas=section_metadatas
        )
        print(f"✅ Added {len(section_embeddings)} section embeddings")

    # --- Persist both vectorstores ---
    vectorstore_cluster.persist()
    vectorstore_section.persist()

    print("\n✅ Vectorstores successfully created:")
    print(f" - Cluster embeddings: {VECTORSTORE_CLUSTER_DIR}")
    print(f" - Section embeddings: {VECTORSTORE_SECTION_DIR}")

    return vectorstore_cluster, vectorstore_section


if __name__ == "__main__":
    build_vectorstore_from_clusters_json(CLUSTERS_FILE, VECTORSTORE_DIR)

