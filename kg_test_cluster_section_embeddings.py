import json
import pandas as pd
import chromadb
import re
import numpy as np
from openai import OpenAI
from operator import itemgetter

from sentence_transformers import CrossEncoder
# --- CONFIG ---
CHROMA_CLUSTER_PATH = "vectorstore_clusters"    # Path where your cluster ChromaDB is stored
CHROMA_SECTION_PATH = "vectorstore_cluster_section"    # Path where your section ChromaDB is stored
CLUSTER_COLLECTION = "vectorstore_cluster_db"
SECTION_COLLECTION = "vectorstore_cluster_section_db"
TOP_CLUSTERS = 5 
TOP_SECTIONS = 10

# Initialize OpenAI client
client = OpenAI(api_key="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")

cluster_client = chromadb.PersistentClient(path=CHROMA_CLUSTER_PATH)
section_client = chromadb.PersistentClient(path=CHROMA_SECTION_PATH)

# --- Load collections ---
cluster_collection = cluster_client.get_or_create_collection(CLUSTER_COLLECTION)
section_collection = section_client.get_or_create_collection(SECTION_COLLECTION)


# --- Utility: Get embedding for user input ---
def get_text_embedding(text: str):
    """
    Uses OpenAI embeddings to convert user text to a vector.
    Replace model name or use your local embedding function if needed.
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


import re
import numpy as np

# Load frame_cluster_scores CSV once at the top
df_frames = pd.read_csv("kg_frame_cluster_scores.csv")

results = []   # will store the matched frames

# Load once
df_primary = pd.read_csv("cluster_primary_frames.csv")

def collect_primary_frame_for_cluster(cluster_id, top_n=5):
    # Fetch the row for this cluster
    row = df_primary[df_primary["cluster_id"] == cluster_id]

    if row.empty:
        print(f"No primary frames found for cluster {cluster_id}")
        return []

    frames_str = row.iloc[0]["primary_frames"]

    if not isinstance(frames_str, str) or frames_str.strip() == "":
        print(f"No primary frames listed for cluster {cluster_id}")
        return []

    # Split by comma → ["frame:global:tfidf", "frame:global:tfidf", ...]
    entries = [x.strip() for x in frames_str.split(",") if x.strip()]

    parsed = []

    for e in entries:
        parts = e.split(":")
        if len(parts) == 3:
            frame_name = parts[0].strip()
            global_score = float(parts[1].strip())
            tf_idf = float(parts[2].strip())

            parsed.append({
                "frame_name": frame_name,
                "global_score": global_score,
                "tf_idf": tf_idf
            })
        else:
            print("Skipping invalid entry:", e)

    # Sort HIGH GLOBAL SCORE FIRST (your instruction)
    parsed = sorted(parsed, key=lambda x: x["tf_idf"], reverse=True)

    # Print top N frames
    print(f"\nTop {top_n} primary frames for cluster {cluster_id}:")
    for item in parsed[:top_n]:
        print(f"  {item['frame_name']}  |  global={item['global_score']}  |  tfidf={item['tf_idf']}")

    return parsed[:top_n]

def query_get_section_children(all_section_matches, query_emb):
    """
    Finds child sections (like 10.47.4.*) for each matched parent section
    and ranks them by cosine similarity using embeddings from ChromaDB.
    """

    # --- Step 1: Fetch all section records with embeddings ---
    all_sections = section_collection.get(include=["embeddings", "metadatas", "documents"])

    child_scores = []
    if not all_sections or not all_sections.get("metadatas"):
        print("❌ Error: No metadata found in Chroma collection.")
        return

    embeddings = all_sections.get("embeddings")
    if embeddings is None or len(embeddings) == 0 or embeddings[0] is None:
        print("⚠️ Warning: No valid embeddings found or all empty.")
        return

    emb_count = len(embeddings)
    if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
      emb_dim = len(embeddings[0])
    else:
      emb_dim = 0

    print(f"✅ Embeddings present for {emb_count} records, each of dimension {emb_dim}.")

    metas = all_sections["metadatas"]
    embs = all_sections["embeddings"]
    docs = all_sections["documents"]

    # --- Step 2: Iterate through matched sections ---
    for section in all_section_matches:
        section_number = section["section_number"]

        print(f"\nParent Section: {section_number} | Distance: {section['distance']:.6f}")

        # --- Step 3: Identify children structurally ---
        pattern = re.compile(rf"^{re.escape(section_number)}\.\d+$")
        children = []

        for meta, emb, doc in zip(metas, embs, docs):
            if not meta or emb is None:
                continue
            sec_num = meta.get("section_number")
            if sec_num and pattern.match(sec_num):
                children.append((sec_num, np.array(emb, dtype=np.float32), doc))

        if not children:
            print("   ⚠️ No children found in Chroma for this section.")
            continue

        print(f"   🔹 Found {len(children)} child sections under {section_number}")

        # --- Step 4: Compute cosine similarity between query_emb and each child embedding ---
        query_vec = np.array(query_emb, dtype=np.float32)
        for sec_num, child_emb, doc in children:
            similarity = np.dot(query_vec, child_emb) / (
                np.linalg.norm(query_vec) * np.linalg.norm(child_emb)
            )
            child_scores.append((sec_num, doc, similarity))

        # --- Step 5: Sort and display results ---
    child_scores.sort(key=lambda x: x[1], reverse=True)
    print("   🧠 Child sections ranked by semantic similarity:")
#    for rank, (sec_num, doc, sim) in enumerate(child_scores[:10], 1):
#            print(f"     {rank:2}. {sec_num:<12} | Similarity: {sim:.6f}, | Content :{doc}")

# --- Query Pipeline ---
def query_and_rerank(user_query: str):
    print(f"\n🔍 User Query: {user_query}")

    # 1️⃣ Create embedding for query
    query_emb = get_text_embedding(user_query).tolist()

    #print("embedding", query_emb)

# --- Load collections ---
    print("\n🧩 DEBUG: Checking cluster Chroma DB contents...")
    # 2️⃣ Find top clusters
    print("\nStep 1: Searching top clusters...")

    cluster_results = cluster_collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_CLUSTERS
    )

#   print(cluster_results)

    cluster_matches = list(
        zip(
            cluster_results["ids"][0],
            cluster_results["distances"][0],
            cluster_results["metadatas"][0]
        )
    )

    for i, (cid, dist, meta) in enumerate(cluster_matches, 1):
        print(f"  {i}. Cluster ID: {meta['cluster_id']} | Distance: {dist:.6f}")

    # 3️⃣ For each top cluster, search within section embeddings
    print("\nStep 2: Searching sections inside top clusters...")
    all_section_matches = []

    primary_frame  = []

    for cid, _, meta in cluster_matches:
      cluster_id = meta["cluster_id"]

      p_frame = collect_primary_frame_for_cluster(cluster_id)
      primary_frame.extend(p_frame)
      
    # --- Step 1: Get cluster entry from cluster_collection ---
      result = cluster_collection.get(where={"cluster_id": cluster_id})

      if not result["metadatas"]:
        print(f"⚠️ Cluster {cluster_id} not found in Chroma DB.")
        continue

      cluster_meta = result["metadatas"][0]
      sections_json = cluster_meta.get("sections")
      if not sections_json:
        print(f"⚠️ Cluster {cluster_id} has no 'sections' metadata.")
        continue

      try:
        sections = json.loads(sections_json)
      except json.JSONDecodeError:
        print(f"❌ Invalid JSON in 'sections' for cluster {cluster_id}")
        continue

#      print(f"\n✅ Loaded cluster {cluster_id} with {len(sections)} sections from Chroma.")

    # --- Step 2: Compute similarity between query_emb and each section embedding ---
      query_vec = np.array(query_emb, dtype=np.float32)
      section_scores = []

      for sec_number, sec_emb in sections:
        sec_vec = np.array(sec_emb, dtype=np.float32)
        similarity = np.dot(query_vec, sec_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(sec_vec)
        )
        section_scores.append((sec_number, similarity))

    # --- Step 3: Sort sections by similarity descending ---
      section_scores.sort(key=lambda x: x[1], reverse=True)
      top_sections = section_scores[:TOP_SECTIONS]

    # --- Step 4: Collect & print matches ---
      for sec_number, sim in top_sections:
        all_section_matches.append({
            "cluster_id": cluster_id,
            "section_number": sec_number,
            "distance": 1 - sim  # optional: convert to distance if consistent with earlier logic
        })

      print(f" step 2  🧠 Top {len(top_sections)} section matches within cluster {cluster_id}:")
      for rank, (sec_num, sim) in enumerate(top_sections, 1):
        print(f"     {rank:2}. Section {sec_num:<12} | Similarity: {sim:.6f}")



    for section in all_section_matches:
        print(
            f"Cluster {section['cluster_id']} | "
            f"Section {section['section_number']} | "
            f"Distance: {section['distance']:.6f}"
        )


    query_get_section_children(all_section_matches, query_emb)

    print("Found primary frames", primary_frame)
    return 

if __name__ == "__main__":
      query = input("Enter your query: ")
      query_and_rerank(query)

