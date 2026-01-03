import numpy as np
from sklearn.cluster import AgglomerativeClustering
import re
import json
import numpy as np
import chromadb
import os


# turn on/off cloud db creation
CHROMADB_CLOUD_SERVICE_MODE_ENV = os.environ.get("CHROMADB_CLOUD_SERVICE_MODE")

CHROMADB_CLOUD_SERVICE_MODE = CHROMADB_CLOUD_SERVICE_MODE_ENV == "true" 

# keep FALSE for local Chroma creation 

# --- Config for LOCAL CHROMA ---
KG_CLUSTERS = os.environ.get("KG_CLUSTERS")
CHROMA_PATH = os.environ.get("CHROMADB_FOR_CLUSTERS")
EMBEDDING_CACHE_FILE = os.environ.get("EMBEDDING_CACHE_FILE")
CHUNKS_FILE = os.environ.get("SPEC_CHUNKS_TRIM")

# --- Config for CHROMA CLOUD ---
CHROMADB_TENANT_ID = os.environ["CHROMADB_TENANT_ID"]  # your tenant ID
CHROMADB_API_KEY = os.environ["CHROMADB_API_KEY"]  # your API key
CHROMADB_CLOUD_DATABASE_NAME = os.environ["CHROMADB_CLOUD_DATABASE_NAME"]
COLLECTION_NAME = os.environ["CHROMADB_CLUSTER_COLLECTION_NAME"]

CHROMADB_BATCH_SIZE = int(
    os.environ.get("CHROMADB_BATCH_SIZE")
)

# Ensure the Chroma directory exists
os.makedirs(CHROMA_PATH, exist_ok=True)

def cosine_similarity_matrix(X):
    """Compute cosine similarity matrix."""
    X = np.array(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.dot(X, X.T) / (norms * norms.T + 1e-10)

def verify_chroma_db():

# --- Connect to Chroma ---
  client = chromadb.PersistentClient(path=CHROMA_PATH)
  collection = client.get_or_create_collection(COLLECTION_NAME)

# --- Target section number ---
  target_section = "10.47"

# --- Fetch all entries ---
  print("📦 Reading Chroma section collection...")

# You can fetch in chunks if DB is large; here we load all for simplicity
  data = collection.get()

  target_section = "10.47"
  found = False

  for id_val, meta in zip(data["ids"], data["metadatas"]):
    sections_json = meta.get("sections")
    if not sections_json:
        continue

    try:
        sections = json.loads(sections_json)
    except json.JSONDecodeError:
        print(f"⚠️ Invalid sections JSON in entry {id_val}")
        continue

    # Each section looks like ["9.4.1.11", [embedding]]
    for section_id, _ in sections:
        if str(section_id) == target_section:
            print(f"✅ Section {target_section} found in cluster {meta.get('cluster_id')}")
            print(f"   Entry ID: {id_val}")
            found = True
            break

    if found:
        break  # stop once found

  if not found:
    print(f"❌ Section {target_section} not found in Chroma DB.")


def create_chroma_db_local(filename=KG_CLUSTERS):

    """Reads clusters.json, creates a persistent ChromaDB collection, and saves it."""
    
    print("--- Starting ChromaDB Indexing (Part 1) ---")
    
    # 1. Load your data
    try:
        with open(filename, 'r') as f:
            clusters_data = json.load(f)
    except FileNotFoundError:
        print("Error: 'clusters.json' not found. Please ensure it is in the same directory.")
        return

    # 2. Format Data for Chroma
    ids = [f"cluster_{entry['cluster_id']}" for entry in clusters_data]

    embeddings = [entry['cluster_embedding'] for entry in clusters_data]
    
    # Store the original data (including sections) in metadata, 
    # converting complex lists/objects to JSON strings for storage.
    metadatas = [
        {
            "cluster_id": entry['cluster_id'], 
            # Dumps the complex 'sections' list into a string
            "sections": json.dumps(entry['sections']) 
        } 
        for entry in clusters_data
    ]

    ''' 
    target_section = "10.47"

    for meta in metadatas:
      cluster_id = meta["cluster_id"]
      try:
        # Parse the JSON string back into a Python list
        sections = json.loads(meta["sections"])
      except json.JSONDecodeError:
        print(f"⚠️ Invalid JSON in cluster {cluster_id}, skipping.")
        continue

    # Each section entry is like ["9.4.1.11", [embedding]]
      for section_id, _ in sections:
        if section_id == target_section:
            print(f"✅ Found section {target_section} in Cluster ID: {cluster_id}")
    '''
    
    # 3. Initialize Chroma Client (Persistent Mode)
    # This client will automatically save data to the CHROMA_PATH directory
    client_db = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # 4. Create or Get Collection
    # Set hnsw:space to 'cosine' to ensure similarity search uses Cosine Distance
    print(f"Creating/getting collection '{COLLECTION_NAME}'...")
    collection = client_db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Insert the Data
    print(f"Adding {len(embeddings)} cluster entries...")
    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"\nSuccessfully indexed all data and saved to disk at: {CHROMA_PATH}")
    print("Run 'query_db.py' now to test the search.")


def save_clusters_to_json(clusters, filename=KG_CLUSTERS):
    # Convert numpy arrays to lists
    serializable_clusters = []
    for cluster in clusters:
        serializable_cluster = {
            "cluster_id": cluster["cluster_id"],
            "cluster_embedding": cluster["cluster_embedding"].tolist()
              if isinstance(cluster["cluster_embedding"], np.ndarray) else cluster["cluster_embedding"],
            "sections": [
                (section_number, emb.tolist() if isinstance(emb, np.ndarray) else emb)
                for section_number, emb in cluster["sections"]
            ]
        }
        serializable_clusters.append(serializable_cluster)

    # Save as JSON
    with open(filename, "w") as f:
        json.dump(serializable_clusters, f, indent=2)

    print(f"✅ Clusters saved to {filename}")


def append_features_and_refine_clusters(clusters, feature_sections, embeddings_cache):
    """
    Append feature sections (parents) with embeddings to each cluster,
    remove redundant children, compute cluster embeddings,
    and retain original cluster structure format (list of list of tuples),
    but with added cluster metadata.

    Returns:
        enriched_clusters (list): each cluster is a dict:
            {
              "cluster_id": int,
              "cluster_embedding": np.ndarray,
              "sections": [(section_number, embedding), ...]
            }
    """

    enriched_clusters = []

    for cluster_idx, cluster in enumerate(clusters):
        # --- Step 1: Convert to a set of section numbers for manipulation ---
        cluster_sections = {sec for sec, _ in cluster}

        # --- Step 2: Append feature sections (parents) ---
        for parent, descs in feature_sections.items():
            if any(d in cluster_sections for d in descs):
                cluster_sections.add(parent)

        # --- Step 3: Remove descendants if parent exists in the same cluster ---
        pruned_sections = set(cluster_sections)
        for sec in cluster_sections:
            for other in cluster_sections:
                if other != sec and other.startswith(sec + "."):
                    pruned_sections.discard(other)

        # --- Step 4: Rebuild cluster as list of (section_number, embedding) tuples ---
        refined_cluster = []
        for s in sorted(pruned_sections):
            if s in embeddings_cache:
                emb = np.array(embeddings_cache[s])
                refined_cluster.append((s, emb))

        # --- Step 5: Compute cluster embedding (mean of all section embeddings) ---

        if refined_cluster:
            embs = np.array([emb for _, emb in refined_cluster])
            cluster_emb = np.mean(embs, axis=0)
        else:
            cluster_emb = None

        # --- Step 6: Maintain same shape but add metadata ---
        enriched_clusters.append({
            "cluster_id": cluster_idx,
            "cluster_embedding": cluster_emb,
            "sections": refined_cluster
        })

    return enriched_clusters


def compute_cluster_cohesion(embeddings, threshold=0.65, var_cutoff=0.02):
    """
    Perform cluster-based cohesion analysis among siblings.
    Uses agglomerative clustering + variance-based outlier rejection.
    """
    if len(embeddings) < 2:
        return True, {"reason": "single_item"}

    # --- Cosine similarity and distances ---
    sim = cosine_similarity_matrix(embeddings)
    dist = 1 - sim

    # --- Cluster siblings ---
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=(1 - threshold)
    )
    labels = clustering.fit_predict(dist)

    # --- Analyze cluster structure ---
    unique, counts = np.unique(labels, return_counts=True)
    largest_cluster_frac = np.max(counts) / len(embeddings)
    n_clusters = len(unique)

    # --- Variance filter for cluster compactness ---
    sim_vals = sim[np.triu_indices(len(sim), 1)]
    sim_var = np.var(sim_vals)
    sim_mean = np.mean(sim_vals)

    #  (sim_mean >= threshold) and
    cohesive = (
        (largest_cluster_frac >= 0.4) and
        (sim_var <= var_cutoff)
    )


    return cohesive, {
        "n_clusters": n_clusters,
        "largest_cluster_frac": largest_cluster_frac,
        "mean_sim": sim_mean,
        "var_sim": sim_var
    }


def get_parent_section(section_number):
    """Return parent by removing last segment (e.g. 10.4.2.3 → 10.4.2)."""
    parts = section_number.split('.')
    if len(parts) > 2:
        return '.'.join(parts[:-1])
    return None


def is_top_level_section(section_number):
    """Limit hierarchy climb to X.Y level (1 decimal precision)."""
    return len(section_number.split('.')) <= 1 


def find_cohesive_parents_from_clusters(clusters, json_data, embeddings_cache):
    """
    Given precomputed section clusters, identify top-most cohesive parents
    based on sibling clustering, variance filtering, and hierarchical climbing.
    """

    # --- Build lookup from section_number → record ---
    data_map = {d["section_number"]: d for d in json_data}

    feature_sections = {}
    checked_parents = set()

    for cluster_idx, cluster in enumerate(clusters):
        #print(f"\n🔹 [Cluster {cluster_idx}] Processing {len(cluster)} sections")

        for section_number, _ in cluster:
            if section_number not in data_map:
                continue

            current = section_number
            last_cohesive_parent = None

            # climb upwards until non-cohesive or top level (X.Y)
            while current and not is_top_level_section(current):
                parent = get_parent_section(current)
                if not parent or parent in checked_parents:
                    break

                # --- Find siblings of parent of the current ---
                siblings = [
                    k for k in data_map.keys()
                    if isinstance(k, str) and re.match(rf"^{re.escape(parent)}\.\d+$", k)
                ]
                if len(siblings) < 2:
                    break

                # --- Get sibling embeddings ---
                sibling_embs = []
                for s in siblings:
                    emb = embeddings_cache.get(s)
                    if emb is not None:
                        sibling_embs.append(emb)

                if len(sibling_embs) < 2:
                    break

                # --- Check cohesion of sibling set ---
                cohesive, stats = compute_cluster_cohesion(sibling_embs)
#                print(f"   [Check] Parent {parent} of Section {current} : "
    #                  f"mean={stats['mean_sim']:.3f}, var={stats['var_sim']:.3f}, "
    #                  f"clusters={stats['n_clusters']}, frac={stats['largest_cluster_frac']:.2f} "
    #                  f"→ {'✅' if cohesive else '❌'}")

                if cohesive:
                    last_cohesive_parent = parent
                    checked_parents.add(parent)
                    current = parent  # climb up
                else:
                    break  # stop climbing when cohesion fails

            # --- Mark top-most cohesive parent ---
            if last_cohesive_parent:
                feature_sections.setdefault(last_cohesive_parent, set()).add(section_number)


    # --- Convert descendant sets to lists ---
    feature_sections = {k: list(v) for k, v in feature_sections.items()}

    print("\n✅ [Summary] Feature Sections Identified:")
#    for k, v in feature_sections.items():
#        print(f"   {k}: {len(v)} descendants")

    # --- Compute feature embeddings (bottom-up avg of descendants) ---
    feature_embeddings = {}

    # Prefer the parent’s own embedding (already computed as mean of descendants)
    for parent, descs in feature_sections.items():
      if parent in embeddings_cache:
        feature_embeddings[parent] = np.array(embeddings_cache[parent])
      else:
        # Fallback: average available descendant embeddings
        print("fallback\n")
        embs = [np.array(embeddings_cache[d]) for d in descs if d in embeddings_cache]
        if embs:
            feature_embeddings[parent] = np.mean(np.vstack(embs), axis=0)

    return feature_sections, feature_embeddings


def cluster_all_sections_from_embeddings(all_section_embeddings, threshold=0.275):
    """
    Clusters all sections based on their embeddings using Agglomerative clustering.
    Returns clusters as lists of (section_number, embedding) tuples.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np
    import pandas as pd

    sections = sorted(all_section_embeddings.keys())
    embeddings = np.array([all_section_embeddings[s] for s in sections])

    print(f"[Info] Clustering {len(sections)} sections...")

    # --- Step 1: Compute cosine similarity matrix and convert to distance ---
    sim_matrix = cosine_similarity(embeddings, embeddings)
    dist_matrix = 1 - sim_matrix

    # --- Step 2: Perform Agglomerative clustering ---
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold
    )
    labels = clustering.fit_predict(dist_matrix)

    # --- Step 3: Convert to cluster structure ---
    clusters_df = pd.DataFrame({"Section": sections, "Cluster": labels}).sort_values("Cluster")
    print(f"[Info] Formed {clusters_df['Cluster'].nunique()} clusters.")

    # --- Step 4: Group sections by cluster ---
    clusters = []
    for cluster_id in sorted(clusters_df["Cluster"].unique()):
        cluster_sections = clusters_df[clusters_df["Cluster"] == cluster_id]["Section"].tolist()
        cluster_data = [(s, all_section_embeddings[s]) for s in cluster_sections if s in all_section_embeddings]
        clusters.append(cluster_data)

    print(f"[Info] Prepared {len(clusters)} clusters for cohesion analysis.")
    return clusters


# --- CHROMA CLOUD PIPELINE ---

def connect_chroma_cloud(tenant_id: str, api_key: str, database: str) -> chromadb.Client:
    
    print("Connecting to Chroma Cloud for Clusters Upload...")

    try:
        client = chromadb.CloudClient(
            tenant=tenant_id,
            database=database,
            api_key=api_key
        )
        client.heartbeat() 
        print("Connection successful (heartbeat OK).")
        return client
    
    except Exception as e:
        print(f"❗ FAILED TO CONNECT: {e}")
        raise e

def create_chroma_db_cloud(filename=KG_CLUSTERS):
    """
    Reads clusters.json, creates a Chroma Cloud collection, 
    and uploads the cluster centers.
    """
    
    print("--- Starting to Load Data ---")
    
    # 1. Load your data 
    try:
        with open(filename, 'r') as f:
            clusters_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return

    # 2. Format Data for Chroma (Added 'documents' list)
    print(f"Formatting data for {len(clusters_data)} clusters...")
    
    # IDs must be strings
    ids = [f"cluster_{entry['cluster_id']}" for entry in clusters_data]
    
    embeddings = [entry['cluster_embedding'] for entry in clusters_data]
    
    # We must provide a 'documents' list. We can use a simple placeholder.
    documents = [
        f"Centroid for cluster {entry['cluster_id']}" 
        for entry in clusters_data
    ]

    # metadatas = [
    #     {
    #         "cluster_id": entry['cluster_id'],  
    #         "sections": json.dumps(entry['sections']) # Store sections and embeddings as a string
    #     } 
    #     for entry in clusters_data
    # ]
    metadatas = []
    for entry in clusters_data:
        # NEW: Extract *only* the section numbers (the first item in each pair)
        # This strips out the large embeddings that were causing the error.
        try:
            section_numbers = [sec[0] for sec in entry.get('sections', [])]
        except (TypeError, IndexError):
            section_numbers = [] # Handle bad data if 'sections' isn't a list of lists

        meta = {
            "cluster_id": entry['cluster_id'], 
            # FIX: Now we only store the *list of section numbers* as a string.
            # This will be much, much smaller than 4KB.
            "sections": json.dumps(section_numbers) 
        }
        metadatas.append(meta)
        
    print("Metadata :")
    print(metadatas[0])  # Print first metadata entry for verification

    # 3. Initialize Chroma Cloud Client (MODIFIED)
    # This now calls our helper function instead of PersistentClient
    client_db = connect_chroma_cloud(CHROMADB_TENANT_ID, CHROMADB_API_KEY, CHROMADB_CLOUD_DATABASE_NAME)
    
    # 4. Create or Get Collection (Unchanged, but now on the cloud)
    print(f"Creating/getting cloud collection '{COLLECTION_NAME}'...")
    collection = client_db.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=None
        # metadata={"hnsw:space": "cosine"} # Still good to set this
    )

    # 5. Insert the Data (Using 'upsert' for cloud)
    print(f"Uploading {len(ids)} cluster entries to the cloud...")
    
    # Use 'upsert' - it's robust and will add or update as needed.
    total = len(ids)
    print(f"Uploading {total} cluster entries in batches of {CHROMADB_BATCH_SIZE}...")
    
    for i in range(0, total, CHROMADB_BATCH_SIZE):
        end = min(i + CHROMADB_BATCH_SIZE, total)
        
        print(f"  Uploading batch {i} to {end-1}...")
        
        # Get the slice for the current batch
        batch_ids = ids[i:end]
        batch_embeddings = embeddings[i:end]
        batch_documents = documents[i:end]
        batch_metadatas = metadatas[i:end]

        try:
            # Upsert this batch
            collection.upsert(
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        except Exception as e:
            print(f"  ❗ ERROR uploading batch {i}-{end-1}: {e}")
            print("     Skipping this batch and continuing...")
            # Optionally, you could re-raise the error to stop the script
    
    print(f"\nSuccessfully uploaded all cluster data to Chroma Cloud collection: '{COLLECTION_NAME}'")


# LOCAL CHROMA DB CREATION PIPELINE

def run_local_pipeline():
    
    print("\n--- Starting Local Data Generation Pipeline ---")
    
    #1️⃣ LOAD DATA + EMBEDDINGS
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    print(f"[Info] Loaded {len(json_data)} sections from JSON")

    with open(EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
        embeddings_cache = json.load(f)
    print(f"[Info] Loaded {len(embeddings_cache)} embeddings")

    #  2️⃣ CLUSTER ALL SECTIONS
    clusters = cluster_all_sections_from_embeddings(embeddings_cache, threshold=0.275)

    print(f"[Info] Generated {len(clusters)} clusters from section embeddings")

    #  3️⃣ FIND COHESIVE PARENTS 
    feature_sections, feature_embeddings = find_cohesive_parents_from_clusters(
        clusters,
        json_data,
        embeddings_cache
    )

    clusters = append_features_and_refine_clusters(clusters, feature_sections, embeddings_cache)

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        section_list = cluster["sections"]
#        print(f"cluster id {cluster_id}")
        
    #for section_number, embedding in section_list:
    #    print(f"section: {section_number}")   

    '''

    # feature_sections: { parent_section_number: [descendant_sections] }
    # feature_embeddings: { parent_section_number: averaged_embedding }

    print(f"[Result] Found {len(feature_sections)} cohesive parent sections.")
#    for parent, descs in feature_sections.items():
#        print(f"  - {parent}: {len(descs)} descendants")

    '''
    save_clusters_to_json(clusters)
    print(f"✅ Local pipeline complete. Saved to {KG_CLUSTERS}")
    return True

    #verify_chroma_db()
    '''
    # ============================================================
    # (OPTIONAL) EXPORT RESULTS
    # ============================================================
    import json
    with open("kg_cohesive_feature_sections.json", "w", encoding="utf-8") as f:
        json.dump(feature_sections, f, indent=2)
    print("[Saved] cohesive_feature_sections.json")

    with open("kg_cohesive_feature_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(feature_embeddings, f, default=lambda o: o.tolist())
    print("[Saved] cohesive_feature_embeddings.json")
    '''


# ============================================================================================== 

def main():
    print("Creating kg_cluster.json file...") 

    run_local_pipeline()
    
    if CHROMADB_CLOUD_SERVICE_MODE:
        print("\n--- CHROMADB_CLOUD_SERVICE: TRUE (Cloud Mode) ---")
        create_chroma_db_cloud()
    else:
        print("\n--- CHROMADB_CLOUD_SERVICE: FALSE (Local Mode) ---")
        create_chroma_db_local()
    
if __name__ == "__main__":
    main()
