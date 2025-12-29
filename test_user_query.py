
import json
import chromadb
import os
from openai import OpenAI

# --- Configuration Constants (MUST match create_db.py) ---
# --- Configuration Constants ---
# Directory where ChromaDB will store the files
CHROMA_PATH = "vectorstore_clusters"

# Name of the collection (index) we are creating
COLLECTION_NAME = "vectorstore_clusters_index"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large" # Replace with your actual model if different

def get_openai_embedding(text, model=OPENAI_EMBEDDING_MODEL):
    """Calls the OpenAI API to generate a text embedding."""
    
    # Requires OPENAI_API_KEY environment variable to be set
    client = OpenAI(api_key="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")

    
    print(f"Generating embedding for query using '{model}'...")
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
        return None

def query_chroma_db(query_text, k=10):
    """Loads the persistent ChromaDB and executes a semantic search."""
    
    print("\n--- Starting ChromaDB Query (Part 2) ---")
    
    # 1. Initialize Chroma Client (Persistent Mode)
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Chroma database not found at '{CHROMA_PATH}'.")
        print("Please run 'create_db.py' first.")
        return
        
    client_db = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # 2. Get the Collection
    try:
        collection = client_db.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error getting collection '{COLLECTION_NAME}': {e}")
        return

    # 3. Generate Query Embedding
    query_emb = get_openai_embedding(query_text)
    if query_emb is None:
        return
        
    # 4. Query ChromaDB
    print(f"Searching for top {k} clusters...")
    
    # Chroma performs the vector similarity search (COSINE distance) automatically
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k, 
        include=['metadatas', 'distances']
    )

    # 5. Process and Display Results
    top_results = results['metadatas'][0]
    distances = results['distances'][0]

    print(f"\nQuery: '{query_text}'")
    print(f"Found {len(top_results)} results:")
    
    for i, (metadata, distance) in enumerate(zip(top_results, distances)):
        # Cosine Similarity = 1 - Cosine Distance
        similarity = 1 - distance
        
        # Parse the 'sections' string back into a Python list
        sections_list = json.loads(metadata['sections'])
        
        # Extract just the section numbers for display
        section_numbers = [s[0] for s in sections_list]
        
        print(f"--------------------------------------------------")
        print(f"Rank {i+1}:")
        print(f"  Cluster ID: {metadata['cluster_id']}")
        print(f"  Cosine Similarity: {similarity:.4f} (Closer to 1 is better)")
        print(f"  Relevant Sections: {section_numbers}")
        
    print("--------------------------------------------------")

user_query = input("Enter your search query (e.g., 'data analysis methodologies'): ")
query_chroma_db(user_query, k=10)

