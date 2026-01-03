
import json
import os
import getpass
import os
import json
import time
from sentence_transformers import CrossEncoder
import sys
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set the environment variables as requested
os.environ['LANGCHAIN_TRACING_V2'] = "true"
LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

BATCH_SIZE = 100   # safe batch size for embeddings

chunk_file = "80211-2020-chunks-trim.json"  # JSON file for saved chunks

# 1. Setup embedding + vectorstore
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")


# 2. Split a large chunk into sub-chunks
def split_chunk(chunk, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    sub_chunks = splitter.split_text(chunk["content"])
    docs = []
    for idx, sub in enumerate(sub_chunks):
        print("sec ", chunk["section_number"])
        docs.append(Document(
            page_content=sub,
            metadata={
                "chunkid": chunk["chunkid"],
                "section_number": chunk["section_number"],
                "section_title": chunk["section_title"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "index": idx,
            }
        ))
    return docs

def make_title_metadata(chunk):
    return {
        "chunkid": chunk["chunkid"],
        "section_number": chunk["section_number"],
        "section_title": chunk["section_title"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "is_parent": True
    }

def build_vectorstore_from_chunks_json(chunk_file, vectorstore_dir):

  if not os.path.exists(chunk_file):
    raise FileNotFoundError(f"{chunk_file} was not found. "
                  "Make sure you create it with check_chunks or supply a valid path.")
  with open(chunk_file, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

  print("chunks", chunks_data[19])
  content_docs = []

  vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./vectorstore_contents" 
  )

 #title = make_title_metadata(chunk)  # e.g., {'parent_id': ..., 'section_number': ...}

# 4. Build docs: children + parent

  for chunk in chunks_data:
    # split_chunk returns list of strings
    content = split_chunk(chunk)
    # attach parent metadata to each child
    content_docs.extend(content)   # children

   # Batch embed + add children
  for i in range(0, len(content_docs), BATCH_SIZE):
        batch = content_docs[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"✅ Added content docs {i} to {i+len(batch)}")

  #vectorstore.persist()



vectorstore = build_vectorstore_from_chunks_json(chunk_file, "vectorstore_contents")


