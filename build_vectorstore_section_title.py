import json
import re
import os
import getpass
import os
import json
import time

from sentence_transformers import CrossEncoder

import sys
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import bs4
from langchain import hub
from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

 
# Set the environment variables as requested
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY']="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"
os.environ['OPENAI_API_KEY']= "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"

# Optionally, you can verify that they were set by printing them
print("LANGCHAIN_TRACING_V2:", os.environ['LANGCHAIN_TRACING_V2'])
print("LANGCHAIN_API_KEY:", os.environ['LANGCHAIN_API_KEY'])
print("OPENAI_API_KEY:", os.environ['OPENAI_API_KEY'])

chunk_file = "80211-2020-chunks-trim.json"  # JSON file for saved chunks
feature_file = "kg_transform_feature_neo4j.json"  # JSON file for saved chunks

def get_section_prefix(section_number):
    """
    Returns the section prefix based on the following rules:
      - If section_number is a single part (e.g., "a"), return "a".
      - If section_number is two parts (e.g., "a.b"), return "a.b".
      - If section_number has more than two parts (e.g., "a.b.c", "a.b.c.d"), return "a.b".
    This function always returns a valid prefix string.
    """
    section_str = str(section_number).strip()  # ensure we have a clean string
    if not section_str:
        return ""  # return an empty string if section_number is empty

    parts = section_str.split(".")
    if len(parts) <= 3:
        return section_str  # Only one part, return it as is
    else:
        # Two or more parts: always return the first two parts joined by '.'
        return ".".join(parts[:3])

def normalize_section_number(sec_num: str) -> str:
    """
    Append '.' if section number has no dots.
    Examples:
        '1' -> '1.'
        '10' -> '10.'
        '1.2' -> '1.2' (unchanged)
        '10.2.4' -> '10.2.4' (unchanged)
    """
    sec_num = sec_num.strip()
    if "." not in sec_num and sec_num.isdigit():
        return sec_num + "."
    return sec_num

def build_vectorstore_from_chunks_json(feature_file, chunk_file, vectorstore_dir="chroma_db_features_section_title"):
    """
    Build or load Chroma vectorstore from a given chunk_file.
    Rebuild only if `chunks.json` is newer or vectorstore_dir doesn't exist.
    """
    if not os.path.exists(chunk_file):
        raise FileNotFoundError(f"{chunk_file} was not found. "
                                "Make sure you create it with check_chunks or supply a valid path.")

    documents = []

# Load feature.json
    with open(feature_file, "r", encoding="utf-8") as f:
      feature_data = json.load(f)

# Extract section numbers from nodes with label "Feature"
    feature_section_numbers = [
      node["name"] for node in feature_data.get("nodes", []) if node.get("label") == "Feature"
      ]

# Load chunks.json
    with open(chunk_file, "r", encoding="utf-8") as f:
      chunks_data = json.load(f)

# Build a lookup: section_number -> chunk

    chunk_lookup = {chunk.get("section_number"): chunk for chunk in chunks_data}

# Collect pairs of section_number and section_title
    section_pairs = []

# regex for X.Y format (digits dot digits)
    pattern = re.compile(r'^\d+(\.\d+)+$')
       
    for chunk in chunks_data:
      section_number = chunk.get("section_number")
      section_title =  chunk.get("section_title") or "NULL"

      if section_number:
        section_number = normalize_section_number(section_number)

      section_title = section_title.strip()

      if not section_number or not pattern.match(section_number):
        continue

      print("lookup", section_number)

      section_pairs.append((section_number, section_title))

      section_pairs.append((section_number, section_title))

      print(f"Section: {section_number}, Title: {section_title}")

      if section_title is None or not isinstance(section_title, str) or not section_title.strip():  # Fix NoneType
        section_number = "100.100"  # Default to empty string
        section_title = "NULL"
        print("section number null\n")

      section_prefix = get_section_prefix(section_number)

    # Embed section title separately
      if section_title:
        chunkid = int(chunk.get("chunkid"))
        doc_section_title = Document(
           page_content=section_title,  # Store section title as text for embedding
           metadata={"chunkid": chunkid, "section_title": section_title, "section_number": section_number, "section_prefix": section_prefix}  # Flag it
           )
        print(int(chunk["chunkid"]), section_number, section_prefix, section_title)
        documents.append(doc_section_title)

    # Initialize vectorstore with persist directory

    embedding = OpenAIEmbeddings()
    vectorstore = Chroma(embedding_function=embedding, persist_directory=vectorstore_dir)

# Break your documents list into manageable chunks
    doc_size = 20000  # or whatever fits well within token limits

    for i in range(0, len(documents), doc_size):
        doc_chunk = documents[i:i + doc_size]
        vectorstore.add_documents(doc_chunk)
        print(f"✅ Added chunk {i} to {i + doc_size}")

# Persist at the end
#    vectorstore = Chroma.from_documents(
#            documents=documents,
#            embedding=OpenAIEmbeddings(),
#            persist_directory=vectorstore_dir
#        )

    vectorstore.persist()
    print("✅ Vectorstore built and persisted.")

    return vectorstore

vectorstore_dir="./vectorstore_features_section_title"


vectorstore = build_vectorstore_from_chunks_json(feature_file, chunk_file, "chroma_db_features_section_title")


embedding_model = OpenAIEmbeddings()
query_embedding = embedding_model.embed_query("TWT")
retrieved_docs = vectorstore.similarity_search_by_vector(
                 query_embedding,
                 k=5)
for doc in retrieved_docs:
  print(f"Section: {doc.metadata['section_number']} ( {doc.metadata['section_title']} Chunk ID: {doc.metadata['chunkid']})")

