import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv

from openai import OpenAI
load_dotenv()
client = OpenAI()

# Load environment variables (ensure your .env file has OPENAI_API_KEY)

# Directory where transcript files are stored
transcripts_folder = "transc"

def get_embedding(text, model="text-embedding-3-small"):
    """
    Get the embedding vector for the given text using OpenAI's embeddings API.
    The text is cleaned by replacing newlines with spaces.
    """
    cleaned_text = text.replace("\n", " ")
    response = client.embeddings.create(input=[cleaned_text], model=model)
    # Extract the embedding vector from the API response
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

def split_text(text, parts=2):
    """
    Split the text into the specified number of parts.
    This simple method splits by character length.
    """
    chunk_size = len(text) // parts
    chunks = []
    for i in range(parts):
        start = i * chunk_size
        # For the last part, take the remainder of the text
        if i == parts - 1:
            chunk = text[start:]
        else:
            chunk = text[start:start + chunk_size]
        chunks.append(chunk)
    return chunks

# Load transcript documents from the folder and split each into two parts
doc_chunks = []  # List to hold each transcript chunk with metadata
for filename in os.listdir(transcripts_folder):
    if filename.endswith("_transcript.txt"):
        filepath = os.path.join(transcripts_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Split the transcript text into two parts
        parts = split_text(content, parts=2)
        for i, part in enumerate(parts):
            doc_chunks.append({
                "filename": filename,
                "chunk_index": i + 1,  # 1-indexed (Part 1, Part 2)
                "text": part
            })

print(f"Loaded and split {len(doc_chunks)} transcript chunks from {len(os.listdir(transcripts_folder))} files.")

# Generate embeddings for each transcript chunk
embeddings = []
for doc in doc_chunks:
    print(f"Embedding {doc['filename']} - Part {doc['chunk_index']}")
    emb = get_embedding(doc["text"], model="text-embedding-3-small")
    embeddings.append(emb)
embeddings = np.array(embeddings)

# Normalize embeddings for cosine similarity (dot product will then be equivalent)
faiss.normalize_L2(embeddings)

# Create a FAISS index using Inner Product (IP)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# Save the FAISS index to disk
faiss.write_index(index, "transcripts.index")
print("FAISS index saved to transcripts.index.")

# Save the document metadata (mapping each vector to its source transcript chunk)
with open("transcripts_metadata.json", "w", encoding="utf-8") as f:
    json.dump(doc_chunks, f, ensure_ascii=False, indent=2)
print("Metadata saved to transcripts_metadata.json.")
