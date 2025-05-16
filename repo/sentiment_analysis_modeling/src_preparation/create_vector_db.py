import os
import pathlib
import itertools
import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# --- 1. Model and Chroma Collection ---
# Load embedding model from local directory
EMBEDDING_MODEL_DIR = str(pathlib.Path(__file__).parent.parent / "model" / "embedding_model")
print(f"Loading embedding model from: {EMBEDDING_MODEL_DIR}")
embedder = SentenceTransformer(EMBEDDING_MODEL_DIR, device="cpu")

CHROMA_DIR = os.path.join("..", "data", "vectorial_database")
CHROMA_DIR = str(pathlib.Path(__file__).parent.parent / "data" / "vectorial_database")
os.makedirs(CHROMA_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_DIR)

COLLECTION_NAME = "reviews"
if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
    collection = client.create_collection(COLLECTION_NAME)
else:
    collection = client.get_collection(COLLECTION_NAME)

# --- 2. Load Data ---
TRAIN_PATH = pathlib.Path(__file__).parent.parent / "data" / "raw" / "train.parquet"
df = pd.read_parquet(TRAIN_PATH)

assert "text" in df.columns and "label" in df.columns, "train.parquet must have 'text' and 'label' columns."

# --- 3. Batch Insert into Chroma ---
BATCH_SIZE = 256
reviews = df["text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()

for batch_idx, start in enumerate(tqdm.tqdm(range(0, len(reviews), BATCH_SIZE), desc="Embedding + insert")):
    batch_reviews = reviews[start:start+BATCH_SIZE]
    batch_labels = labels[start:start+BATCH_SIZE]

    # Embeddings
    embs = embedder.encode(
        batch_reviews,
        batch_size=64,
        normalize_embeddings=True
    ).astype("float32")

    # Prepare metadata
    metas = [{"label": label} for label in batch_labels]

    # Insert into Chroma
    collection.add(
        documents=batch_reviews,
        embeddings=embs.tolist(),
        ids=[f"{batch_idx}_{i}" for i in range(len(batch_reviews))],
        metadatas=metas
    )

print("Vectorial database created at data/vectorial_database with original review, embedding, and label.")
