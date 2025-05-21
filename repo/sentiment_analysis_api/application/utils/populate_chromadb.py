#!/usr/bin/env python
import os
import sys
import pathlib
import time

# Añadir directorio raíz al path para importar desde application
root_dir = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# Importamos initialize_models
from application.services.sentiment_service import initialize_models
from sentence_transformers import SentenceTransformer
import chromadb

# Definir datos de ejemplo
SAMPLE_REVIEWS = [
    {
        "text": "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged from start to finish.",
        "label": "1"  # positive
    },
    {
        "text": "I was deeply disappointed with this film. The plot had numerous holes and the characters were poorly developed.",
        "label": "0"  # negative
    },
    {
        "text": "One of the best movies of the year! The director's vision really shines through in every scene.",
        "label": "1"  # positive
    },
    {
        "text": "A complete waste of time. The special effects were terrible and the dialogue was cringe-worthy.",
        "label": "0"  # negative
    },
    {
        "text": "I loved every minute of this brilliant cinematic masterpiece. The cinematography was breathtaking.",
        "label": "1"  # positive
    },
    {
        "text": "Such a boring and predictable plot. I almost fell asleep halfway through the movie.",
        "label": "0"  # negative
    },
    {
        "text": "The performances in this film were outstanding, particularly the lead actor who delivered an Oscar-worthy performance.",
        "label": "1"  # positive
    },
    {
        "text": "This movie tries too hard to be profound but ends up being pretentious and dull.",
        "label": "0"  # negative
    },
    {
        "text": "A heartwarming story with incredible performances. I was moved to tears by the ending.",
        "label": "1"  # positive
    },
    {
        "text": "Terrible acting, awful script, and poor direction. One of the worst films I've seen this year.",
        "label": "0"  # negative
    }
]

def populate_chromadb():
    """
    Populate ChromaDB with sample movie reviews
    """
    print("Starting ChromaDB population process...")
    
    # Initialize models and get ChromaDB client
    initialize_models()
    
    # Check if running in Docker by looking for environment variables
    in_docker = os.getenv("CHROMA_DB_HOST") is not None
    
    if in_docker:
        # Connect to ChromaDB service running in Docker
        chroma_host = os.getenv("CHROMA_DB_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_DB_PORT", "8000"))
        print(f"Running in Docker: Connecting to ChromaDB at {chroma_host}:{chroma_port}")
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        # Connect to local persistent ChromaDB
        VECTOR_DB_DIR = root_dir / "resources" / "vectorial_database"
        print(f"Running locally: Connecting to vector database at: {VECTOR_DB_DIR}")
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    
    # Wait for ChromaDB to be ready (especially important in Docker)
    max_retries = 10
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1} to connect to ChromaDB...")
            # Try to list collections - this will fail if ChromaDB isn't ready
            client.list_collections()
            print("Successfully connected to ChromaDB!")
            break
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Maximum retry attempts reached. Exiting.")
                sys.exit(1)
    
    # Get or create collection
    collection = client.get_or_create_collection("reviews")
    
    # Check if collection already has data
    if collection.count() > 0:
        print(f"Collection 'reviews' already contains {collection.count()} items. Skipping population.")
        return
    
    print("Preparing to add sample reviews to ChromaDB...")
    
    # Siempre inicializamos un nuevo modelo de embedding aquí
    # en lugar de usar el de sentiment_service
    embedding_model = SentenceTransformer(str(root_dir / "resources" / "embedding_model"), device="cpu")
    
    # Generate embeddings for sample reviews
    texts = [review["text"] for review in SAMPLE_REVIEWS]
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    
    # Generate IDs for the reviews
    ids = [f"review_{i}" for i in range(len(SAMPLE_REVIEWS))]
    
    # Prepare metadata
    metadatas = [{"label": review["label"]} for review in SAMPLE_REVIEWS]
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=[emb.tolist() for emb in embeddings],
        documents=texts,
        metadatas=metadatas
    )
    
    print(f"Successfully added {len(SAMPLE_REVIEWS)} sample reviews to ChromaDB collection 'reviews'")
    print(f"Collection now contains {collection.count()} items")

if __name__ == "__main__":
    populate_chromadb()