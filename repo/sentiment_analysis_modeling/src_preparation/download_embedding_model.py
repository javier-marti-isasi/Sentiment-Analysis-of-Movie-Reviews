"""
Script to download and save the SentenceTransformer embedding model
"""
import os
import pathlib
from sentence_transformers import SentenceTransformer

def main():
    # Define model name and local path
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_dir = os.path.join("model", "embedding_model")
    
    # Create absolute path
    abs_model_dir = str(pathlib.Path(__file__).parent.parent / "model" / "embedding_model")
    
    print(f"Downloading embedding model: {model_name}")
    print(f"This will be saved to: {os.path.abspath(abs_model_dir)}")
    
    # Create directory if it doesn't exist
    os.makedirs(abs_model_dir, exist_ok=True)
    
    # Download and save the model
    print("Downloading sentence transformer model...")
    model = SentenceTransformer(model_name)
    model.save(abs_model_dir)
    print("Embedding model saved successfully.")
    
    print(f"\nEmbedding model has been downloaded and saved to: {os.path.abspath(abs_model_dir)}")
    print("You can now use this local model in your vector database creation scripts.")

if __name__ == "__main__":
    main()
