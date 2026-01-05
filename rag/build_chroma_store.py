# rag/build_chroma_store.py
import os
import pyarrow.parquet as pq
import chromadb
from chromadb.config import Settings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time

# Configuration
PARQUET_PATH = "data/raw/complaint_embeddings.parquet"
CHROMA_DIR = "vector_store/chroma"
BATCH_SIZE = 1000  # Process in batches to manage memory

def main():
    print("Building Chroma vector store...")
    
    # Create output directory if it doesn't exist
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize Chroma client
    print("Initializing Chroma client...")
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Create or get collection
    print("Setting up collection...")
    collection = client.get_or_create_collection(
        name="cfpb_complaints",
        metadata={"hnsw:space": "cosine"}  # Using cosine similarity
    )
    
    # Check if collection is empty
    if collection.count() > 0:
        print(f"Collection already contains {collection.count()} documents.")
        response = input("Do you want to reset the collection? (y/n): ")
        if response.lower() == 'y':
            client.reset()
            collection = client.get_or_create_collection(
                name="cfpb_complaints",
                metadata={"hnsw:space": "cosine"}
            )
            print("Collection has been reset.")
        else:
            print("Using existing collection.")
            return
    
    # Read parquet file
    print(f"Reading data from {PARQUET_PATH}...")
    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas()
    
    # Convert embeddings to list of lists if they're not already
    if isinstance(df['embedding'].iloc[0], np.ndarray):
        df['embedding'] = df['embedding'].apply(lambda x: x.tolist())
    
    total_docs = len(df)
    print(f"Found {total_docs} documents to process")
    
    # Process in batches
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing batches"):
        batch = df.iloc[i:i+BATCH_SIZE]
        
        # Prepare batch data
        ids = [str(idx) for idx in batch.index]
        embeddings = batch['embedding'].tolist()
        documents = batch['document'].tolist()
        metadatas = batch['metadata'].tolist()
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    print(f"\nSuccessfully built Chroma vector store with {collection.count()} documents")
    print(f"Vector store location: {os.path.abspath(CHROMA_DIR)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal time: {(time.time() - start_time)/60:.2f} minutes")