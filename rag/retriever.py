# retriever.py
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
import logging

class Retriever:
    def __init__(self, chroma_dir: str = "vector_store/chroma", model_name: str = "all-MiniLM-L6-v2"):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name)
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.chroma_client.get_collection("cfpb_complaints")
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents for the query."""
        try:
            # Encode the query
            query_embedding = self.model.encode(query).tolist()
            
            # Search the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': results['distances'][0][i] if 'distances' in results else None
                })
                
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error in retrieve: {str(e)}")
            return []