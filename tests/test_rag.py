# test_rag.py
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.rag_pipeline import RAGPipeline

def main():
    # Initialize the RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline()
    
    # Test query
    question = "What are common credit card complaints?"
    print(f"\nQuerying: {question}")
    response = rag.query(question, k=3)
    
    # Print results
    print("\nAnswer:")
    print(response["answer"])
    
    print("\nSources:")
    for i, src in enumerate(response["sources"], 1):
        print(f"\nSource {i}:")
        print(f"Product: {src['metadata'].get('product', 'N/A')}")
        print(f"Issue: {src['metadata'].get('issue', 'N/A')}")
        print(f"Text: {src['text'][:200]}...")

if __name__ == "__main__":
    main()