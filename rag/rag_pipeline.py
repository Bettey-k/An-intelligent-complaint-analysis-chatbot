# rag/rag_pipeline.py
from .retriever import Retriever
from .generator import Generator
from typing import Dict, Any

class RAGPipeline:
    def __init__(self, chroma_dir: str = "vector_store/chroma", model_name: str = "all-MiniLM-L6-v2"):
        self.retriever = Retriever(chroma_dir=chroma_dir, model_name=model_name)
        self.generator = Generator()

    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        # Retrieve documents
        retrieved = self.retriever.retrieve(question, k)

        # ✅ Extract only text for the generator
        context_texts = [r["text"] for r in retrieved]

        # Generate answer
        answer = self.generator.generate_response(question, context_texts)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved
        }
