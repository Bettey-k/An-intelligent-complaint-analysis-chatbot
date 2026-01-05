from typing import List, Dict, Any
import pandas as pd
import logging

class RAGEvaluator:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def save_to_markdown(results, output_file: str) -> None:
        """Save evaluation results to a markdown file."""
        if isinstance(results, pd.DataFrame):
            df = results
        else:
            df = pd.DataFrame(results)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RAG Pipeline Evaluation Results\n\n")
            f.write(df.to_markdown(index=False))

    def evaluate_single(self, question: str) -> Dict[str, Any]:
        try:
            response = self.rag.query(question)
            sources = response.get("sources", [])[:2]
            quality_score = self._calculate_initial_quality(response)

            return {
                "question": question,
                "answer": response.get("answer", "No answer generated"),
                "sources": sources,
                "score": quality_score,
                "analysis": "Auto-evaluated"
            }

        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "score": 0,
                "analysis": f"Error: {str(e)[:100]}..."
            }

    def _calculate_initial_quality(self, response: Dict[str, Any]) -> float:
        answer = response.get("answer", "")
        sources = response.get("sources", [])

        if not answer or "Error" in answer:
            return 1.0

        score = 0.5
        if 20 < len(answer) < 500:
            score += 0.2
        if sources:
            score += 0.3

        return round(score * 5, 1)  # 1–5 scale


TEST_QUESTIONS = [
    "What are the most common issues with credit cards?",
    "How do customers typically report unauthorized transactions?",
    "What are customers saying about late payment fees?",
    "How do customers describe their experiences with customer service?",
    "What issues do customers face with credit score reporting?"
]

