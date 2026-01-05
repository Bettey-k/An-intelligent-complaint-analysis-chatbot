# tests/test_rag_pipeline.py
from rag.rag_pipeline import RAGPipeline
from rag.evaluation import RAGEvaluator

def main():
    # Initialize the RAG pipeline
    rag = RAGPipeline()
    
    # Example questions for evaluation
    test_questions = [
        "What are common issues with credit cards?",
        "How do customers report unauthorized transactions?",
        "What problems do people have with late payment fees?",
        "How do customers feel about the dispute resolution process?",
        "What are the main complaints about customer service?"
    ]
    
    # Run evaluation
    evaluator = RAGEvaluator(rag)
    results_df = evaluator.evaluate_questions(test_questions)
    
    # Save results
    evaluator.save_to_markdown(results_df, "evaluation_results.md")
    print("Evaluation complete! Results saved to evaluation_results.md")
    
    # Example of querying the pipeline
    while True:
        question = input("\nEnter a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        response = rag.query(question)
        print("\nAnswer:")
        print(response["answer"])
        print("\nSources:")
        for i, src in enumerate(response["sources"][:2], 1):
            print(f"\nSource {i}:")
            print(f"Product: {src['metadata'].get('product', 'N/A')}")
            print(f"Issue: {src['metadata'].get('issue', 'N/A')}")
            print(f"Text: {src['text'][:200]}...")

if __name__ == "__main__":
    main()