# run_evaluation.py
import logging
import time
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm
from rag_pipeline import RAGPipeline
from evaluation import RAGEvaluator, TEST_QUESTIONS
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

def retry_on_error(max_retries=3, delay=5):
    """Decorator to retry a function on failure."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {retries} failed. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator

def evaluate_with_timeout(evaluator, question, timeout=60):  # Increased to 60 seconds
    """Evaluate a single question with timeout using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(evaluator.evaluate_single, question)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            logger.warning(f"Question timed out after {timeout} seconds: {question[:50]}...")
            return {
                'question': question,
                'answer': f"Error: Generation timed out after {timeout} seconds",
                'sources': [],
                'score': 0,
                'analysis': "Generation took too long"
            }

@retry_on_error(max_retries=2, delay=3)
def evaluate_question(evaluator, question: str) -> Dict[str, Any]:
    """Evaluate a single question with timeout and retry logic."""
    try:
        start_time = time.time()
        result = evaluate_with_timeout(evaluator, question, timeout=30)
        elapsed = time.time() - start_time
        logger.info(f"Question processed in {elapsed:.1f}s: {question[:50]}...")
        return result
    except Exception as e:
        logger.error(f"Error evaluating question: {question[:50]}... - {str(e)}")
        return {
            'question': question,
            'answer': f"Error: {str(e)[:200]}",
            'sources': [],
            'score': 0,
            'analysis': f"Error: {str(e)[:100]}..."
        }

def main():
    try:
        # Initialize the RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag = RAGPipeline()
        evaluator = RAGEvaluator(rag)
        
        # Process questions one at a time
        results = []
        for i, question in enumerate(TEST_QUESTIONS):
            logger.info(f"\nProcessing question {i+1}/{len(TEST_QUESTIONS)}: {question[:50]}...")
            try:
                result = evaluate_question(evaluator, question)
                results.append(result)
                
                # Save after each question
                results_df = pd.DataFrame(results)
                RAGEvaluator.save_to_markdown(results_df, "evaluation_partial.md")
                logger.info(f"Progress saved. Processed {i+1}/{len(TEST_QUESTIONS)} questions.")
                
            except Exception as e:
                logger.error(f"Fatal error processing question: {str(e)}", exc_info=True)
                results.append({
                    'question': question,
                    'answer': f"Fatal Error: {str(e)[:200]}",
                    'sources': [],
                    'score': 0,
                    'analysis': "Fatal error during processing"
                })
        
        # Save final results
        output_path = "evaluation_results.md"
        RAGEvaluator.save_to_markdown(pd.DataFrame(results), output_path)
        logger.info(f"\nEvaluation complete! Results saved to {output_path}")
        
        # Print a sample of the results
        logger.info("\nSample of evaluation results:")
        sample = pd.DataFrame(results)[['question', 'answer']].head()
        logger.info("\n" + sample.to_string())
        
        return 0
        
    except Exception as e:
        logger.critical(f"Critical error in evaluation: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)