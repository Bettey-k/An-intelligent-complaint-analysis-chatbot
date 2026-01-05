# generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
from typing import List, Dict, Any

class Generator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.logger = self._setup_logging()
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        try:
            self.logger.info(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            self.logger.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _setup_logging(self):
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
        
        return logger

    def format_prompt(self, query: str, context: List[str]) -> str:
        """Format the prompt for the model."""
        context_str = "\n".join([f"- {c}" for c in context])
        return f"""You are a helpful AI assistant. Answer the question based on the following context:

Context:
{context_str}

Question: {query}

Answer:"""

    def generate_response(self, query: str, context: List[str], max_new_tokens: int = 100) -> str:
        """Generate a response using the model."""
        try:
            prompt = self.format_prompt(query, context)
            
            # Generate response with explicit parameters
            response = self.pipeline(
                prompt,
                max_new_tokens=min(max_new_tokens, 150),
                max_length=4096,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True,
                max_time=20
            )
            
            # Clean up the response
            if not response or not isinstance(response, list) or not response[0].get('generated_text'):
                return "Error: Invalid response format from model"
                
            full_response = response[0]['generated_text']
            answer = full_response.split("Answer:")[-1].strip()
            return answer.split(self.tokenizer.eos_token or '')[0].strip()
            
        except Exception as e:
            self.logger.error(f"Generation error: {str(e)}")
            return f"Error generating response: {str(e)[:150]}"