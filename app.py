import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
from rag.rag_pipeline import RAGPipeline

# Initialize RAG pipeline once (important for performance)
rag = RAGPipeline()

def ask_question(question):
    """
    Handles user question and returns answer + sources.
    """
    if not question.strip():
        return "Please enter a question.", ""

    result = rag.query(question, k=3)

    answer = result["answer"]

    # Format sources nicely
    sources_text = ""
    for i, src in enumerate(result["sources"], start=1):
        meta = src["metadata"]
        sources_text += f"""
### Source {i}
**Product:** {meta.get('product')}
**Issue:** {meta.get('issue')}
**Company:** {meta.get('company')}
**State:** {meta.get('state')}

> {src['text'][:500]}...
---
"""

    return answer, sources_text


def clear_chat():
    return "", "", ""


# Build UI
with gr.Blocks(title="CFPB Complaint Analysis Assistant") as demo:
    gr.Markdown(
        """
        # 🏦 CFPB Complaint Analysis Assistant
        Ask questions about customer complaints related to financial products.
        """
    )

    question_input = gr.Textbox(
        label="Enter your question",
        placeholder="e.g. What problems do customers report with credit cards?"
    )

    ask_btn = gr.Button("Ask")
    clear_btn = gr.Button("Clear")

    answer_output = gr.Textbox(
        label="AI Answer",
        lines=6
    )

    sources_output = gr.Markdown(
        label="Sources (Retrieved Complaints)"
    )

    ask_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[question_input, answer_output, sources_output]
    )

# Run app
if __name__ == "__main__":
    demo.launch()
