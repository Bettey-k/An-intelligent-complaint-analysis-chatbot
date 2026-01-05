🏦 CFPB Complaint Analysis Assistant (RAG-Based)

An intelligent Retrieval-Augmented Generation (RAG) system built on Consumer Financial Protection Bureau (CFPB) complaint data.
The system enables users to ask natural-language questions about financial complaints and receive evidence-backed answers grounded in real customer narratives.

📌 Project Overview

Financial institutions receive thousands of customer complaints across products such as credit cards, loans, and bank accounts.
Manually analyzing these complaints is slow and error-prone.

This project builds an end-to-end RAG pipeline that:

Analyzes and preprocesses CFPB complaints

Creates semantic embeddings and vector stores

Retrieves relevant complaint excerpts

Generates grounded answers using a Large Language Model

Provides an interactive web interface for non-technical users

🧩 Tasks Summary
Task	Description
Task 1	Exploratory Data Analysis (EDA) & preprocessing
Task 2	Text chunking & embedding pipeline (sample data)
Task 3	Full-scale RAG pipeline & qualitative evaluation
Task 4	Interactive UI using Gradio
📁 Repository Structure
An-intelligent-complaint-analysis-chatbot/
│
├── data/
│   ├── raw/
│   │   ├── complaints.csv
│   │   └── complaint_embeddings.parquet
│   └── processed/
│       └── filtered_complaints.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── chunking_embeddings.ipynb
│
├── rag/
│   ├── build_chroma_store.py
│   ├── retriever.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   ├── evaluation.py
│   └── run_evaluation.py
│
├── vector_store/
│   └── chroma/
│
├── tests/
│   └── test_retriever.py
│
├── app.py
├── requirements.txt
└── README.md
📊 Task 1 – EDA & Preprocessing

Notebook: notebooks/EDA.ipynb

Key Steps

Loaded full CFPB complaints dataset

Analyzed:

Product distribution

Missing complaint narratives

Narrative length statistics

Filtered to required financial products

Cleaned complaint text (lowercasing, whitespace cleanup)

Saved cleaned data to:
data/processed/filtered_complaints.csv
Task 2 – Chunking & Embeddings (Sample)

Notebook: notebooks/chunking_embeddings.ipynb

Highlights

Stratified sampling (10k–15k complaints)

Text chunking:

Chunk size: 500 characters

Overlap: 50 characters

Sentence embeddings using:

all-MiniLM-L6-v2
FAISS index built for experimentation

🧠 Task 3 – RAG Pipeline & Evaluation
Vector Store

Used pre-built embeddings (complaint_embeddings.parquet)

Built ChromaDB vector store with ~1.3M chunks

Metadata preserved per chunk (product, issue, company, state, etc.)

RAG Components

Retriever: Semantic similarity search (top-k)

Generator: Mistral-7B-Instruct

Prompt Engineering: Context-restricted, analyst-style answers

Evaluation

Evaluated on 5 representative questions

Generated qualitative evaluation table:

Question

Generated Answer

Retrieved Sources

Quality Score

Analysis

Run evaluation:

python -m rag.run_evaluation


Output:

evaluation_results.md

💬 Task 4 – Interactive UI (Gradio)

File: app.py

Features

Question input box

Ask & Clear buttons

AI-generated answer display

Source complaint excerpts shown below the answer

Clean, user-friendly layout

Run the App
python app.py


Then open the local Gradio link in your browser.

Example Questions

What are the most common issues customers report with credit cards?

How do customers describe unauthorized transactions?

What complaints are common about late payment fees?

🛠 Tech Stack

Python 3.11

Pandas / NumPy

SentenceTransformers

ChromaDB

FAISS

Transformers (Mistral LLM)

Gradio

PyArrow

📦 Installation
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

✅ Key Outcomes

End-to-end RAG system on real financial complaints

Scalable vector search over 1M+ text chunks

Evidence-backed AI responses

Interactive UI for non-technical users

Clean, modular, production-ready codebase

📌 Future Improvements (Optional)

Response streaming in UI

Faster LLM inference (quantization / smaller models)

Hybrid keyword + semantic retrieval

User feedback loop for evaluation

👤 Author

Betelhem Kibret Getu
