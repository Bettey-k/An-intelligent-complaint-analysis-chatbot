# setup.py
from setuptools import setup, find_packages

setup(
    name="rag_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
        "chromadb",
        "pandas",
        "numpy",
        "tqdm",
        "pyarrow",
        "transformers",
        "torch",
        "accelerate"
    ]
)