# PDF Summarization with AI

## Overview
This project allows users to upload a **PDF document** and receive a **summary** using **LLM-based text processing**. It leverages **LangChain**, **ChromaDB**, and **Hugging Face Falcon-7B** for efficient document retrieval and summarization.

## Features
- **Upload PDFs**: Extract text from documents.
- **AI-powered summarization**: Uses **Falcon-7B** LLM for responses.
- **Vector search**: Stores document embeddings in **ChromaDB** for retrieval.
- **Efficient text splitting**: Uses **RecursiveCharacterTextSplitter** for better context understanding.

## Installation
Ensure you have **Python 3.8+** and install dependencies:
```bash
pip install torch langchain langchain_ibm langchain_community chromadb sentence-transformers pypdf
```

## Project Structure
```
pdf-summarization-ai
┣ main.py                # Main script for processing PDFs
┣ requirements.txt        # Dependencies list
┣ README.md               # Project documentation
```

## Usage
1. **Initialize the LLM model**:
   ```python
   from main import init_llm
   init_llm()
   ```
2. **Process a PDF document**:
   ```python
   from main import process_document
   process_document("sample.pdf")
   ```
3. **Ask a question based on the document**:
   ```python
   from main import process_prompt
   response = process_prompt("What is the main topic of the document?")
   print(response)
   ```

## Environment Variables
Set up your Hugging Face API key:
```bash
export HUGGINGFACEHUB_API_TOKEN="your_api_key"
```

## License
**Apache License 2.0 @ Krzychu 2025**.

---
