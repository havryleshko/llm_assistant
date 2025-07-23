## Who is this AI for?
**- For people involved in finance, investing in assets long- & short- term to better understand financial health**

## What ***one workflow*** does this AI automate?
**- Manual financial risk analysis of 10-K reports**

## What does it do *better* than ChatGPT?
**1. Remembers you and your documents**

**2. Gives precise answers from your files, not generic**

**3. Works faster for your one use case**

**4. Prompted to be your own financial research analyst**

### LLM Assistant
Personal AI document assistant built with LangChain, OpenAI, FastAPI, and Streamlit

## Overview
This project is a conversational AI assistant designed to answer questions based on your personal documents. It leverages retrieval-augmented generation (RAG) with LangChain and FAISS for vector search, OpenAI’s GPT-4o for natural language understanding, and provides a simple and interactive Streamlit frontend

## Features
Conversational Retrieval: Uses LangChain's ConversationalRetrievalChain to combine document retrieval and conversational AI

**Vector Search:** Efficient semantic search powered by FAISS local index.

**Memory:** Maintains conversation history per user for context-aware responses; implemented long-term and short-term

**Streaming Responses:** Supports token-level streaming responses (designed but limited by Streamlit constraints, can be used with other (advanced) deployment tools).

**Custom Prompting:** I have designed custom prompt for certain behaviour

**User Management:** Tracks chat sessions by user name (chat_id)

**File Upload:** Upload your own documents to expand the knowledge base (planned/partially implemented).

## Tech Stack
***see requirements.txt***

## Prerequisites
Python 3.10 or higher

OpenAI API key (set as OPENAI_API_KEY in .env)

## Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Setup

## Clone the repository:

bash
Copy
Edit
git clone https://github.com/havryleshko/llm_assistant.git
cd llm_assistant

## Create a .env file in the root with your OpenAI API key:

env
Copy
Edit
OPENAI_API_KEY=your_openai_api_key_here
Prepare or add your FAISS vector index (faiss_index folder)

## Usage
Run the FastAPI backend
bash
Copy
Edit
uvicorn api:app --reload
This will start the API server on http://localhost:8000.

## Run the Streamlit frontend
bash
Copy
Edit
streamlit run streamlit_deploy.py
Visit http://localhost:8501 in your browser to interact with the assistant.

## Project Structure
api.py — FastAPI backend 

streamlit_deploy.py — Streamlit frontend app

main.py — Core components setup - brain: OpenAI embeddings, FAISS index loading, prompt templates, and chain initialization

faiss_index/ — Local folder containing the FAISS vector store.

.env — Environment variables (OpenAI API key).

## Notes
1. Streaming token-by-token responses is limited by Streamlit's architecture. Consider alternative frontends or further integration for full streaming experience

2. Currently supports conversational memory per user via chat_id

3. Prompt templates are designed for a warm, human-like assistant voice.

4. Contributing
Contributions welcome! Feel free to open issues or pull requests to improve functionality, add features, or fix bugs.

# *License*
MIT License

# *Contact*
Alex Havryleshko
**X:** https://x.com/alexhavryleshko
**Youtube:** https://www.youtube.com/@havryleshko

## Lessons I have learned (main):
1. Architecting modular AI systems for everything: separating core AI logic (FastAPI backend) and user interface (Streamlit frontend) is always the best idea
2. Not all deployment tools are designed to work with AI on LLMs, so choose carefully
3. Use LangChain’s ConversationBufferMemory to track chat history per user enables AI to maintain context over multiple exchanges
4. Building AI products demands iterative cycles: testing code, adjusting parameters, fixing bugs, and refining user experience - be ready to test it for longer











Ask ChatGPT



Tools


