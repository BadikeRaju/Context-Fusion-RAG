# Context Fusion – RAG-Based Conversational AI Platform

> An AI-powered Retrieval-Augmented Generation (RAG) application that enables intelligent conversational search over custom documents using FastAPI, React, and the Mistral API.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![React](https://img.shields.io/badge/React-19-blue)
![License](https://img.shields.io/badge/License-MIT-success)

---

# Overview

Context Fusion is a full-stack AI application that allows users to upload documents and interact with them through natural language conversations. It combines Retrieval-Augmented Generation (RAG) with hybrid document retrieval techniques to provide accurate, context-aware responses while reducing hallucinations from large language models.

The application is designed for enterprise knowledge retrieval, research assistance, technical documentation search, and intelligent document exploration.

---

# Features

## AI-Powered Conversational Search

- Context-aware responses
- Natural language question answering
- Multi-turn conversations
- Conversation history

---

## Retrieval-Augmented Generation

- Hybrid document retrieval
- Context injection
- Prompt construction
- Response generation

---

## Hybrid Search Engine

Supports multiple retrieval techniques:

- BM25
- TF-IDF
- Word2Vec
- Annoy Vector Search

---

## Document Management

Supported formats

- PDF
- DOCX
- TXT
- Markdown

Features

- Upload documents
- Delete documents
- Re-index documents
- Metadata extraction

---

## AI Features

- Context-aware answers
- Similarity reranking
- Source-aware responses
- Low-latency retrieval

---

# Tech Stack

## Frontend

- React.js
- React Router
- Tailwind CSS
- Axios

## Backend

- Python
- FastAPI
- Pydantic

## AI & NLP

- Mistral API
- BM25
- TF-IDF
- Word2Vec
- Annoy

## Database

- MySQL

## Tools

- Docker
- Git
- GitHub

---

# System Architecture

```
                React.js
                    │
              REST API
                    │
                FastAPI
                    │
        Document Processing
                    │
      Hybrid Retrieval Engine
(BM25 + TF-IDF + Word2Vec + Annoy)
                    │
         Similarity Reranking
                    │
              Mistral API
                    │
              AI Response
                    │
                 MySQL
```

---

# RAG Pipeline

```
User Query
     │
     ▼
FastAPI
     │
     ▼
Query Processing
     │
     ▼
Hybrid Retrieval
     │
     ▼
Similarity Reranking
     │
     ▼
Relevant Context
     │
     ▼
Mistral API
     │
     ▼
Generated Response
```

---

# REST APIs

## Documents

```http
POST   /documents/upload
GET    /documents
DELETE /documents/{id}
POST   /documents/reindex
```

## Chat

```http
POST /chat
GET  /chat/history
```

## Search

```http
POST /search
GET  /search/history
```

---

# Document Processing Workflow

1. Upload a document.
2. Extract and preprocess text.
3. Split content into smaller chunks.
4. Build BM25, TF-IDF, Word2Vec, and Annoy indexes.
5. Accept user queries.
6. Retrieve relevant document chunks.
7. Rerank retrieved content.
8. Send contextual prompt to the Mistral API.
9. Display the generated response.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/context-fusion.git
cd context-fusion
```

## Backend

```bash
python -m venv venv
```

Activate

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run FastAPI

```bash
uvicorn main:app --reload
```

---

## Frontend

```bash
cd frontend

npm install

npm run dev
```

---

## Docker

```bash
docker compose up --build
```

---

# Future Enhancements

- FAISS Integration
- Multi-LLM Support
- Streaming Responses
- OCR Support
- Voice-Based Search
- Document Summarization
- Citation Generation
- Feedback-Based Learning

---

# Skills Demonstrated

- Python
- FastAPI
- React.js
- REST APIs
- MySQL
- Docker
- Retrieval-Augmented Generation (RAG)
- BM25
- TF-IDF
- Word2Vec
- Annoy
- NLP
- Semantic Search
- Backend Development
- API Design

---

# Use Cases

- Enterprise Knowledge Bases
- Research Assistance
- Technical Documentation Search
- Customer Support
- Academic Search
- Internal Documentation

---

# License

This project is licensed under the MIT License.

---

# Author

**Raju Badike**
