# Industrial Safety Document Q&A System

A question-answering system for industrial safety documentation using BM25 search and TF-IDF embeddings.

## Overview

This project implements a document retrieval and question-answering system specifically designed for industrial safety manuals. The system processes PDF documents, extracts text chunks, and uses a combination of BM25 keyword search and TF-IDF embeddings to provide accurate answers to safety-related questions.

## Deployed URL
https://python-developer-technical-assessment.onrender.com
## Architecture

- **PDF Processing**: Extracts and chunks text from safety documentation
- **Embedding Generation**: Creates TF-IDF vectors for semantic similarity
- **Search Engine**: BM25 keyword search with TF-IDF reranking
- **Web Interface**: Flask-based REST API with modern frontend
- **Database**: SQLite for storing document chunks and metadata

## Setup

1. **Install Dependencies**
   ```powershell
   .\setup.ps1
   ```

2. **Start the System**
   ```powershell
   .\start.ps1
   ```

3. **Access the Interface**
   - Web UI: http://localhost:5000
   - API Documentation: http://localhost:5000/api

## API Usage

### Basic Query
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main types of machine guarding?"}'
```

### Complex Query
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between SIL and PL in safety standards?"}'
```

### System Status
```bash
curl http://localhost:5000/status
```

## Results

The system was evaluated on 8 test questions covering various aspects of industrial safety:

| Question Type | Accuracy | Avg Response Time |
|--------------|----------|------------------|
| Easy         | 100%     | 0.8s            |
| Medium       | 75%      | 1.1s            |
| Hard         | 67%      | 1.3s            |
| **Overall**  | **81%**  | **1.1s**        |

### Performance Analysis
- **BM25 Search**: Effective for keyword-based safety queries
- **TF-IDF Reranking**: Improves relevance for technical terminology
- **Citation Accuracy**: 100% of answers include correct document references
- **Response Time**: Sub-2 second average across all query types

## Project Structure

```
qa_service/
├── src/
│   ├── api/           # Flask web server
│   ├── search/        # Search and answering logic
│   ├── ingest/        # PDF processing and embeddings
│   └── utils/         # Configuration and database utilities
├── static/            # Web interface files
├── data/              # SQLite database and models
├── pdfs/              # Source documents (19 safety manuals)
└── test_questions.json # Evaluation dataset
```

## What I Learned

During this project, I gained valuable experience in several key areas of information retrieval and natural language processing. The most significant learning was understanding the trade-offs between different search approaches - while vector embeddings excel at semantic similarity, BM25 keyword search often performs better for technical documentation where exact terminology matters. This led me to implement a hybrid approach that leverages both methods.

I also learned the importance of proper text preprocessing for technical documents. Industrial safety manuals contain specialized terminology, standards references (like ISO 13849-1), and structured information that requires careful handling during the chunking process. The experience taught me that domain-specific optimizations often outperform generic NLP approaches, especially when working with specialized technical documentation.

## Technical Details

- **Documents**: 19 industrial safety PDFs processed into 1,122 text chunks
- **Search Algorithm**: BM25 with TF-IDF reranking (1000 features)
- **Database**: SQLite with optimized indexing
- **Framework**: Flask with CORS support
- **Response Format**: JSON with confidence scores and source citations
- **Evaluation**: 8-question test set with difficulty classifications

## Files

- `sources.json`: Document metadata and URL mappings
- `test_questions.json`: 8-question evaluation dataset
- `requirements.txt`: Python dependencies
- `start.ps1`: Quick start script
- `setup.ps1`: Environment setup

---


*This system demonstrates practical application of information retrieval techniques for domain-specific document search, with emphasis on accuracy, performance, and user experience.*
