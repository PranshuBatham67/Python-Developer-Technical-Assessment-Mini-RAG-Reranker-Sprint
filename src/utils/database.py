"""
Database Management for QA Service

This file handles all database operations.
Think of it as the librarian that organizes all our documents and text chunks.
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from .config import Config

class DatabaseManager:
    """
    Manages all database operations
    
    This is like a smart filing system that:
    1. Stores document information
    2. Stores text chunks with their embeddings
    3. Keeps track of what we've processed
    """
    
    def __init__(self):
        """Initialize database connection and create tables if needed"""
        self.db_path = Config.DATABASE_PATH
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection
        
        Each time we need to talk to the database, we get a connection.
        It's like picking up the phone to call the database.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This lets us access columns by name
        return conn
    
    def init_database(self):
        """
        Create all necessary database tables
        
        This sets up our filing system with different types of folders:
        - documents: Information about each PDF file
        - chunks: Small pieces of text from the documents
        - metadata: Settings and configuration
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Table to store document information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                title TEXT,
                url TEXT,
                file_hash TEXT UNIQUE,
                total_chunks INTEGER DEFAULT 0,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table to store text chunks with embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT,
                chunk_index INTEGER,
                text TEXT NOT NULL,
                word_count INTEGER,
                embedding BLOB,
                bm25_terms TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (id)
            )
        """)
        
        # Table for storing configuration and metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster searching
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_word_count ON chunks(word_count)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)")
        
        # Create full-text search index for chunks
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                content='chunks',
                content_rowid='rowid'
            )
        """)
        
        conn.commit()
        conn.close()
    
    def file_exists(self, file_hash: str) -> bool:
        """
        Check if a file has already been processed
        
        This prevents us from processing the same PDF twice.
        It's like checking if we already filed a document.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def add_document(self, file_name: str, title: str, url: str = None, file_hash: str = None) -> str:
        """
        Add a new document to the database
        
        This is like creating a new folder in our filing cabinet
        and writing the document's basic information on the tab.
        """
        doc_id = self._generate_doc_id(file_name, file_hash)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents 
            (id, file_name, title, url, file_hash, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, file_name, title, url, file_hash, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return doc_id
    
    def add_chunk(self, doc_id: str, chunk_index: int, text: str, embedding: np.ndarray = None) -> str:
        """
        Add a text chunk to the database
        
        This stores a small piece of text from a document.
        It's like putting a note card with a quote into our filing system.
        """
        chunk_id = f"{doc_id}_chunk_{chunk_index}"
        word_count = len(text.split())
        
        # Convert embedding to binary format for storage
        embedding_blob = None
        if embedding is not None:
            embedding_blob = embedding.astype(np.float32).tobytes()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO chunks 
            (id, doc_id, chunk_index, text, word_count, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (chunk_id, doc_id, chunk_index, text, word_count, embedding_blob, datetime.now()))
        
        # Add to full-text search index
        cursor.execute("""
            INSERT OR REPLACE INTO chunks_fts (chunk_id, text)
            VALUES (?, ?)
        """, (chunk_id, text))
        
        conn.commit()
        conn.close()
        
        return chunk_id
    
    def get_all_chunks_with_embeddings(self) -> List[Dict]:
        """
        Get all chunks that have embeddings
        
        This retrieves all our stored text pieces with their number representations.
        It's like getting all the note cards that have been indexed.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.id, c.text, c.embedding, d.file_name, d.title
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY d.file_name, c.chunk_index
        """)
        
        results = []
        for row in cursor.fetchall():
            # Convert embedding back to numpy array
            embedding = None
            if row['embedding']:
                embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            
            results.append({
                'id': row['id'],
                'text': row['text'],
                'embedding': embedding,
                'file_name': row['file_name'],
                'title': row['title']
            })
        
        conn.close()
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get a specific chunk by its ID
        
        This is like finding a specific note card by its number.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.*, d.file_name, d.title, d.url
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE c.id = ?
        """, (chunk_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row['id'],
                'text': row['text'],
                'file_name': row['file_name'],
                'title': row['title'],
                'url': row['url'],
                'chunk_index': row['chunk_index']
            }
        return None
    
    def search_chunks_fts(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search chunks using full-text search
        
        This finds chunks that contain specific words from the query.
        It's like searching through all note cards for specific keywords.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Escape the query for FTS5
        escaped_query = query.replace('"', '""')
        
        cursor.execute("""
            SELECT c.id, c.text, d.file_name, d.title, d.url,
                   chunks_fts.rank as fts_score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.chunk_id = c.id
            JOIN documents d ON c.doc_id = d.id
            WHERE chunks_fts MATCH ?
            ORDER BY chunks_fts.rank
            LIMIT ?
        """, (f'"{escaped_query}"', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'text': row['text'],
                'file_name': row['file_name'],
                'title': row['title'],
                'url': row['url'],
                'fts_score': row['fts_score']
            })
        
        conn.close()
        return results
    
    def update_document_chunk_count(self, doc_id: str):
        """
        Update the total chunk count for a document
        
        This keeps track of how many pieces we've stored for each document.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
        count = cursor.fetchone()[0]
        
        cursor.execute(
            "UPDATE documents SET total_chunks = ? WHERE id = ?", 
            (count, doc_id)
        )
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict:
        """
        Get database statistics
        
        This gives us a summary of what's in our filing system.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Count documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        # Count chunks
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        
        # Count chunks with embeddings
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        embedded_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'documents': doc_count,
            'total_chunks': chunk_count,
            'embedded_chunks': embedded_count,
            'embedding_coverage': f"{(embedded_count/chunk_count*100):.1f}%" if chunk_count > 0 else "0%"
        }
    
    def _generate_doc_id(self, file_name: str, file_hash: str = None) -> str:
        """
        Generate a unique ID for a document
        
        This creates a unique identifier for each document.
        Like giving each file a unique barcode.
        """
        if file_hash:
            return hashlib.md5(file_hash.encode()).hexdigest()[:16]
        else:
            return hashlib.md5(file_name.encode()).hexdigest()[:16]