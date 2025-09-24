"""
Simple Search and Retrieval Engine for QA Service

This file handles finding the most relevant text chunks for a given question.
It uses TF-IDF embeddings with cosine similarity for reliable search results.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import pickle
import logging
import re

from ..utils.config import Config
from ..utils.database import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleHybridRetriever:
    """
    Combines TF-IDF similarity and keyword matching for better search results
    
    This is like having two different search strategies:
    1. TF-IDF search: Finds text with similar word importance patterns
    2. Keyword search: Finds text with exact word matches
    Then combines both for the best results.
    """
    
    def __init__(self):
        """Initialize the retriever"""
        self.db = DatabaseManager()
        self.chunks_data = None
        self.embeddings_matrix = None
        self.bm25_index = None
        self.chunk_texts = None
        self.vectorizer = None
        self._load_data()
    
    def _load_data(self):
        """
        Load all chunks with embeddings from database
        
        This prepares all our text chunks and their numerical representations
        for fast searching.
        """
        logger.info("📚 Loading chunks and embeddings...")
        
        # Get all chunks with embeddings
        chunks = self.db.get_all_chunks_with_embeddings()
        
        if not chunks:
            logger.warning("⚠️ No chunks with embeddings found. Run embedding generation first.")
            return
        
        self.chunks_data = chunks
        
        # Create embeddings matrix for vector search
        embeddings_list = [chunk['embedding'] for chunk in chunks]
        self.embeddings_matrix = np.vstack(embeddings_list)
        
        # Prepare texts for BM25 keyword search
        self.chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Load the TF-IDF vectorizer
        self._load_vectorizer()
        
        # Create BM25 index
        self._build_bm25_index()
        
        logger.info(f"✅ Loaded {len(chunks)} chunks with embeddings")
    
    def reload_data(self):
        """
        Force reload all data from database
        
        Use this when the database has been updated.
        """
        logger.info("🔄 Reloading data from database...")
        self.chunks_data = None
        self.embeddings_matrix = None
        self.bm25_index = None
        self.chunk_texts = None
        self.vectorizer = None
        self._load_data()
    
    def _load_vectorizer(self):
        """
        Load the saved TF-IDF vectorizer
        """
        try:
            with open(Config.DATA_DIR / "tfidf_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("✅ TF-IDF vectorizer loaded")
        except FileNotFoundError:
            logger.error("❌ TF-IDF vectorizer not found. Run embedding generation first.")
    
    def _build_bm25_index(self):
        """
        Build BM25 keyword search index
        
        BM25 is a scoring algorithm that finds text containing
        specific keywords, giving higher scores to rare, important words.
        """
        if not self.chunk_texts:
            logger.warning("No chunk texts available for BM25 indexing")
            return
        
        logger.info("🔍 Building BM25 keyword index...")
        
        # Tokenize texts (split into words)
        tokenized_texts = []
        for text in self.chunk_texts:
            # Simple tokenization: lowercase, remove punctuation, split by spaces
            tokens = re.findall(r'\b\w+\b', text.lower())
            tokenized_texts.append(tokens)
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_texts)
        
        logger.info("✅ BM25 index built successfully")
    
    def _vector_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Search using TF-IDF similarity
        
        This finds chunks that have similar word importance patterns to the query.
        """
        if self.embeddings_matrix is None or len(self.chunks_data) == 0 or not self.vectorizer:
            return []
        
        try:
            # Transform query using the same vectorizer
            query_vector = self.vectorizer.transform([query]).toarray()
            
            # Clean query vector of NaN/inf values
            query_vector = np.nan_to_num(query_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clean embeddings matrix of NaN/inf values
            clean_embeddings = np.nan_to_num(self.embeddings_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate cosine similarities with all chunks
            similarities = cosine_similarity(query_vector, clean_embeddings)[0]
            
            # Clean similarities of NaN/inf values
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include chunks with positive similarity
                    chunk = self.chunks_data[idx]
                    results.append({
                        'chunk_id': chunk['id'],
                        'text': chunk['text'],
                        'file_name': chunk['file_name'],
                        'title': chunk['title'],
                        'vector_score': float(similarities[idx])
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Search using BM25 keyword matching
        
        This finds chunks that contain the exact words from the query,
        with higher scores for rare, specific terms.
        """
        if self.bm25_index is None:
            return []
        
        # Tokenize the query
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Get BM25 scores for all chunks
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k highest scoring chunks
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunks_data) and bm25_scores[idx] > 0:  # Only include chunks with positive scores
                chunk = self.chunks_data[idx]
                results.append({
                    'chunk_id': chunk['id'],
                    'text': chunk['text'],
                    'file_name': chunk['file_name'],
                    'title': chunk['title'],
                    'keyword_score': float(bm25_scores[idx])
                })
        
        return results
    
    def _normalize_scores(self, results: List[Dict], score_key: str) -> List[Dict]:
        """
        Normalize scores to 0-1 range
        
        This makes scores from different search methods comparable.
        """
        if not results:
            return results
        
        scores = [r[score_key] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for result in results:
                result[f'normalized_{score_key}'] = 1.0
        else:
            # Normalize to 0-1
            for result in results:
                normalized = (result[score_key] - min_score) / (max_score - min_score)
                result[f'normalized_{score_key}'] = normalized
        
        return results
    
    def _combine_results(self, vector_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """
        Combine results from vector and keyword search
        
        This merges the two types of search results, giving each chunk
        a combined score based on both TF-IDF similarity and keyword matching.
        """
        # Create a dictionary to merge results by chunk_id
        combined = {}
        
        # Add vector search results
        for result in vector_results:
            chunk_id = result['chunk_id']
            combined[chunk_id] = result.copy()
            combined[chunk_id]['normalized_keyword_score'] = 0.0  # Default if not found in keyword search
        
        # Add keyword search results
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id in combined:
                # Merge with existing result
                combined[chunk_id]['keyword_score'] = result['keyword_score']
                combined[chunk_id]['normalized_keyword_score'] = result['normalized_keyword_score']
            else:
                # Add new result
                combined[chunk_id] = result.copy()
                combined[chunk_id]['vector_score'] = 0.0  # Default if not found in vector search
                combined[chunk_id]['normalized_vector_score'] = 0.0
        
        # Calculate final hybrid scores
        for chunk_id, result in combined.items():
            vector_score = result.get('normalized_vector_score', 0.0)
            keyword_score = result.get('normalized_keyword_score', 0.0)
            
            # Weighted combination (configurable in Config)
            final_score = (Config.VECTOR_WEIGHT * vector_score + 
                          Config.KEYWORD_WEIGHT * keyword_score)
            
            result['final_score'] = final_score
        
        # Sort by final score
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_results
    
    def search(self, query: str, top_k: int = None, mode: str = 'hybrid') -> List[Dict]:
        """
        Main search function
        
        This is the main entry point for searching. You can choose:
        - 'hybrid': Use both TF-IDF and keyword search (recommended)
        - 'vector': Use only TF-IDF search
        - 'keyword': Use only keyword-based search
        """
        top_k = top_k or Config.TOP_K_RESULTS
        
        if not self.chunks_data:
            logger.warning("No data available for search. Run data loading first.")
            return []
        
        if mode == 'vector':
            # Vector search only
            results = self._vector_search(query, top_k)
            results = self._normalize_scores(results, 'vector_score')
            for result in results:
                result['final_score'] = result['normalized_vector_score']
            
        elif mode == 'keyword':
            # Keyword search only
            results = self._keyword_search(query, top_k)
            results = self._normalize_scores(results, 'keyword_score')
            for result in results:
                result['final_score'] = result['normalized_keyword_score']
            
        else:  # hybrid mode (default)
            # Get results from both search methods
            vector_results = self._vector_search(query, top_k)
            keyword_results = self._keyword_search(query, top_k)
            
            # Normalize scores
            vector_results = self._normalize_scores(vector_results, 'vector_score')
            keyword_results = self._normalize_scores(keyword_results, 'keyword_score')
            
            # Combine results
            results = self._combine_results(vector_results, keyword_results)
        
        # Return top-k results
        return results[:top_k]
    
    def reload_data(self):
        """
        Reload data from database
        
        Call this if new documents have been added or embeddings have been generated.
        """
        logger.info("🔄 Reloading search data...")
        self._load_data()
    
    def get_search_stats(self) -> Dict:
        """
        Get statistics about the search system
        
        This shows how many chunks are available for searching.
        """
        return {
            'chunks_available': len(self.chunks_data) if self.chunks_data else 0,
            'embeddings_loaded': self.embeddings_matrix is not None,
            'bm25_index_ready': self.bm25_index is not None,
            'vectorizer_loaded': self.vectorizer is not None,
            'vector_weight': Config.VECTOR_WEIGHT,
            'keyword_weight': Config.KEYWORD_WEIGHT
        }


def main():
    """
    Test the search functionality
    
    This demonstrates how the search system works with example queries.
    """
    print("🔍 Testing Search System...")
    
    retriever = SimpleHybridRetriever()
    
    # Show search statistics
    stats = retriever.get_search_stats()
    print(f"📊 Search System Status:")
    print(f"   - Chunks available: {stats['chunks_available']}")
    print(f"   - Embeddings loaded: {stats['embeddings_loaded']}")
    print(f"   - BM25 index ready: {stats['bm25_index_ready']}")
    print(f"   - Vectorizer loaded: {stats['vectorizer_loaded']}")
    print(f"   - Vector weight: {stats['vector_weight']}")
    print(f"   - Keyword weight: {stats['keyword_weight']}")
    
    if stats['chunks_available'] == 0:
        print("❌ No chunks available for search. Please:")
        print("   1. Process PDFs first: python -m src.ingest.pdf_processor")
        print("   2. Generate embeddings: python -m src.ingest.simple_embedder")
        return
    
    # Test searches
    test_queries = [
        "industrial safety equipment",
        "worker protection measures",
        "risk assessment procedures",
        "lockout tagout",
        "machine guarding"
    ]
    
    for query in test_queries:
        print(f"\\n🔍 Testing query: '{query}'")
        
        # Test different search modes
        for mode in ['hybrid', 'vector', 'keyword']:
            print(f"\\n   📋 {mode.title()} search:")
            results = retriever.search(query, top_k=3, mode=mode)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"      {i}. Score: {result['final_score']:.3f}")
                    print(f"         Source: {result['file_name']}")
                    print(f"         Text preview: {result['text'][:100]}...")
            else:
                print("      No results found")
    
    print("\\n✨ Search system test complete!")


if __name__ == "__main__":
    main()