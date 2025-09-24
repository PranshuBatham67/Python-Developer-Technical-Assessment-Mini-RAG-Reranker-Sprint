"""
Simple Embedding Generation for QA Service

This file converts text chunks into numerical vectors using TF-IDF.
While not as sophisticated as neural embeddings, it's reliable and works well.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import pickle

from ..utils.config import Config
from ..utils.database import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmbeddingGenerator:
    """
    Generates TF-IDF embeddings for text chunks
    
    This creates numerical representations of text using TF-IDF,
    which measures how important words are to each document.
    """
    
    def __init__(self):
        """Initialize the embedding generator"""
        self.db = DatabaseManager()
        self.vectorizer = None
        self._setup_vectorizer()
    
    def _setup_vectorizer(self):
        """
        Set up the TF-IDF vectorizer
        
        This creates a tool that converts text into numerical vectors
        based on word importance and frequency.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit to most important 1000 words
            stop_words='english',  # Remove common words like "the", "and"
            ngram_range=(1, 2),  # Use single words and word pairs
            max_df=0.8,  # Ignore words that appear in more than 80% of documents
            min_df=2  # Ignore words that appear in fewer than 2 documents
        )
        logger.info("TF-IDF vectorizer initialized")
    
    def generate_embeddings_for_text(self, texts: List[str]) -> np.ndarray:
        """
        Generate TF-IDF embeddings for a list of texts
        
        This converts a list of text chunks into numerical vectors
        that represent the importance of words in each chunk.
        """
        if not texts:
            return np.array([])
        
        try:
            logger.info(f"Generating TF-IDF embeddings for {len(texts)} text chunks")
            
            # Fit the vectorizer on all texts and transform them
            embeddings = self.vectorizer.fit_transform(texts)
            
            # Convert to dense numpy array
            embeddings_dense = embeddings.toarray().astype(np.float32)
            
            logger.info(f"Generated embeddings with shape: {embeddings_dense.shape}")
            
            # Save the vectorizer for later use
            with open(Config.DATA_DIR / "tfidf_vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            return embeddings_dense
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def process_chunks_without_embeddings(self) -> Dict[str, int]:
        """
        Find text chunks that don't have embeddings yet and generate them
        
        This goes through our database and creates embeddings for any
        text chunks that don't have them yet.
        """
        logger.info("Finding chunks without embeddings...")
        
        # Get chunks that don't have embeddings
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.id, c.text, c.doc_id, c.chunk_index
            FROM chunks c
            WHERE c.embedding IS NULL
            ORDER BY c.doc_id, c.chunk_index
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            logger.info("All chunks already have embeddings!")
            return {"processed": 0, "failed": 0}
        
        # Extract texts and IDs
        chunk_texts = [row['text'] for row in results]
        chunk_ids = [row['id'] for row in results]
        
        logger.info(f"Found {len(chunk_texts)} chunks needing embeddings")
        
        try:
            # Generate embeddings for all chunks
            embeddings = self.generate_embeddings_for_text(chunk_texts)
            
            # Store embeddings back in database
            logger.info("Storing embeddings in database...")
            
            processed = 0
            failed = 0
            
            for chunk_id, embedding in tqdm(zip(chunk_ids, embeddings), desc="Storing embeddings"):
                try:
                    # Update chunk with embedding
                    self._update_chunk_embedding(chunk_id, embedding)
                    processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to store embedding for chunk {chunk_id}: {e}")
                    failed += 1
            
            logger.info(f"Embedding generation complete!")
            logger.info(f"   - Successfully processed: {processed}")
            logger.info(f"   - Failed: {failed}")
            
            return {"processed": processed, "failed": failed}
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            return {"processed": 0, "failed": len(chunk_texts)}
    
    def _update_chunk_embedding(self, chunk_id: str, embedding: np.ndarray):
        """
        Update a chunk with its embedding in the database
        
        This saves the numerical representation of a text chunk
        back into our database for later use in searches.
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Convert embedding to binary format for storage
        embedding_blob = embedding.astype(np.float32).tobytes()
        
        cursor.execute("""
            UPDATE chunks 
            SET embedding = ?
            WHERE id = ?
        """, (embedding_blob, chunk_id))
        
        conn.commit()
        conn.close()
    
    def get_embedding_stats(self) -> Dict:
        """
        Get statistics about embedding coverage
        
        This shows how many chunks have embeddings vs. how many don't.
        """
        db_stats = self.db.get_stats()
        
        return {
            "total_chunks": db_stats["total_chunks"],
            "embedded_chunks": db_stats["embedded_chunks"],
            "coverage_percentage": f"{(db_stats['embedded_chunks']/db_stats['total_chunks']*100):.1f}%" if db_stats['total_chunks'] > 0 else "0%",
            "chunks_remaining": db_stats["total_chunks"] - db_stats["embedded_chunks"]
        }
    
    def test_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Test similarity between two pieces of text
        
        This is useful for debugging - it shows how similar
        two texts are according to our embedding model.
        """
        if not self.vectorizer:
            logger.error("Vectorizer not trained yet. Process chunks first.")
            return 0.0
        
        try:
            # Load the saved vectorizer
            with open(Config.DATA_DIR / "tfidf_vectorizer.pkl", 'rb') as f:
                vectorizer = pickle.load(f)
            
            embeddings = vectorizer.transform([text1, text2]).toarray()
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0


def main():
    """
    Main function to run embedding generation
    
    This is what gets called when you run this file directly.
    """
    print("🧠 Starting Simple Embedding Generation...")
    
    embedder = SimpleEmbeddingGenerator()
    
    # Show current status
    stats = embedder.get_embedding_stats()
    print(f"📊 Current Status:")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Chunks with embeddings: {stats['embedded_chunks']}")
    print(f"   - Coverage: {stats['coverage_percentage']}")
    print(f"   - Remaining: {stats['chunks_remaining']}")
    
    if stats['chunks_remaining'] == 0:
        print("✅ All chunks already have embeddings!")
    else:
        print(f"\\n🔄 Processing {stats['chunks_remaining']} chunks...")
        
        # Generate embeddings
        results = embedder.process_chunks_without_embeddings()
        
        print(f"\\n📈 Generation Results:")
        print(f"   - Successfully processed: {results['processed']}")
        print(f"   - Failed: {results['failed']}")
        
        # Show final status
        final_stats = embedder.get_embedding_stats()
        print(f"\\n📊 Final Status:")
        print(f"   - Total embedded: {final_stats['embedded_chunks']}")
        print(f"   - Coverage: {final_stats['coverage_percentage']}")
    
    print("\\n✨ Embedding generation complete!")
    print("💡 Next step: You can now start the API to ask questions!")
    
    # Test similarity with example
    if stats['embedded_chunks'] > 0:
        print("\\n🧪 Testing embedding similarity...")
        similarity = embedder.test_embedding_similarity(
            "industrial safety equipment",
            "safety gear for workers"
        )
        print(f"   Similarity between similar phrases: {similarity:.3f}")
        
        similarity2 = embedder.test_embedding_similarity(
            "industrial safety equipment", 
            "cooking recipes"
        )
        print(f"   Similarity between different topics: {similarity2:.3f}")


if __name__ == "__main__":
    main()