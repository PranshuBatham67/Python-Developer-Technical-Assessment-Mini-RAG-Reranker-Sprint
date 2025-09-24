"""
Simple Answer Generation for QA Service

This file takes search results and generates clear answers with citations.
Works with TF-IDF embeddings for reliable performance.
"""

import re
from typing import List, Dict, Optional, Tuple
import logging

from ..utils.config import Config
from .simple_retriever import SimpleHybridRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAnswerGenerator:
    """
    Generates answers from retrieved text chunks using TF-IDF search
    
    This is like a smart research assistant that:
    1. Gets relevant text chunks from search
    2. Finds the best answer within those chunks
    3. Provides proper citations
    4. Knows when to say "I don't know"
    """
    
    def __init__(self):
        """Initialize the answer generator"""
        self.retriever = SimpleHybridRetriever()
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
    
    def _extract_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences
        
        This breaks up a chunk of text into separate sentences
        so we can find the most relevant one for the answer.
        """
        # Simple sentence splitting (handles most cases)
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short fragments
                # Add back the period if it's missing
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_sentence_relevance(self, sentence: str, query: str) -> float:
        """
        Calculate how relevant a sentence is to the query
        
        This gives a score based on:
        1. How many query words appear in the sentence
        2. How rare/important those words are
        3. How close together the matching words are
        """
        sentence_lower = sentence.lower()
        query_words = query.lower().split()
        
        # Count word matches
        word_matches = 0
        total_query_words = len(query_words)
        
        for word in query_words:
            if len(word) > 2 and word in sentence_lower:  # Skip short words like "a", "is"
                word_matches += 1
        
        # Calculate basic relevance score
        if total_query_words == 0:
            return 0.0
        
        word_match_score = word_matches / total_query_words
        
        # Bonus for exact phrase matches
        query_phrases = [phrase.strip() for phrase in query.split() if len(phrase.strip()) > 3]
        phrase_bonus = 0
        
        for phrase in query_phrases:
            if phrase.lower() in sentence_lower:
                phrase_bonus += 0.2
        
        # Combine scores
        final_score = min(1.0, word_match_score + phrase_bonus)
        
        return final_score
    
    def _find_best_answer_sentence(self, chunks: List[Dict], query: str) -> Optional[Dict]:
        """
        Find the best sentence across all chunks to answer the query
        
        This looks through all the retrieved text chunks and finds
        the single sentence that best answers the question.
        """
        best_sentence = None
        best_score = 0.0
        best_chunk = None
        
        for chunk in chunks:
            text = chunk.get('text', '')
            sentences = self._extract_sentences(text)
            
            for sentence in sentences:
                relevance_score = self._calculate_sentence_relevance(sentence, query)
                
                # Combine with chunk score for final relevance
                chunk_score = chunk.get('final_score', 0.0)
                combined_score = 0.7 * relevance_score + 0.3 * chunk_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_sentence = sentence
                    best_chunk = chunk
        
        if best_sentence and best_score > 0.1:  # Minimum relevance threshold
            return {
                'sentence': best_sentence,
                'score': best_score,
                'chunk': best_chunk
            }
        
        return None
    
    def _create_citation(self, chunk: Dict) -> str:
        """
        Create a proper citation for a text chunk
        
        This creates a reference that tells the user exactly
        where the information came from.
        """
        file_name = chunk.get('file_name', 'Unknown')
        title = chunk.get('title', file_name)
        
        # Create a simple citation format
        citation = f"{title}"
        if file_name != title:
            citation += f" ({file_name})"
        
        return citation
    
    def _should_abstain(self, search_results: List[Dict], query: str) -> Tuple[bool, str]:
        """
        Decide whether to abstain from answering
        
        This determines when we should say "I don't know" instead
        of giving a potentially wrong answer.
        """
        if not search_results:
            return True, "No relevant documents found for your question."
        
        # Check if top result meets confidence threshold
        top_score = search_results[0].get('final_score', 0.0)
        if top_score < self.confidence_threshold:
            return True, f"I couldn't find a confident answer to your question. The best match had a confidence of {top_score:.2f}, which is below my threshold of {self.confidence_threshold}."
        
        # Check if there's a big gap between top results (indicates uncertainty)
        if len(search_results) > 1:
            second_score = search_results[1].get('final_score', 0.0)
            score_gap = top_score - second_score
            
            if score_gap < 0.1:  # Very close scores indicate uncertainty
                return True, "I found multiple possible answers with similar confidence levels. I'm not sure which is most accurate."
        
        return False, ""
    
    def generate_answer(self, query: str, top_k: int = 5, search_mode: str = 'hybrid') -> Dict:
        """
        Generate an answer to a question
        
        This is the main function that:
        1. Searches for relevant chunks
        2. Finds the best answer
        3. Creates citations
        4. Decides whether to answer or abstain
        """
        logger.info(f"Generating answer for: '{query}'")
        
        # Search for relevant chunks
        search_results = self.retriever.search(query, top_k=top_k, mode=search_mode)
        
        # Check if we should abstain
        should_abstain, abstain_reason = self._should_abstain(search_results, query)
        
        if should_abstain:
            return {
                'answer': None,
                'abstain_reason': abstain_reason,
                'confidence': 0.0,
                'citations': [],
                'chunks_considered': len(search_results),
                'search_mode': search_mode
            }
        
        # Find the best answer sentence
        best_answer = self._find_best_answer_sentence(search_results, query)
        
        if not best_answer:
            return {
                'answer': None,
                'abstain_reason': "I couldn't extract a clear answer from the relevant documents.",
                'confidence': 0.0,
                'citations': [],
                'chunks_considered': len(search_results),
                'search_mode': search_mode
            }
        
        # Create citation
        citation = self._create_citation(best_answer['chunk'])
        
        # Prepare context chunks for reference
        context_chunks = []
        for i, chunk in enumerate(search_results[:3]):  # Show top 3 contexts
            context_chunks.append({
                'rank': i + 1,
                'text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'score': chunk['final_score'],
                'source': self._create_citation(chunk)
            })
        
        return {
            'answer': best_answer['sentence'],
            'confidence': best_answer['score'],
            'citations': [citation],
            'chunks_considered': len(search_results),
            'context_chunks': context_chunks,
            'search_mode': search_mode,
            'abstain_reason': None
        }
    
    def batch_answer(self, questions: List[str]) -> List[Dict]:
        """
        Answer multiple questions at once
        
        This is useful for evaluation - you can test many questions
        at the same time.
        """
        answers = []
        for question in questions:
            answer = self.generate_answer(question)
            answers.append({
                'question': question,
                'answer_data': answer
            })
        return answers
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the answering system
        
        This shows the current configuration and capabilities.
        """
        search_stats = self.retriever.get_search_stats()
        
        return {
            'search_system_ready': search_stats['chunks_available'] > 0,
            'chunks_available': search_stats['chunks_available'],
            'confidence_threshold': self.confidence_threshold,
            'search_mode_supported': ['hybrid', 'vector', 'keyword']
        }


def main():
    """
    Test the answer generation system
    
    This demonstrates how the answering system works with example questions.
    """
    print("🤖 Testing Answer Generation System...")
    
    answerer = SimpleAnswerGenerator()
    
    # Show system status
    stats = answerer.get_stats()
    print(f"📊 System Status:")
    print(f"   - Search system ready: {stats['search_system_ready']}")
    print(f"   - Chunks available: {stats['chunks_available']}")
    print(f"   - Confidence threshold: {stats['confidence_threshold']}")
    
    if not stats['search_system_ready']:
        print("❌ Answer generation system not ready. Please:")
        print("   1. Process PDFs: python -m src.ingest.pdf_processor")
        print("   2. Generate embeddings: python -m src.ingest.simple_embedder")
        return
    
    # Test questions
    test_questions = [
        "What are the main types of industrial safety equipment?",
        "How should workers be protected from machine hazards?",
        "What is the purpose of risk assessment?",
        "What are lockout tagout procedures?",
        "Tell me about quantum physics"  # This should trigger abstention
    ]
    
    for question in test_questions:
        print(f"\\n❓ Question: {question}")
        
        # Generate answer
        result = answerer.generate_answer(question)
        
        if result['answer']:
            print(f"✅ Answer: {result['answer']}")
            print(f"📈 Confidence: {result['confidence']:.3f}")
            print(f"📚 Citation: {result['citations'][0]}")
            print(f"🔍 Search mode: {result['search_mode']}")
            print(f"📄 Chunks considered: {result['chunks_considered']}")
        else:
            print(f"❌ No answer provided")
            print(f"💭 Reason: {result['abstain_reason']}")
    
    print("\\n✨ Answer generation testing complete!")


if __name__ == "__main__":
    main()