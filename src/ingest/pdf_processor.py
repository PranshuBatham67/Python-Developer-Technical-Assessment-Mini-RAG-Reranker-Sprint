"""
PDF Processing and Text Chunking for QA Service

This file reads PDF files and breaks them into small, manageable pieces.
Think of it as taking a big book and creating a summary card for each paragraph.
"""

import os
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
from tqdm import tqdm
import logging

from ..utils.config import Config
from ..utils.database import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Processes PDF files and breaks them into chunks
    
    This is like a smart document scanner that:
    1. Reads PDF files
    2. Extracts the text 
    3. Splits text into meaningful paragraphs
    4. Stores everything in our database
    """
    
    def __init__(self):
        """Initialize the PDF processor"""
        self.db = DatabaseManager()
        self.sources_data = self._load_sources_json()
    
    def _load_sources_json(self) -> Dict:
        """
        Load the sources.json file with document metadata
        
        This file contains information about each PDF like title and URL.
        It's like a library catalog that describes each book.
        """
        if Config.SOURCES_JSON.exists():
            with open(Config.SOURCES_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create empty sources structure
            return {"documents": {}}
    
    def _save_sources_json(self):
        """
        Save the updated sources.json file
        
        This updates our library catalog with new information.
        """
        with open(Config.SOURCES_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.sources_data, f, indent=2, ensure_ascii=False)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate a unique hash for a file
        
        This creates a unique fingerprint for each PDF file.
        Even if two files have the same name, they'll have different fingerprints.
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _extract_text_from_pdf(self, file_path: Path) -> Optional[str]:
        """
        Extract text from a PDF file
        
        This is like reading through a PDF page by page 
        and typing out all the text content.
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text_content.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
                        continue
                
                return "\n\n".join(text_content)
        
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        This fixes common issues with PDF text extraction:
        - Removes extra spaces
        - Fixes broken lines
        - Normalizes whitespace
        """
        if not text:
            return ""
        
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix broken words (e.g., "hel- lo" -> "hello")
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        
        # Remove extra newlines but keep paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _split_into_chunks(self, text: str, min_size: int = None, max_size: int = None) -> List[str]:
        """
        Split text into manageable chunks
        
        This breaks up long documents into paragraph-sized pieces.
        Think of it as creating index cards from a long essay.
        """
        min_size = min_size or Config.MIN_CHUNK_SIZE
        max_size = max_size or Config.MAX_CHUNK_SIZE
        
        # First, split by double newlines (paragraph breaks)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Count words in paragraph
            paragraph_words = len(paragraph.split())
            current_words = len(current_chunk.split()) if current_chunk else 0
            
            # If adding this paragraph would exceed max size, save current chunk
            if current_chunk and (current_words + paragraph_words) > max_size:
                if current_words >= min_size:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            
            # If paragraph itself is too big, split it further
            elif paragraph_words > max_size:
                # Save current chunk if it exists
                if current_chunk and current_words >= min_size:
                    chunks.append(current_chunk.strip())
                
                # Split the big paragraph by sentences
                sentences = re.split(r'[.!?]+', paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_words = len(sentence.split())
                    temp_words = len(temp_chunk.split()) if temp_chunk else 0
                    
                    if temp_chunk and (temp_words + sentence_words) > max_size:
                        if temp_words >= min_size:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                    else:
                        temp_chunk = temp_chunk + ". " + sentence if temp_chunk else sentence
                
                # Save the remaining temp chunk
                if temp_chunk and len(temp_chunk.split()) >= min_size:
                    chunks.append(temp_chunk.strip())
                
                current_chunk = ""
            
            else:
                # Add paragraph to current chunk
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk.split()) >= min_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_single_pdf(self, file_path: Path) -> bool:
        """
        Process a single PDF file
        
        This is the main function that:
        1. Reads the PDF
        2. Extracts text
        3. Splits into chunks
        4. Stores in database
        """
        logger.info(f"Processing {file_path.name}")
        
        # Calculate file hash to check if already processed
        file_hash = self._calculate_file_hash(file_path)
        
        # Skip if already processed
        if self.db.file_exists(file_hash):
            logger.info(f"File {file_path.name} already processed, skipping")
            return True
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(file_path)
        if not text:
            logger.error(f"Failed to extract text from {file_path.name}")
            return False
        
        # Clean the text
        text = self._clean_text(text)
        
        # Get document metadata from sources.json
        doc_info = self.sources_data.get("documents", {}).get(file_path.name, {})
        title = doc_info.get("title", file_path.stem)
        url = doc_info.get("url", "")
        
        # Add document to database
        doc_id = self.db.add_document(
            file_name=file_path.name,
            title=title,
            url=url,
            file_hash=file_hash
        )
        
        # Split text into chunks
        chunks = self._split_into_chunks(text)
        
        # Store each chunk in database
        logger.info(f"Storing {len(chunks)} chunks for {file_path.name}")
        
        for i, chunk_text in enumerate(chunks):
            self.db.add_chunk(
                doc_id=doc_id,
                chunk_index=i,
                text=chunk_text
            )
        
        # Update document chunk count
        self.db.update_document_chunk_count(doc_id)
        
        # Update sources.json if this was a new document
        if file_path.name not in self.sources_data.get("documents", {}):
            if "documents" not in self.sources_data:
                self.sources_data["documents"] = {}
            
            self.sources_data["documents"][file_path.name] = {
                "title": title,
                "url": url or f"file://{file_path}",
                "description": f"Industrial safety document with {len(chunks)} text chunks",
                "processed_at": str(datetime.now()),
                "chunk_count": len(chunks)
            }
            
            self._save_sources_json()
        
        logger.info(f"Successfully processed {file_path.name} -> {len(chunks)} chunks")
        return True
    
    def process_all_pdfs(self) -> Dict[str, int]:
        """
        Process all PDF files in the pdfs directory
        
        This goes through all PDF files and processes each one.
        It's like going through a stack of documents and filing each one.
        """
        pdf_files = list(Config.PDFS_DIR.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {Config.PDFS_DIR}")
            return {"processed": 0, "failed": 0, "skipped": 0}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        stats = {"processed": 0, "failed": 0, "skipped": 0}
        
        # Process each PDF file
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                file_hash = self._calculate_file_hash(pdf_file)
                
                if self.db.file_exists(file_hash):
                    logger.info(f"Skipping {pdf_file.name} (already processed)")
                    stats["skipped"] += 1
                else:
                    success = self.process_single_pdf(pdf_file)
                    if success:
                        stats["processed"] += 1
                    else:
                        stats["failed"] += 1
                        
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                stats["failed"] += 1
        
        logger.info(f"Processing complete: {stats}")
        return stats
    
    def get_processing_status(self) -> Dict:
        """
        Get current processing status
        
        This shows a summary of what's been processed so far.
        """
        db_stats = self.db.get_stats()
        pdf_count = len(list(Config.PDFS_DIR.glob("*.pdf")))
        
        return {
            "pdf_files_available": pdf_count,
            "documents_processed": db_stats["documents"],
            "total_chunks": db_stats["total_chunks"],
            "embedded_chunks": db_stats["embedded_chunks"],
            "processing_complete": db_stats["documents"] == pdf_count
        }


def main():
    """
    Main function to run PDF processing
    
    This is what gets called when you run this file directly.
    """
    
    print("🔄 Starting PDF Processing...")
    
    processor = PDFProcessor()
    
    # Show current status
    status = processor.get_processing_status()
    print(f"📊 Current Status:")
    print(f"   - PDF files available: {status['pdf_files_available']}")
    print(f"   - Documents processed: {status['documents_processed']}")
    print(f"   - Total text chunks: {status['total_chunks']}")
    
    if status['processing_complete']:
        print("✅ All PDFs already processed!")
    else:
        # Process PDFs
        results = processor.process_all_pdfs()
        print(f"\\n📈 Processing Results:")
        print(f"   - Successfully processed: {results['processed']}")
        print(f"   - Failed: {results['failed']}")
        print(f"   - Skipped (already done): {results['skipped']}")
        
        # Show final status
        final_status = processor.get_processing_status()
        print(f"\\n📊 Final Status:")
        print(f"   - Total documents: {final_status['documents_processed']}")
        print(f"   - Total chunks: {final_status['total_chunks']}")
    
    print("\\n✨ PDF processing complete!")
    print("💡 Next step: Run embedding generation to convert text to vectors")


if __name__ == "__main__":
    main()