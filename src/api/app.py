"""
Flask REST API for Industrial Safety QA System
Web service for querying industrial safety documents.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import traceback
from datetime import datetime
import os

from ..utils.config import Config
from ..search.simple_answerer import SimpleAnswerGenerator
from ..ingest.pdf_processor import PDFProcessor
from ..ingest.simple_embedder import SimpleEmbeddingGenerator

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            static_folder='../../static',
            static_url_path='/static')
CORS(app)

# Global components
answerer = None
pdf_processor = None
embedder = None

# Initialize components immediately
init_components()

def init_components():
    """Initialize system components"""
    global answerer, pdf_processor, embedder

    try:
        logger.info("Initializing QA Service components")

        # Initialize answerer with error handling
        try:
            answerer = SimpleAnswerGenerator()
            logger.info("Answerer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize answerer: {e}")
            answerer = None

        # Initialize PDF processor with error handling
        try:
            pdf_processor = PDFProcessor()
            logger.info("PDF processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PDF processor: {e}")
            pdf_processor = None

        # Initialize embedder with error handling
        try:
            embedder = SimpleEmbeddingGenerator()
            logger.info("Embedder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            embedder = None

        # Check if all components initialized successfully
        if answerer and pdf_processor and embedder:
            logger.info("All components initialized successfully")
        else:
            logger.warning("Some components failed to initialize. Check logs above.")

    except Exception as e:
        logger.error(f"Unexpected error during component initialization: {e}")
        traceback.print_exc()

@app.route('/')
def home():
    """Serve the web interface"""
    try:
        static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static')
        return send_from_directory(static_dir, 'index.html')
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        return f"Error loading web interface: {e}", 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Process question and return answer"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing required field: question',
                'example': {'question': 'What is industrial safety?'}
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        # Get parameters
        top_k = data.get('k', 5)
        search_mode = data.get('search_mode', data.get('mode', 'hybrid'))
        
        # Map frontend modes to backend modes
        if search_mode == 'keyword':
            search_mode = 'bm25'
        
        # Validate search mode
        if search_mode not in ['hybrid', 'vector', 'bm25']:
            return jsonify({
                'error': 'Invalid search mode. Use: hybrid, vector, or bm25'
            }), 400
        
        if not answerer:
            return jsonify({
                'error': 'QA system not initialized. Please check server logs.'
            }), 503
        
        # Generate answer
        logger.info(f"Processing question: {question[:100]}...")
        result = answerer.generate_answer(question, top_k=top_k, search_mode=search_mode)
        
        # Format response
        response = {
            'question': question,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'citations': result['citations'],
            'reranker_used': search_mode != 'vector',
            'search_mode': result['search_mode'],
            'chunks_considered': result['chunks_considered'],
            'timestamp': datetime.now().isoformat(),
            'context_chunks': result.get('context_chunks', [])
        }
        
        # Add abstention info if applicable
        if result['abstain_reason']:
            response['abstain_reason'] = result['abstain_reason']
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({
            'error': 'Internal server error while processing question',
            'details': str(e) if app.debug else 'Check server logs for details'
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status information"""
    try:
        status = {
            'api_status': 'running',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check answerer status
        if answerer:
            answerer_stats = answerer.get_stats()
            status['components']['answerer'] = {
                'status': 'ready' if answerer_stats['search_system_ready'] else 'not_ready',
                'chunks_available': answerer_stats['chunks_available'],
                'confidence_threshold': answerer_stats['confidence_threshold']
            }
        else:
            status['components']['answerer'] = {'status': 'not_initialized'}
        
        # Check PDF processor status
        if pdf_processor:
            pdf_stats = pdf_processor.get_processing_status()
            status['components']['pdf_processor'] = {
                'status': 'ready',
                'pdfs_available': pdf_stats['pdf_files_available'],
                'documents_processed': pdf_stats['documents_processed']
            }
        else:
            status['components']['pdf_processor'] = {'status': 'not_initialized'}
        
        # Check embedder status
        if embedder:
            embedding_stats = embedder.get_embedding_stats()
            status['components']['embedder'] = {
                'status': 'ready',
                'total_chunks': embedding_stats['total_chunks'],
                'embedded_chunks': embedding_stats['embedded_chunks'],
                'coverage': embedding_stats['coverage_percentage']
            }
        else:
            status['components']['embedder'] = {'status': 'not_initialized'}
        
        # Overall system health
        all_ready = all(
            comp.get('status') == 'ready' 
            for comp in status['components'].values()
        )
        status['overall_status'] = 'healthy' if all_ready else 'partial'
        
        # Add frontend-friendly fields
        if answerer:
            answerer_stats = answerer.get_stats()
            status['chunks_indexed'] = answerer_stats['chunks_available']
        else:
            status['chunks_indexed'] = 0
            
        if pdf_processor:
            pdf_stats = pdf_processor.get_processing_status()
            status['documents_count'] = pdf_stats['documents_processed']
        else:
            status['documents_count'] = 0
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            'error': 'Failed to get system status',
            'details': str(e) if app.debug else 'Check server logs'
        }), 500

@app.route('/ingest', methods=['POST'])
def trigger_ingestion():
    """Trigger PDF ingestion and embedding generation"""
    try:
        # Simple authentication
        auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if auth_token != 'simple_ingest_token_123':
            return jsonify({
                'error': 'Authentication required',
                'hint': 'Include Authorization: Bearer simple_ingest_token_123 header'
            }), 401
        
        if not pdf_processor or not embedder:
            return jsonify({
                'error': 'Ingestion components not initialized'
            }), 503
        
        logger.info("Starting manual ingestion process")
        
        # Process PDFs
        pdf_results = pdf_processor.process_all_pdfs()
        
        # Generate embeddings
        embedding_results = embedder.process_chunks_without_embeddings()
        
        # Reload answerer data
        if answerer:
            answerer.retriever.reload_data()
        
        response = {
            'message': 'Ingestion completed',
            'pdf_processing': pdf_results,
            'embedding_generation': embedding_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        return jsonify({
            'error': 'Ingestion failed',
            'details': str(e) if app.debug else 'Check server logs'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/ask (POST)', '/status (GET)', '/ingest (POST)', '/ (GET)']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong. Check server logs for details.'
    }), 500

def main():
    """Start the Flask web server"""
    print("Starting QA Service API")
    
    # Initialize components
    init_components()
    
    # Show startup information
    print(f"Server Configuration:")
    print(f"   - Host: {Config.FLASK_HOST}")
    print(f"   - Port: {Config.FLASK_PORT}")
    print(f"   - Debug: {Config.FLASK_DEBUG}")
    
    if answerer:
        stats = answerer.get_stats()
        print(f"   - Chunks available: {stats['chunks_available']}")
        print(f"   - System ready: {stats['search_system_ready']}")
    
    print(f"API Endpoints:")
    print(f"   - Home: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/")
    print(f"   - Ask: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/ask (POST)")
    print(f"   - Status: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/status (GET)")
    
    print("Example usage:")
    print(f'   curl -X POST http://{Config.FLASK_HOST}:{Config.FLASK_PORT}/ask \\')
    print('   -H "Content-Type: application/json" \\')
    print('   -d \'{"question": "What is industrial safety?"}\'')
    
    # Start the server
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG
    )

if __name__ == "__main__":
    main()
