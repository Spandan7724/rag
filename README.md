# Bajaj Hackrx 

## Features

### Core Capabilities
- **Intelligent Challenge Detection**: Automatically detects and routes specialized question types (geographic puzzles, flight numbers, secret tokens, web scraping)
- **Multi-Format Document Processing**: Supports PDF, Word, Excel, PowerPoint, images, and text files
- **Advanced RAG Pipeline**: BGE-M3 embeddings with FAISS vector storage and persistent caching
- **Hybrid Processing**: Smart page-level processing with OCR fallback for complex documents
- **Query Transformation**: Multi-query decomposition for complex questions
- **Multilingual Support**: Cross-language semantic matching and processing

### Document Processing
- **Multi-Format Support**: PDF, Word (.docx/.doc), Excel (.xlsx/.xls), PowerPoint (.pptx/.ppt), text files, and images
- **PDF Processing**: Text extraction, OCR, table detection, and image processing
- **Office Documents**: Native support for Microsoft Office formats with content extraction
- **Image Processing**: OCR-based text extraction from images (PNG, JPEG, GIF, BMP, TIFF, WebP)
- **Table Extraction**: Specialized extraction for landmark-to-city mappings and structured data
- **Web Scraping**: Integration with HackRX API endpoints and general web content extraction
- **File Upload Management**: Temporary storage with configurable retention policies
- **Blob Caching**: File caching to prevent re-downloading and reprocessing

### Architecture Highlights
- **Individual Question Processing**: Reliable processing without race conditions
- **Challenge-Aware Routing**: Specialized handlers for different question types
- **Persistent Vector Storage**: FAISS indices with disk persistence
- **GPU Acceleration**: CUDA support for embeddings and OCR
- **Comprehensive Logging**: Detailed question and response logging with metadata

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Contributing](#contributing)

##  Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (optional, for acceleration)
- Git
- uv (recommended) or pip

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd bajaj
   ```

2. **Set up Python environment**:
   ```bash
   # Using uv (recommended)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install uv
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Install system dependencies** (Ubuntu/Debian):
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
     poppler-utils \
     tesseract-ocr \
     tesseract-ocr-eng \
     libopencv-dev \
     libmagic1
   ```

##  Quick Start

### Development Server
```bash
# Preferred method
python run.py

# Alternative using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker (Recommended for Production)
```bash
# Using Docker Compose
docker-compose up -d

# Or build manually
docker build -t bajaj-rag:latest .
docker run -d -p 8000:8000 --name bajaj-rag bajaj-rag:latest
```

### Basic Usage
Once running, the API will be available at `http://localhost:8000`

**Example Request**:
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 8915ddf1d1760f2b6a3b027c6fa7b16d2d87a042c41452f49a1d43b3cfa6245b" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What are the coverage limits?"
    ]
  }'
```

##  Configuration

### LLM Provider Configuration

The system supports multiple LLM providers for answer generation:

#### GitHub Copilot (Default - Recommended)

The system leverages GitHub Copilot's OpenAI-compatible API endpoint (`https://api.githubcopilot.com`) to access advanced models including Claude Sonnet 4 and GPT-4.1. This is implemented through custom code in `app/services/copilot_provider.py` which treats the Copilot endpoint like an OpenAI endpoint.

**Setup Steps:**

1. **Authenticate with GitHub CLI**:
   ```bash
   gh auth login
   ```

2. **Install Copilot CLI extension**:
   ```bash
   gh extension install github/gh-copilot
   ```

3. **Get your Copilot access token**:
   After installation, Copilot auto-generates `~/.config/github-copilot/apps.json` containing your access token.

4. **Configure environment**:
   ```bash
   export LLM_PROVIDER=copilot
   export LLM_MODEL=claude-sonnet-4  # or gpt-4.1-2025-04-14, gpt-4o, etc.
   export COPILOT_ACCESS_TOKEN=your_copilot_token_from_apps_json
   ```

**Available Copilot Models:**
- `claude-sonnet-4` - Latest Claude Sonnet 4 model (via Copilot)
- `gpt-4.1-2025-04-14` - Latest GPT-4.1 model (via Copilot)
- `gpt-4o` - GPT-4 Omni model


#### OpenAI Direct API

For direct OpenAI API access (standard implementation):

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o  # or gpt-4o-mini, gpt-4-turbo, etc.
export OPENAI_API_KEY=your_openai_api_key_here
```

**Available OpenAI Models:**
- `gpt-4o` - Latest GPT-4 Omni model
- `gpt-4o-mini` - Faster, cost-effective GPT-4 Omni
- `gpt-4-turbo` - GPT-4 Turbo with improved performance
- `gpt-4` - Standard GPT-4 model
- `gpt-3.5-turbo` - Cost-effective for simpler tasks
- `o1-preview` - Advanced reasoning model
- `o1-mini` - Faster reasoning model

#### Why Use GitHub Copilot?

The GitHub Copilot integration provides several advantages:
- **Access to Claude Models**: Use Claude Sonnet 4 through the Copilot API
- **Latest GPT Models**: Access to GPT-4.1 and other cutting-edge models
- **Unified API**: Single endpoint for multiple model providers
- **Enterprise Features**: Enhanced rate limits and enterprise support (if available)

#### Technical Implementation

The custom Copilot provider (`app/services/copilot_provider.py`) implements:
- **OpenAI-Compatible Interface**: Translates OpenAI API calls to Copilot API format
- **Model Mapping**: Maps model names to Copilot-supported models
- **Authentication Handling**: Uses Copilot access tokens for authentication
- **Response Formatting**: Ensures consistent response format across providers

This allows seamless switching between providers without changing the core RAG logic.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `copilot` | LLM provider (`copilot` or `openai`) |
| `LLM_MODEL` | `gpt-4.1-2025-04-14` | Model for answer generation |
| `COPILOT_ACCESS_TOKEN` | - | Required for Copilot provider |
| `OPENAI_API_KEY` | - | Required for OpenAI provider |
| `DEBUG` | `false` | Enable debug mode |
| `BEARER_TOKEN` | Generated | API authentication token |
| `MAX_FILE_SIZE` | `524288000` | Maximum file size (500MB) |
| `ENABLE_QUESTION_LOGGING` | `true` | Enable query logging |

### Core Settings

Key configuration options in `app/core/config.py`:

- **Embedding Model**: `BAAI/bge-m3` (BGE-M3 with 8192 token limit)
- **Chunk Size**: 450 tokens with 100 token overlap
- **Retrieval**: k=35 chunks with similarity threshold 1.5
- **OCR Provider**: RapidOCR with GPU acceleration
- **Vector Storage**: Persistent FAISS indices

##  API Documentation

### Main Endpoints

#### POST `/hackrx/run`
Complete RAG pipeline with document processing and Q&A
- **Input**: Document URL/file and list of questions
- **Output**: Answers with sources and metadata
- **Features**: Challenge detection, query transformation, caching

#### POST `/upload`
Upload files for processing
- **Input**: Multipart file upload
- **Output**: File ID for later reference
- **Supports**: PDF, Word (.docx/.doc), Excel (.xlsx/.xls), PowerPoint (.pptx/.ppt), text files (.txt/.csv/.tsv), images (.png/.jpg/.gif/.bmp/.tiff/.webp)

#### GET `/health`
Health check endpoint
- **Output**: System status and component health

### Debug Endpoints

- `GET /debug/system-stats` - System statistics and performance metrics
- `GET /debug/documents` - List of processed documents
- `GET /debug/directories` - Directory status and validation
- `GET /debug/file-manager` - File manager statistics

### File Management

- `GET /uploads` - List uploaded files
- `GET /uploads/{file_id}` - Get file information
- `DELETE /uploads/{file_id}` - Delete uploaded file

##  Architecture

### Challenge Detection System

The system uses intelligent challenge detection to route questions:

```
Question Input → Challenge Detection → Route Decision:
├── SECRET_TOKEN → Web Client → HackRX API → Token Response
├── GEOGRAPHIC_PUZZLE → Challenge Solver → Multi-step → Flight Number
├── FLIGHT_NUMBER → Direct Extraction OR Geographic Fallback
├── WEB_SCRAPING → General web content extraction
└── STANDARD_RAG → Vector Search → LLM Generation → Answer
```

### Document Processing Pipeline

```
Document URL → Type Detection:
├── API Endpoint → Skip Processing → Mark for Web Scraping
└── Document File → Processing Pipeline:
    ├── Geographic Content → Direct Pipeline → Landmark Mapping
    └── Standard Content → RAG Pipeline → Chunking → Vector Storage
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **RAG Coordinator** | Main orchestrator with individual question processing |
| **Challenge Detector** | Pattern-based question classification |
| **Document Processor** | Multi-format document processing with OCR |
| **Vector Store** | FAISS-based storage with persistence |
| **Embedding Manager** | BGE-M3 embeddings with GPU acceleration |
| **Query Transformer** | Multi-query decomposition for complex questions |
| **Web Client** | HackRX API integration and web scraping |

##  Docker Deployment

### Quick Deployment
```bash
# Clone and configure
git clone <repository-url>
cd bajaj
cp .env.example .env
# Edit .env with your API keys

# Deploy with Docker Compose
docker-compose up -d
```

### Production Configuration
```yaml
# docker-compose.yml
services:
  bajaj-rag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=copilot
      - COPILOT_ACCESS_TOKEN=${COPILOT_ACCESS_TOKEN}
      - DEBUG=false
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### GPU Support
```bash
# Enable GPU acceleration
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  bajaj-rag:latest
```

##  Development

### Project Structure
```
bajaj/
├── app/
│   ├── api/                 # FastAPI routes
│   ├── core/                # Configuration and security
│   ├── models/              # Pydantic models
│   ├── services/            # Core business logic
│   │   ├── extractors/      # Document format extractors
│   │   ├── challenge_detector.py
│   │   ├── rag_coordinator.py
│   │   ├── document_processor.py
│   │   └── ...
│   └── utils/               # Utility functions
├── data/                    # Persistent data storage
├── docker-compose.yml       # Docker deployment
├── Dockerfile              # Multi-stage container build
├── pyproject.toml          # Python dependencies
└── run.py                  # Application entry point
```

### Development Commands
```bash
# Start development server
python run.py

# Install new dependency
uv add package-name

# Run with debug mode
DEBUG=true python run.py

# Test API endpoints
curl http://localhost:8000/health
```

### Key Development Notes

1. **Individual Processing**: The system processes questions individually to prevent race conditions
2. **Challenge Thresholds**: Confidence thresholds are tuned for reliable classification
3. **GPU Acceleration**: Enabled for embeddings and OCR when available
4. **Caching**: Multiple layers of caching for performance optimization
5. **Error Handling**: Comprehensive error handling with graceful degradation

## Dependencies

### Core ML/AI Libraries
- **torch** - PyTorch for BGE-M3 embeddings
- **flagembedding** - BGE-M3 embedding model
- **faiss-cpu** - Vector similarity search and indexing
- **sentence-transformers** - Reranking and answer generation

### Document Processing
- **pymupdf** - Primary PDF text extraction
- **rapidocr-onnxruntime** - OCR with GPU acceleration
- **pdfplumber** - Table extraction for landmark mappings
- **python-docx** - Word document processing
- **openpyxl** - Excel file processing

### API and Web
- **fastapi** - Web framework for API endpoints
- **uvicorn** - ASGI server for development and deployment
- **httpx** - Async HTTP client for web scraping
- **aiohttp** - Additional HTTP client support

## Performance Features

- **GPU Acceleration**: CUDA support for embeddings and OCR
- **Persistent Caching**: Vector indices and document blobs
- **Parallel Processing**: Multi-threaded document processing
- **Memory Optimization**: Efficient chunk processing and storage
- **Request Batching**: Optimized embedding generation

##  Security

- **Bearer Token Authentication**: Required for all API endpoints
- **File Size Limits**: Configurable upload size restrictions
- **File Type Validation**: Whitelist-based file type checking
- **Temporary Storage**: Automatic cleanup of uploaded files
- **Non-root Container**: Docker containers run as non-privileged user

##  Monitoring and Logging

- **Health Checks**: Built-in health monitoring endpoints
- **Performance Metrics**: Detailed timing and statistics
- **Question Logging**: Complete query and response logging
- **Debug Endpoints**: System introspection and diagnostics
- **Directory Management**: Automatic directory validation
