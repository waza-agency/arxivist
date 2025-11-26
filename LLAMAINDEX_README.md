# ArXivist LlamaIndex Integration

This guide covers how to use LlamaIndex for document indexing and RAG-powered querying of your ArXiv paper collection.

## Overview

The LlamaIndex integration provides:
- Local document storage using ChromaDB or file-based persistence
- Local embeddings (HuggingFace) or OpenAI embeddings
- RAG-powered Q&A using Ollama (local) or OpenAI LLMs
- Interactive chat mode for exploring your papers

## Installation

### Basic Installation

```bash
pip install llama-index llama-index-embeddings-huggingface python-dotenv
```

### With ChromaDB (Recommended)

```bash
pip install chromadb llama-index-vector-stores-chroma
```

### With Ollama Support (Local LLM)

```bash
pip install llama-index-llms-ollama
```

### With OpenAI Support

```bash
pip install llama-index-llms-openai llama-index-embeddings-openai
```

### Full Installation (All Features)

```bash
pip install \
    llama-index \
    llama-index-embeddings-huggingface \
    llama-index-embeddings-openai \
    llama-index-vector-stores-chroma \
    llama-index-llms-ollama \
    llama-index-llms-openai \
    chromadb \
    python-dotenv
```

## Configuration

Create a `.env` file in your project root:

```env
# Embedding Configuration
USE_LOCAL_EMBEDDINGS=true

# OpenAI (if USE_LOCAL_EMBEDDINGS=false or using OpenAI LLM)
OPENAI_API_KEY=your_openai_api_key

# Ollama (if using local LLM)
OLLAMA_HOST=http://localhost:11434
```

## Loading Documents

### Basic Usage

Load PDF documents from `./papers` directory:

```bash
python load_documents_llamaindex.py
```

### Command Line Options

```bash
# Custom papers directory
python load_documents_llamaindex.py --papers-dir /path/to/pdfs

# Custom storage location
python load_documents_llamaindex.py --persist-dir ./my_storage

# Use local file storage instead of ChromaDB
python load_documents_llamaindex.py --no-chroma

# Larger batch size for faster processing
python load_documents_llamaindex.py --batch-size 10

# Use GPU for embeddings
python load_documents_llamaindex.py --device cuda

# Custom collection name
python load_documents_llamaindex.py --collection my_papers
```

### Programmatic Usage

```python
from load_documents_llamaindex import load_documents_llamaindex
from pathlib import Path

load_documents_llamaindex(
    papers_dir=Path("./papers"),
    persist_dir=Path("./llamaindex_storage"),
    batch_size=5,
    use_chroma=True,
    collection_name="arxivist",
    device="cpu"  # or "cuda" for GPU
)
```

## Querying Documents

### Interactive Chat Mode

Start the interactive agent:

```bash
python query_llamaindex.py
```

Chat commands:
- Type your question naturally to get RAG-powered answers
- `/search <query>` - Search only (no LLM processing)
- `/sources` - Show sources from last query
- `/help` - Show available commands
- `/quit` - Exit

### Single Query Mode

```bash
# Basic query
python query_llamaindex.py -q "What is the attention mechanism in transformers?"

# Get more results
python query_llamaindex.py -q "neural networks" -k 10

# Search only (no LLM)
python query_llamaindex.py --llm none -q "machine learning"
```

### Using Different LLMs

```bash
# Use Ollama (default)
python query_llamaindex.py --llm ollama --model llama3.2

# Use OpenAI
python query_llamaindex.py --llm openai --model gpt-4o-mini

# Disable LLM (search only)
python query_llamaindex.py --llm none
```

### Command Line Options

```bash
python query_llamaindex.py --help

Options:
  --query, -q       Query to answer
  --persist-dir     Directory containing the index
  --no-chroma       Use local file storage instead of ChromaDB
  --collection      ChromaDB collection name
  --llm             LLM to use: ollama, openai, none
  --model           Model name for the LLM
  --top-k, -k       Number of documents to retrieve
  --temperature, -t LLM temperature
  --device          Device for embeddings: cpu, cuda, mps
```

## Programmatic Usage

### Using the ArxivistAgent Class

```python
from query_llamaindex import ArxivistAgent
from pathlib import Path

# Initialize the agent
agent = ArxivistAgent(
    persist_dir=Path("./llamaindex_storage"),
    use_chroma=True,
    collection_name="arxivist",
    llm_type="ollama",  # or "openai" or "none"
    llm_model="llama3.2",
    use_local_embeddings=True,
    device="cpu",
    temperature=0.1,
    top_k=5
)

# Query with RAG
result = agent.query("What are the main contributions of BERT?")
print(result['answer'])
print(f"Based on {len(result['sources'])} sources")

# Search only (no LLM processing)
results = agent.search("transformer architecture", top_k=10)
for r in results:
    print(f"Score: {r['score']}, Title: {r['metadata'].get('title')}")
```

### Building Custom RAG Pipelines

```python
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

# Setup embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Setup LLM
Settings.llm = Ollama(model="llama3.2", temperature=0.1)

# Load from ChromaDB
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

chroma_client = chromadb.PersistentClient(path="./llamaindex_storage/chroma")
chroma_collection = chroma_client.get_collection("arxivist")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store)

# Create query engine with custom settings
query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_mode="tree_summarize"  # or "compact", "refine", etc.
)

response = query_engine.query("Explain transformer self-attention")
print(response)
```

### Creating a Chat Engine

```python
from llama_index.core.memory import ChatMemoryBuffer

# Setup memory for conversation history
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Create chat engine
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="You are a helpful research assistant specializing in AI papers."
)

# Have a conversation
response = chat_engine.chat("What papers discuss attention mechanisms?")
print(response)

response = chat_engine.chat("Tell me more about the first one")
print(response)
```

## Storage Options

### ChromaDB (Recommended)

ChromaDB provides efficient vector storage with persistence:

```python
# Default location: ./llamaindex_storage/chroma/
```

Advantages:
- Fast similarity search
- Persistent storage
- Supports metadata filtering

### Local File Storage

Uses LlamaIndex's built-in file persistence:

```python
# Default location: ./llamaindex_storage/index/
python load_documents_llamaindex.py --no-chroma
```

Advantages:
- No additional dependencies
- Simple file-based storage

## Embedding Options

### Local Embeddings (Default)

Uses HuggingFace sentence-transformers:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Free and runs locally

```bash
USE_LOCAL_EMBEDDINGS=true
```

### OpenAI Embeddings

For higher quality embeddings:
- Model: `text-embedding-3-small`
- Dimension: 1536
- Requires API key

```bash
USE_LOCAL_EMBEDDINGS=false
OPENAI_API_KEY=your_key
```

## LLM Options

### Ollama (Default - Local)

Run LLMs locally with Ollama:

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Use in queries:

```bash
python query_llamaindex.py --llm ollama --model llama3.2
```

Other good models:
- `mistral` - Fast and capable
- `mixtral` - Larger, more capable
- `phi3` - Small and fast
- `codellama` - Good for code-related papers

### OpenAI

Use OpenAI's GPT models:

```bash
python query_llamaindex.py --llm openai --model gpt-4o-mini
```

Models:
- `gpt-4o-mini` - Fast and cost-effective
- `gpt-4o` - Most capable

### No LLM (Search Only)

For pure vector similarity search:

```bash
python query_llamaindex.py --llm none
```

## Directory Structure

After loading documents:

```
llamaindex_storage/
├── chroma/                    # ChromaDB files (if using ChromaDB)
│   ├── chroma.sqlite3
│   └── ...
├── index/                     # Local index files (if not using ChromaDB)
│   ├── docstore.json
│   ├── index_store.json
│   └── vector_store.json
└── loaded_documents.json      # Tracking file for idempotent loading
```

## Tips

### GPU Acceleration

For faster embedding generation:

```bash
# Using CUDA
python load_documents_llamaindex.py --device cuda

# Using Apple Silicon
python load_documents_llamaindex.py --device mps
```

### Batch Size Tuning

- Larger batches = faster but more memory
- Smaller batches = safer for problematic PDFs

```bash
# Fast processing
python load_documents_llamaindex.py --batch-size 20

# Safe processing
python load_documents_llamaindex.py --batch-size 1
```

### Reindexing

To reindex all documents:

```bash
# Remove tracking file and storage
rm -rf llamaindex_storage/
python load_documents_llamaindex.py
```

## Troubleshooting

### "Index not found" Error

Run the loader first:
```bash
python load_documents_llamaindex.py
```

### ChromaDB Not Available

Install ChromaDB:
```bash
pip install chromadb llama-index-vector-stores-chroma
```

Or use local storage:
```bash
python load_documents_llamaindex.py --no-chroma
```

### Ollama Connection Error

1. Make sure Ollama is running: `ollama serve`
2. Check the model is downloaded: `ollama list`
3. Pull if needed: `ollama pull llama3.2`

### Out of Memory

- Reduce batch size: `--batch-size 1`
- Use CPU instead of GPU: `--device cpu`
- Use a smaller embedding model
