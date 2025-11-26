#!/usr/bin/env python3
"""
ArXivist Query Agent - RAG-powered Q&A agent using LlamaIndex.
Supports both local LLMs (Ollama) and OpenAI for question answering.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Optional imports
try:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from llama_index.llms.ollama import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from llama_index.llms.openai import OpenAI as OpenAILLM
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def get_env_variable(var_name: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(var_name, default)
    if not value and default is None:
        raise ValueError(f"Environment variable {var_name} is required")
    return value


def setup_embeddings(use_local: bool = True, device: str = "cpu"):
    """Setup embedding model."""
    if use_local:
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_length=512,
            device=device
        )
    else:
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimensions=1536
        )

    Settings.embed_model = embed_model
    return embed_model


def setup_llm(
    llm_type: str = "ollama",
    model_name: Optional[str] = None,
    temperature: float = 0.1
):
    """Setup LLM for RAG responses."""

    if llm_type == "ollama":
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not installed. Run: pip install llama-index-llms-ollama")

        model = model_name or "llama3.2"
        llm = Ollama(
            model=model,
            request_timeout=120.0,
            temperature=temperature
        )
        print(f"Using Ollama with model: {model}")

    elif llm_type == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not installed. Run: pip install llama-index-llms-openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required for OpenAI")

        model = model_name or "gpt-4o-mini"
        llm = OpenAILLM(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
        print(f"Using OpenAI with model: {model}")

    elif llm_type == "none":
        llm = None
        print("LLM disabled - using search-only mode")

    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    Settings.llm = llm
    return llm


def load_chroma_index(persist_dir: Path, collection_name: str = "arxivist"):
    """Load index from ChromaDB."""
    if not CHROMA_AVAILABLE:
        raise ImportError("ChromaDB not installed")

    chroma_path = persist_dir / "chroma"
    if not chroma_path.exists():
        raise FileNotFoundError(f"ChromaDB storage not found at {chroma_path}")

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        show_progress=False
    )

    return index


def load_local_index(persist_dir: Path):
    """Load index from local file storage."""
    index_path = persist_dir / "index"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path}")

    storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
    index = load_index_from_storage(storage_context)

    return index


class ArxivistAgent:
    """RAG Agent for querying ArXiv documents."""

    def __init__(
        self,
        persist_dir: Path = Path("./llamaindex_storage"),
        use_chroma: bool = True,
        collection_name: str = "arxivist",
        llm_type: str = "ollama",
        llm_model: Optional[str] = None,
        use_local_embeddings: bool = True,
        device: str = "cpu",
        temperature: float = 0.1,
        top_k: int = 5
    ):
        """Initialize the ArXivist agent."""
        load_dotenv()

        self.persist_dir = Path(persist_dir)
        self.top_k = top_k
        self.llm_type = llm_type

        # Setup embeddings
        print("Setting up embeddings...")
        setup_embeddings(use_local_embeddings, device)

        # Setup LLM
        print("Setting up LLM...")
        self.llm = setup_llm(llm_type, llm_model, temperature)

        # Load index
        print(f"Loading index from {persist_dir}...")
        if use_chroma:
            self.index = load_chroma_index(persist_dir, collection_name)
        else:
            self.index = load_local_index(persist_dir)
        print("✓ Index loaded successfully")

        # Create query engine
        if self.llm is not None:
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="compact"
            )
        else:
            self.query_engine = None

        # Create retriever for search-only mode
        self.retriever = self.index.as_retriever(similarity_top_k=top_k)

    def query(self, question: str) -> Dict[str, Any]:
        """Query the knowledge base."""
        if self.query_engine is not None:
            # RAG mode with LLM
            response = self.query_engine.query(question)

            # Extract source documents
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    sources.append({
                        'text': node.text[:500] + "..." if len(node.text) > 500 else node.text,
                        'score': node.score if hasattr(node, 'score') else None,
                        'metadata': node.metadata
                    })

            return {
                'answer': str(response),
                'sources': sources,
                'mode': 'rag'
            }
        else:
            # Search-only mode
            nodes = self.retriever.retrieve(question)

            sources = []
            for node in nodes:
                sources.append({
                    'text': node.text[:500] + "..." if len(node.text) > 500 else node.text,
                    'score': node.score if hasattr(node, 'score') else None,
                    'metadata': node.metadata
                })

            return {
                'answer': None,
                'sources': sources,
                'mode': 'search'
            }

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents without LLM processing."""
        k = top_k or self.top_k
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                'text': node.text,
                'score': node.score if hasattr(node, 'score') else None,
                'metadata': node.metadata
            })

        return results

    def chat(self):
        """Interactive chat mode."""
        print("\nArXivist Chat Mode")
        print("=" * 40)
        print("Commands:")
        print("  /search <query>  - Search only (no LLM)")
        print("  /sources         - Show sources from last query")
        print("  /help            - Show this help")
        print("  /quit            - Exit")
        print("=" * 40)

        last_sources = []

        while True:
            try:
                user_input = input("\n📚 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['/quit', '/exit', '/q']:
                    print("Goodbye!")
                    break

                if user_input.lower() == '/help':
                    print("Commands: /search, /sources, /help, /quit")
                    continue

                if user_input.lower() == '/sources':
                    if last_sources:
                        print("\n📄 Sources from last query:")
                        for i, src in enumerate(last_sources, 1):
                            print(f"\n[{i}] Score: {src.get('score', 'N/A')}")
                            meta = src.get('metadata', {})
                            if 'title' in meta:
                                print(f"    Title: {meta['title']}")
                            if 'arxiv_id' in meta:
                                print(f"    ArXiv: {meta['arxiv_id']}")
                            print(f"    Text: {src['text'][:200]}...")
                    else:
                        print("No sources from previous query.")
                    continue

                if user_input.lower().startswith('/search '):
                    query = user_input[8:].strip()
                    if query:
                        print("\n🔍 Searching...")
                        results = self.search(query)
                        last_sources = results

                        print(f"\nFound {len(results)} relevant documents:")
                        for i, result in enumerate(results, 1):
                            print(f"\n[{i}] Score: {result.get('score', 'N/A')}")
                            meta = result.get('metadata', {})
                            if 'title' in meta:
                                print(f"    Title: {meta['title']}")
                            if 'arxiv_id' in meta:
                                print(f"    ArXiv: {meta['arxiv_id']}")
                            print(f"    Text: {result['text'][:300]}...")
                    continue

                # Regular query
                print("\n🤔 Thinking...")
                response = self.query(user_input)
                last_sources = response.get('sources', [])

                if response['mode'] == 'rag':
                    print(f"\n🤖 Assistant: {response['answer']}")
                    print(f"\n(Based on {len(response['sources'])} sources - type /sources to view)")
                else:
                    print("\nSearch results (LLM disabled):")
                    for i, src in enumerate(response['sources'], 1):
                        print(f"\n[{i}] Score: {src.get('score', 'N/A')}")
                        meta = src.get('metadata', {})
                        if 'title' in meta:
                            print(f"    Title: {meta['title']}")
                        print(f"    Text: {src['text'][:300]}...")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def display_result(result: Dict[str, Any]):
    """Display query result."""
    if result['mode'] == 'rag':
        print("\n" + "=" * 50)
        print("🤖 Answer:")
        print("-" * 50)
        print(result['answer'])
        print("\n" + "=" * 50)
        print(f"📚 Sources ({len(result['sources'])} documents):")
        print("-" * 50)
    else:
        print("\n" + "=" * 50)
        print("🔍 Search Results (LLM disabled):")
        print("-" * 50)

    for i, source in enumerate(result['sources'], 1):
        print(f"\n[{i}] Score: {source.get('score', 'N/A')}")
        meta = source.get('metadata', {})
        if 'title' in meta:
            print(f"    📖 Title: {meta['title']}")
        if 'arxiv_id' in meta:
            print(f"    🔗 ArXiv ID: {meta['arxiv_id']}")
        if 'filename' in meta:
            print(f"    📁 File: {meta['filename']}")
        print(f"    📝 Text: {source['text']}")
        print("-" * 30)


def main():
    """Main CLI function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ArXivist Query Agent - RAG-powered Q&A for ArXiv papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  USE_LOCAL_EMBEDDINGS=true (default)
  OPENAI_API_KEY=... (required if using OpenAI LLM or embeddings)
  OLLAMA_HOST=http://localhost:11434 (default Ollama endpoint)

Examples:
  %(prog)s                                    # Interactive chat mode
  %(prog)s -q "What is attention mechanism?"  # Single query
  %(prog)s --llm openai -q "Explain GPT"      # Use OpenAI
  %(prog)s --llm none -q "transformers"       # Search only, no LLM
  %(prog)s --model llama3.2 -q "BERT"         # Specify Ollama model
        """
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query to answer (if not provided, enters interactive mode)"
    )

    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("./llamaindex_storage"),
        help="Directory containing the index (default: ./llamaindex_storage)"
    )

    parser.add_argument(
        "--no-chroma",
        action="store_true",
        help="Use local file storage instead of ChromaDB"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="arxivist",
        help="ChromaDB collection name (default: arxivist)"
    )

    parser.add_argument(
        "--llm",
        type=str,
        choices=["ollama", "openai", "none"],
        default="ollama",
        help="LLM to use (default: ollama)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name (default: llama3.2 for Ollama, gpt-4o-mini for OpenAI)"
    )

    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for embeddings (default: cpu)"
    )

    args = parser.parse_args()

    print("ArXivist Query Agent")
    print("=" * 40)

    try:
        use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"

        agent = ArxivistAgent(
            persist_dir=args.persist_dir,
            use_chroma=not args.no_chroma,
            collection_name=args.collection,
            llm_type=args.llm,
            llm_model=args.model,
            use_local_embeddings=use_local_embeddings,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k
        )

        if args.query:
            result = agent.query(args.query)
            display_result(result)
        else:
            agent.chat()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you've run load_documents_llamaindex.py first!")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
