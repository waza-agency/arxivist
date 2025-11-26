#!/usr/bin/env python3
"""
Document loader for LlamaIndex with local storage.
Loads all PDF documents from the papers directory into a local vector store.
Supports ChromaDB and local file-based persistence.
"""

import os
import sys
from pathlib import Path
from typing import Set, List, Optional
import hashlib
import json
from dotenv import load_dotenv

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Optional: ChromaDB support
try:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


def get_env_variable(var_name: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(var_name, default)
    if not value and default is None:
        raise ValueError(f"Environment variable {var_name} is required")
    return value


def get_document_hash(file_path: Path) -> str:
    """Generate hash for document to track if it's already loaded."""
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.md5(content).hexdigest()


def extract_arxiv_metadata(file_path: Path) -> dict:
    """Extract arXiv ID and title from filename."""
    filename = file_path.stem

    if "_" in filename:
        parts = filename.split("_", 1)
        arxiv_id = parts[0]
        title = parts[1] if len(parts) > 1 else "Unknown Title"
    else:
        arxiv_id = filename
        title = "Unknown Title"

    return {
        "arxiv_id": arxiv_id,
        "title": title.replace("_", " "),
        "file_path": str(file_path),
        "filename": file_path.name
    }


def load_tracking_file(tracking_path: Path) -> Set[str]:
    """Load set of document hashes already loaded."""
    if tracking_path.exists():
        with open(tracking_path, 'r') as f:
            data = json.load(f)
            return set(data.get("loaded_hashes", []))
    return set()


def save_tracking_file(tracking_path: Path, loaded_hashes: Set[str]):
    """Save document tracking file."""
    with open(tracking_path, 'w') as f:
        json.dump({"loaded_hashes": list(loaded_hashes)}, f, indent=2)


def setup_llama_index(use_local_embeddings: bool = True, device: str = "cpu"):
    """Configure LlamaIndex settings."""
    if use_local_embeddings:
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_length=512,
            device=device
        )
        embedding_dimension = 384
    else:
        # OpenAI embeddings
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimensions=1536
        )
        embedding_dimension = 1536

    node_parser = SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    return embedding_dimension


def create_chroma_store(persist_dir: Path, collection_name: str = "arxivist"):
    """Create ChromaDB vector store."""
    if not CHROMA_AVAILABLE:
        raise ImportError("ChromaDB not installed. Run: pip install chromadb llama-index-vector-stores-chroma")

    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context, vector_store


def create_local_store(persist_dir: Path):
    """Create local file-based vector store."""
    persist_dir.mkdir(parents=True, exist_ok=True)

    index_path = persist_dir / "index"
    if index_path.exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
    else:
        storage_context = StorageContext.from_defaults()

    return storage_context, None


def load_documents_llamaindex(
    papers_dir: Path = Path("./papers"),
    persist_dir: Path = Path("./llamaindex_storage"),
    batch_size: int = 5,
    use_chroma: bool = True,
    collection_name: str = "arxivist",
    device: str = "cpu"
) -> None:
    """Load all PDF documents from papers directory to LlamaIndex local storage."""

    load_dotenv()

    use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"

    if not use_local_embeddings:
        openai_api_key = get_env_variable("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = openai_api_key

    print(f"Setting up LlamaIndex with {'local' if use_local_embeddings else 'OpenAI'} embeddings...")
    print(f"Device: {device}")
    embedding_dimension = setup_llama_index(use_local_embeddings, device)

    # Setup storage
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    tracking_path = persist_dir / "loaded_documents.json"
    loaded_hashes = load_tracking_file(tracking_path)
    print(f"Found {len(loaded_hashes)} documents already loaded")

    # Setup vector store
    if use_chroma:
        print("Using ChromaDB for vector storage...")
        try:
            storage_context, vector_store = create_chroma_store(
                persist_dir / "chroma",
                collection_name
            )
        except ImportError as e:
            print(f"ChromaDB not available: {e}")
            print("Falling back to local file storage...")
            use_chroma = False
            storage_context, vector_store = create_local_store(persist_dir)
    else:
        print("Using local file storage...")
        storage_context, vector_store = create_local_store(persist_dir)

    # Find PDF files
    if not papers_dir.exists():
        print(f"Papers directory {papers_dir} does not exist!")
        return

    pdf_files = list(papers_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {papers_dir}")

    if not pdf_files:
        print("No PDF files found to process")
        return

    # Try to load existing index
    index = None
    index_path = persist_dir / "index"

    if not use_chroma and index_path.exists():
        try:
            print("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
            index = load_index_from_storage(storage_context)
            print("✓ Existing index loaded")
        except Exception as e:
            print(f"Could not load existing index: {e}")
            index = None

    # Process documents
    loaded_count = 0
    skipped_count = 0
    error_count = 0
    batch_docs = []
    current_batch = 0

    print(f"Processing {len(pdf_files)} files in batches of {batch_size}...")

    for i, pdf_file in enumerate(pdf_files, 1):
        try:
            doc_hash = get_document_hash(pdf_file)

            if doc_hash in loaded_hashes:
                print(f"[{i}/{len(pdf_files)}] Skipping {pdf_file.name}: already loaded")
                skipped_count += 1
                continue

            print(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")

            metadata = extract_arxiv_metadata(pdf_file)
            metadata["doc_hash"] = doc_hash

            try:
                reader = SimpleDirectoryReader(
                    input_files=[str(pdf_file)],
                    filename_as_id=True
                )
                documents = reader.load_data()

                if not documents:
                    print(f"Warning: No content extracted from {pdf_file.name}")
                    continue

                for doc in documents:
                    doc.metadata.update(metadata)

                batch_docs.extend(documents)
                print(f"✓ Prepared: {pdf_file.name} ({len(documents)} chunks)")

                # Process batch
                if len(batch_docs) >= batch_size or i == len(pdf_files):
                    current_batch += 1
                    print(f"\n--- Processing Batch {current_batch} ({len(batch_docs)} documents) ---")

                    try:
                        if use_chroma:
                            if index is None:
                                index = VectorStoreIndex.from_documents(
                                    batch_docs,
                                    storage_context=storage_context,
                                    show_progress=True
                                )
                            else:
                                for doc in batch_docs:
                                    index.insert(doc)
                        else:
                            if index is None:
                                index = VectorStoreIndex.from_documents(
                                    batch_docs,
                                    show_progress=True
                                )
                            else:
                                for doc in batch_docs:
                                    index.insert(doc)

                            index.storage_context.persist(persist_dir=str(index_path))

                        unique_hashes = set([d.metadata.get("doc_hash") for d in batch_docs if d.metadata.get("doc_hash")])
                        loaded_count += len(unique_hashes)
                        loaded_hashes.update(unique_hashes)

                        save_tracking_file(tracking_path, loaded_hashes)

                        print(f"✓ Batch {current_batch} committed ({len(batch_docs)} documents)")
                        batch_docs = []

                    except Exception as batch_error:
                        print(f"✗ Error processing batch {current_batch}: {batch_error}")
                        print("Attempting individual document processing...")

                        individual_success = 0
                        for doc in batch_docs:
                            try:
                                if index is None:
                                    index = VectorStoreIndex.from_documents(
                                        [doc],
                                        storage_context=storage_context if use_chroma else None,
                                        show_progress=False
                                    )
                                else:
                                    index.insert(doc)

                                individual_success += 1
                                if doc.metadata.get("doc_hash"):
                                    loaded_hashes.add(doc.metadata.get("doc_hash"))
                            except Exception as e:
                                print(f"  Failed: {doc.metadata.get('filename', 'unknown')}: {e}")

                        loaded_count += individual_success
                        error_count += len(batch_docs) - individual_success
                        batch_docs = []

                        save_tracking_file(tracking_path, loaded_hashes)
                        if not use_chroma and index is not None:
                            index.storage_context.persist(persist_dir=str(index_path))

            except Exception as e:
                print(f"✗ Error processing {pdf_file.name}: {e}")
                error_count += 1
                continue

        except Exception as e:
            print(f"✗ Error with {pdf_file.name}: {e}")
            error_count += 1
            continue

        if i % 50 == 0:
            print(f"\nProgress Update: {i}/{len(pdf_files)} files processed")
            print(f"Loaded: {loaded_count}, Skipped: {skipped_count}, Errors: {error_count}")
            print("-" * 50)

    # Final persistence
    if not use_chroma and index is not None:
        print("Persisting final index...")
        index.storage_context.persist(persist_dir=str(index_path))

    save_tracking_file(tracking_path, loaded_hashes)

    print(f"\n=== LOADING SUMMARY ===")
    print(f"Successfully loaded: {loaded_count} documents")
    print(f"Skipped (already loaded): {skipped_count} documents")
    print(f"Errors: {error_count} documents")
    print(f"Total documents tracked: {len(loaded_hashes)}")
    print(f"Storage location: {persist_dir}")


def main():
    """Main CLI function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load documents into LlamaIndex local storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  USE_LOCAL_EMBEDDINGS=true (default: true, set to false to use OpenAI)
  OPENAI_API_KEY=your_openai_api_key (only required if USE_LOCAL_EMBEDDINGS=false)

Examples:
  %(prog)s                                    # Load from ./papers with ChromaDB
  %(prog)s --papers-dir /path/to/pdfs         # Load from custom directory
  %(prog)s --persist-dir ./my_storage         # Custom storage location
  %(prog)s --no-chroma                        # Use local file storage instead of ChromaDB
  %(prog)s --batch-size 10                    # Process 10 docs per batch
  %(prog)s --device cuda                      # Use GPU for embeddings
        """
    )

    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("./papers"),
        help="Directory containing PDF files (default: ./papers)"
    )

    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("./llamaindex_storage"),
        help="Directory for storing index (default: ./llamaindex_storage)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of documents per batch (default: 5)"
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
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for embeddings (default: cpu)"
    )

    args = parser.parse_args()

    print("ArXivist Document Loader for LlamaIndex")
    print("=" * 40)

    load_documents_llamaindex(
        papers_dir=args.papers_dir,
        persist_dir=args.persist_dir,
        batch_size=args.batch_size,
        use_chroma=not args.no_chroma,
        collection_name=args.collection,
        device=args.device
    )


if __name__ == "__main__":
    main()
