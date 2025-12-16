#!/usr/bin/env python3
"""
Document loader for LlamaIndex with local storage.
Loads all PDF documents from the papers directory into a local vector store.
Supports ChromaDB and local file-based persistence.
"""

import os
import sys
from pathlib import Path
from typing import Set, List, Optional, Tuple, Dict, Any
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
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


def hash_file_worker(file_path: Path) -> Tuple[Path, str]:
    """Worker function to hash a single file. Returns (path, hash)."""
    try:
        doc_hash = get_document_hash(file_path)
        return (file_path, doc_hash)
    except Exception as e:
        print(f"Error hashing {file_path.name}: {e}")
        return (file_path, None)


def hash_files_parallel(pdf_files: List[Path], max_workers: int = None) -> Dict[Path, str]:
    """Hash multiple files in parallel using threads (I/O bound)."""
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 4)

    file_hashes = {}
    total = len(pdf_files)
    print(f"Hashing {total:,} files with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(hash_file_worker, f): f for f in pdf_files}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            file_path, doc_hash = future.result()
            if doc_hash:
                file_hashes[file_path] = doc_hash

            # Progress update every 1000 files or at 10% intervals
            if completed % 1000 == 0 or completed == total:
                pct = (completed / total) * 100
                print(f"\r  Hashed {completed:,}/{total:,} ({pct:.1f}%)", end="", flush=True)

    print(f"\n✓ Hashed {len(file_hashes):,} files")
    return file_hashes


def read_pdf_bytes(file_path: Path) -> bytes:
    """Read PDF file into memory (I/O operation)."""
    with open(file_path, 'rb') as f:
        return f.read()


def parse_pdf_from_bytes(args: Tuple[Path, bytes, str]) -> Tuple[Path, List[Any], str, Optional[str]]:
    """
    Worker function to parse PDF from bytes (CPU-bound, no I/O).
    Must be a top-level function for ProcessPoolExecutor pickling.
    Returns (file_path, documents, doc_hash, error_message).
    """
    file_path, pdf_bytes, doc_hash = args
    try:
        import tempfile
        from llama_index.core.readers import SimpleDirectoryReader

        # Write bytes to temp file for SimpleDirectoryReader
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            reader = SimpleDirectoryReader(
                input_files=[tmp_path],
                filename_as_id=True
            )
            documents = reader.load_data()
        finally:
            os.unlink(tmp_path)

        if not documents:
            return (file_path, [], doc_hash, "No content extracted")

        # Extract metadata
        metadata = extract_arxiv_metadata(file_path)
        metadata["doc_hash"] = doc_hash

        # Add metadata to each document
        for doc in documents:
            doc.metadata.update(metadata)
            # Use original path in metadata, not temp path
            doc.metadata["file_path"] = str(file_path)

        return (file_path, documents, doc_hash, None)

    except Exception as e:
        return (file_path, [], doc_hash, str(e))


def parse_pdf_worker(args: Tuple[Path, str]) -> Tuple[Path, List[Any], str, Optional[str]]:
    """
    Worker function to parse a single PDF file (includes I/O).
    Must be a top-level function for ProcessPoolExecutor pickling.
    Returns (file_path, documents, doc_hash, error_message).
    """
    file_path, doc_hash = args
    try:
        from llama_index.core.readers import SimpleDirectoryReader

        reader = SimpleDirectoryReader(
            input_files=[str(file_path)],
            filename_as_id=True
        )
        documents = reader.load_data()

        if not documents:
            return (file_path, [], doc_hash, "No content extracted")

        # Extract metadata
        metadata = extract_arxiv_metadata(file_path)
        metadata["doc_hash"] = doc_hash

        # Add metadata to each document
        for doc in documents:
            doc.metadata.update(metadata)

        return (file_path, documents, doc_hash, None)

    except Exception as e:
        return (file_path, [], doc_hash, str(e))


def parse_pdfs_parallel(
    files_to_process: List[Tuple[Path, str]],
    max_workers: int = None,
    sequential_io: bool = True
) -> Tuple[List[Any], List[str], Dict[Path, str]]:
    """
    Parse multiple PDFs in parallel using processes (CPU bound).

    Args:
        files_to_process: List of (file_path, doc_hash) tuples
        max_workers: Number of parallel processes
        sequential_io: If True, read files sequentially then parse in parallel.
                      Best for HDDs. If False, each worker reads its own file.
                      Best for SSDs.

    Returns (all_documents, errors, successful_hashes).
    """
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)

    all_documents = []
    errors = []
    successful_hashes = {}
    total = len(files_to_process)

    if sequential_io:
        # PHASE 1: Sequential I/O - read all files into memory
        # This is faster on HDDs where parallel reads cause seek thrashing
        print(f"Reading {total} PDFs sequentially (optimized for HDD)...")
        pdf_data = []
        for i, (file_path, doc_hash) in enumerate(files_to_process, 1):
            try:
                pdf_bytes = read_pdf_bytes(file_path)
                pdf_data.append((file_path, pdf_bytes, doc_hash))
                if i % 50 == 0:
                    print(f"  Read {i}/{total} files...")
            except Exception as e:
                errors.append(f"{file_path.name}: Failed to read: {e}")

        print(f"✓ Read {len(pdf_data)} files into memory")

        # PHASE 2: Parallel CPU - parse all files from memory
        print(f"Parsing {len(pdf_data)} PDFs with {max_workers} processes...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(parse_pdf_from_bytes, args): args[0] for args in pdf_data}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                file_path, documents, doc_hash, error = future.result()

                if error:
                    errors.append(f"{file_path.name}: {error}")
                    print(f"[{completed}/{len(pdf_data)}] ✗ {file_path.name}: {error}")
                else:
                    all_documents.extend(documents)
                    successful_hashes[file_path] = doc_hash
                    print(f"[{completed}/{len(pdf_data)}] ✓ {file_path.name} ({len(documents)} chunks)")
    else:
        # Original approach: each worker reads its own file
        # Better for SSDs with fast random access
        print(f"Parsing {total} PDFs with {max_workers} processes (parallel I/O)...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(parse_pdf_worker, args): args[0] for args in files_to_process}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                file_path, documents, doc_hash, error = future.result()

                if error:
                    errors.append(f"{file_path.name}: {error}")
                    print(f"[{completed}/{total}] ✗ {file_path.name}: {error}")
                else:
                    all_documents.extend(documents)
                    successful_hashes[file_path] = doc_hash
                    print(f"[{completed}/{total}] ✓ {file_path.name} ({len(documents)} chunks)")

    print(f"✓ Parsed {len(successful_hashes)} PDFs, {len(errors)} errors")
    return all_documents, errors, successful_hashes


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


def iter_pdf_files(papers_dir: Path):
    """Generator that yields PDF files one at a time to avoid loading all paths into memory."""
    for entry in papers_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() == '.pdf':
            yield entry


def count_pdf_files(papers_dir: Path) -> int:
    """Count PDF files without loading all paths into memory."""
    count = 0
    for entry in papers_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() == '.pdf':
            count += 1
    return count


def process_file_batch(
    pdf_files_batch: List[Path],
    loaded_hashes: Set[str],
    hash_workers: int,
    num_workers: int,
    sequential_io: bool
) -> Tuple[List[Any], List[str], Dict[Path, str], int]:
    """
    Process a batch of PDF files: hash, filter, parse.

    Returns (documents, errors, successful_hashes, skipped_count).
    """
    # Hash files in batch
    file_hashes = hash_files_parallel(pdf_files_batch, max_workers=hash_workers)

    # Filter out already-loaded files
    files_to_process = []
    skipped_count = 0
    for file_path, doc_hash in file_hashes.items():
        if doc_hash in loaded_hashes:
            skipped_count += 1
        else:
            files_to_process.append((file_path, doc_hash))

    if not files_to_process:
        return [], [], {}, skipped_count

    # Parse PDFs
    all_documents, parse_errors, successful_hashes = parse_pdfs_parallel(
        files_to_process,
        max_workers=num_workers,
        sequential_io=sequential_io
    )

    return all_documents, parse_errors, successful_hashes, skipped_count


def save_tracking_file(tracking_path: Path, loaded_hashes: Set[str]):
    """Save document tracking file."""
    with open(tracking_path, 'w') as f:
        json.dump({"loaded_hashes": list(loaded_hashes)}, f, indent=2)


def setup_llama_index(
    use_local_embeddings: bool = True,
    device: str = "cpu",
    embed_batch_size: int = 64
):
    """Configure LlamaIndex settings with GPU and batch support."""
    if use_local_embeddings:
        # Auto-detect CUDA if device is "auto"
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            except ImportError:
                device = "cpu"

        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_length=512,
            device=device,
            embed_batch_size=embed_batch_size,  # Process multiple texts at once
        )
        embedding_dimension = 384
        print(f"Embedding model: all-MiniLM-L6-v2 on {device}, batch_size={embed_batch_size}")
    else:
        # OpenAI embeddings
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimensions=1536
        )
        embedding_dimension = 1536
        print("Embedding model: OpenAI text-embedding-3-small")

    node_parser = SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    Settings.embed_model = embed_model
    Settings.node_parser = node_parser
    Settings.num_workers = os.cpu_count()  # Enable parallel node parsing

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
    batch_size: int = 100,
    file_batch_size: int = 500,
    use_chroma: bool = True,
    collection_name: str = "arxivist",
    device: str = "auto",
    embed_batch_size: int = 32,
    num_workers: int = None,
    hash_workers: int = None,
    sequential_io: bool = True,
) -> None:
    """
    Load all PDF documents from papers directory to LlamaIndex local storage.

    Processes files in batches to handle large directories (100k+ files) without
    running out of memory. Uses multithreading for I/O-bound tasks (hashing)
    and multiprocessing for CPU-bound tasks (PDF parsing).

    Args:
        papers_dir: Directory containing PDF files
        persist_dir: Directory for storing index
        batch_size: Number of document chunks per embedding batch
        file_batch_size: Number of PDF files to process at once (default: 500)
        use_chroma: Use ChromaDB (True) or local file storage (False)
        collection_name: ChromaDB collection name
        device: Device for embeddings ('auto', 'cpu', 'cuda', 'mps')
        embed_batch_size: Batch size for embedding generation
        num_workers: Number of processes for PDF parsing (default: cpu_count - 1)
        hash_workers: Number of threads for file hashing (default: cpu_count * 4)
        sequential_io: Read files sequentially (True for HDD, False for SSD)
    """

    load_dotenv()

    # Set worker defaults
    cpu_count = os.cpu_count() or 1
    if num_workers is None:
        num_workers = max(1, cpu_count - 1)
    if hash_workers is None:
        hash_workers = min(32, cpu_count * 4)

    use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"

    if not use_local_embeddings:
        openai_api_key = get_env_variable("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = openai_api_key

    print(f"Setting up LlamaIndex with {'local' if use_local_embeddings else 'OpenAI'} embeddings...")
    print(f"Device: {device} | PDF workers: {num_workers} | Hash workers: {hash_workers}")
    print(f"File batch size: {file_batch_size} | Embedding batch size: {batch_size}")
    embedding_dimension = setup_llama_index(use_local_embeddings, device, embed_batch_size)

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

    # Check papers directory
    if not papers_dir.exists():
        print(f"Papers directory {papers_dir} does not exist!")
        return

    # Count PDF files without loading all into memory
    print(f"Counting PDF files in {papers_dir}...")
    total_files = count_pdf_files(papers_dir)
    print(f"Found {total_files:,} PDF files")

    if total_files == 0:
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
            print("Existing index loaded")
        except Exception as e:
            print(f"Could not load existing index: {e}")
            index = None

    # Process files in batches
    total_loaded = 0
    total_skipped = 0
    total_errors = 0
    total_chunks = 0
    file_batch_num = 0
    total_file_batches = (total_files + file_batch_size - 1) // file_batch_size

    print(f"\n{'='*60}")
    print(f"Processing {total_files:,} files in {total_file_batches} batches of {file_batch_size}")
    print(f"{'='*60}")

    # Collect files in batches using generator
    current_batch = []
    files_seen = 0

    for pdf_file in iter_pdf_files(papers_dir):
        current_batch.append(pdf_file)
        files_seen += 1

        # Process when batch is full or at the end
        if len(current_batch) >= file_batch_size or files_seen == total_files:
            file_batch_num += 1
            print(f"\n{'='*60}")
            print(f"FILE BATCH {file_batch_num}/{total_file_batches} ({len(current_batch)} files)")
            print(f"{'='*60}")

            # Process this batch: hash -> filter -> parse
            batch_docs, batch_errors, batch_hashes, batch_skipped = process_file_batch(
                current_batch,
                loaded_hashes,
                hash_workers,
                num_workers,
                sequential_io
            )

            total_skipped += batch_skipped
            total_errors += len(batch_errors)

            if batch_errors:
                print(f"Batch errors: {len(batch_errors)}")

            if batch_docs:
                # Index the documents in embedding batches
                print(f"\nIndexing {len(batch_docs)} document chunks...")
                total_chunks += len(batch_docs)

                # Track which doc_hashes were successfully embedded
                successfully_embedded_hashes = set()

                for emb_start in range(0, len(batch_docs), batch_size):
                    emb_end = min(emb_start + batch_size, len(batch_docs))
                    emb_batch = batch_docs[emb_start:emb_end]
                    emb_batch_num = (emb_start // batch_size) + 1
                    total_emb_batches = (len(batch_docs) + batch_size - 1) // batch_size

                    try:
                        if index is None:
                            index = VectorStoreIndex.from_documents(
                                emb_batch,
                                storage_context=storage_context if use_chroma else None,
                                show_progress=True
                            )
                        else:
                            index.refresh_ref_docs(emb_batch)

                        # Only track as successful if embedding completed without error
                        for doc in emb_batch:
                            if "doc_hash" in doc.metadata:
                                successfully_embedded_hashes.add(doc.metadata["doc_hash"])

                        print(f"  Indexed embedding batch {emb_batch_num}/{total_emb_batches}")

                    except Exception as e:
                        print(f"  Error indexing embedding batch {emb_batch_num}: {e}")
                        total_errors += len(emb_batch)

                # Update tracking ONLY with successfully embedded hashes
                for file_path, doc_hash in batch_hashes.items():
                    if doc_hash in successfully_embedded_hashes:
                        loaded_hashes.add(doc_hash)
                        total_loaded += 1
                    else:
                        print(f"  Skipping tracking for {file_path.name} (embedding failed)")

                # Persist after each file batch
                if not use_chroma and index is not None:
                    index.storage_context.persist(persist_dir=str(index_path))
                save_tracking_file(tracking_path, loaded_hashes)
                print(f"Progress saved ({total_loaded:,} loaded, {total_skipped:,} skipped)")

            else:
                print(f"No new documents in this batch (skipped: {batch_skipped})")

            # Clear batch for next iteration
            current_batch = []

    # Final persistence
    if not use_chroma and index is not None:
        print("\nPersisting final index...")
        index.storage_context.persist(persist_dir=str(index_path))

    save_tracking_file(tracking_path, loaded_hashes)

    print(f"\n{'='*60}")
    print(f"=== LOADING SUMMARY ===")
    print(f"{'='*60}")
    print(f"Total files scanned: {total_files:,}")
    print(f"Successfully loaded: {total_loaded:,} PDFs ({total_chunks:,} chunks)")
    print(f"Skipped (already loaded): {total_skipped:,} PDFs")
    print(f"Errors: {total_errors}")
    print(f"Total documents tracked: {len(loaded_hashes):,}")
    print(f"Storage location: {persist_dir}")
    print(f"{'='*60}")


def main():
    """Main CLI function."""
    import argparse

    cpu_count = os.cpu_count() or 1

    parser = argparse.ArgumentParser(
        description="Load documents into LlamaIndex local storage (multithreaded)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Environment Variables:
  USE_LOCAL_EMBEDDINGS=true (default: true, set to false to use OpenAI)
  OPENAI_API_KEY=your_openai_api_key (only required if USE_LOCAL_EMBEDDINGS=false)

Performance Tips:
  - Use --device auto to auto-detect GPU
  - Increase --embed-batch-size for faster GPU embedding (if VRAM allows, e.g., 64 or 128)
  - Decrease --embed-batch-size if you get CUDA OOM errors (try 16 or 8)
  - Adjust --num-workers based on CPU cores (detected: {cpu_count})
  - Use --batch-size 200+ for large document sets

Examples:
  %(prog)s                                    # Load with auto GPU detection
  %(prog)s --device cuda --embed-batch-size 128  # Force CUDA with large batches
  %(prog)s --num-workers 8 --hash-workers 32     # Custom parallelism
  %(prog)s --papers-dir /path/to/pdfs         # Load from custom directory
  %(prog)s --no-chroma                        # Use local file storage instead of ChromaDB
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
        default=100,
        help="Number of document chunks per embedding batch (default: 100)"
    )

    parser.add_argument(
        "--file-batch-size",
        type=int,
        default=500,
        help="Number of PDF files to process at once (default: 500). Lower for less memory usage."
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
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for embeddings: auto detects GPU (default: auto)"
    )

    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32, increase for GPU with more VRAM)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=f"Number of processes for PDF parsing (default: {max(1, cpu_count - 1)})"
    )

    parser.add_argument(
        "--hash-workers",
        type=int,
        default=None,
        help=f"Number of threads for file hashing (default: {min(32, cpu_count * 4)})"
    )

    parser.add_argument(
        "--ssd",
        action="store_true",
        help="Enable parallel I/O (use for SSDs). Default is sequential I/O for HDDs."
    )

    args = parser.parse_args()

    print("ArXivist Document Loader for LlamaIndex (Multithreaded)")
    print("=" * 55)

    load_documents_llamaindex(
        papers_dir=args.papers_dir,
        persist_dir=args.persist_dir,
        batch_size=args.batch_size,
        file_batch_size=args.file_batch_size,
        use_chroma=not args.no_chroma,
        collection_name=args.collection,
        device=args.device,
        embed_batch_size=args.embed_batch_size,
        num_workers=args.num_workers,
        hash_workers=args.hash_workers,
        sequential_io=not args.ssd,
    )


if __name__ == "__main__":
    main()
