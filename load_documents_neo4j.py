#!/usr/bin/env python3
"""
Document loader for Neo4j using LlamaIndex.
Loads all PDF documents from the papers directory into Neo4j with idempotent behavior.
"""

import os
import sys
from pathlib import Path
from typing import Set, List
import hashlib
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import neo4j


def get_env_variable(var_name: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(var_name, default)
    if not value:
        raise ValueError(f"Environment variable {var_name} is required")
    return value


def get_document_hash(file_path: Path) -> str:
    """Generate hash for document to track if it's already loaded."""
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.md5(content).hexdigest()


def get_loaded_document_hashes(neo4j_driver) -> Set[str]:
    """Get set of document hashes already loaded in Neo4j."""
    loaded_hashes = set()
    
    with neo4j_driver.session() as session:
        result = session.run(
            "MATCH (n) WHERE n.doc_hash IS NOT NULL RETURN DISTINCT n.doc_hash as hash"
        )
        for record in result:
            loaded_hashes.add(record["hash"])
    
    return loaded_hashes



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


def setup_llama_index(use_local_embeddings: bool = True):
    """Configure LlamaIndex settings."""
    # Configure embeddings
    if use_local_embeddings:
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_length=512,
            device="cpu"
        )
        embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
    else:
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimensions=1536
        )
        embedding_dimension = 1536
    
    # Configure text splitter
    node_parser = SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Set global settings
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser
    
    return embedding_dimension


def load_documents_to_neo4j(papers_dir: Path = Path("./papers"), batch_size: int = 5) -> None:
    """Load all PDF documents from papers directory to Neo4j with batch processing."""
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    neo4j_url = get_env_variable("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = get_env_variable("NEO4J_USER", "neo4j")
    neo4j_password = get_env_variable("NEO4J_PASSWORD")
    use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    
    # Set OpenAI API key only if not using local embeddings
    if not use_local_embeddings:
        openai_api_key = get_env_variable("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    print(f"Setting up LlamaIndex with {'local' if use_local_embeddings else 'OpenAI'} embeddings...")
    embedding_dimension = setup_llama_index(use_local_embeddings)
    
    # Connect to Neo4j
    print(f"Connecting to Neo4j at {neo4j_url}...")
    neo4j_driver = neo4j.GraphDatabase.driver(
        neo4j_url,
        auth=(neo4j_user, neo4j_password)
    )
    
    try:
        # Test connection
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
        print("✓ Neo4j connection established")
        
        # Get already loaded document hashes
        loaded_hashes = get_loaded_document_hashes(neo4j_driver)
        print(f"Found {len(loaded_hashes)} documents already loaded in Neo4j")
        
        # Find all PDF files
        if not papers_dir.exists():
            print(f"Papers directory {papers_dir} does not exist!")
            return
            
        pdf_files = list(papers_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in {papers_dir}")
        
        if not pdf_files:
            print("No PDF files found to process")
            return
        
        # Setup Neo4j vector store
        print("Setting up Neo4j vector store...")
        vector_store = Neo4jVectorStore(
            username=neo4j_user,
            password=neo4j_password,
            url=neo4j_url,
            embedding_dimension=embedding_dimension,
            distance_strategy="cosine"
        )
        
        # Process documents in batches
        loaded_count = 0
        skipped_count = 0
        error_count = 0
        batch_docs = []
        current_batch = 0
        
        print(f"Processing {len(pdf_files)} files in batches of {batch_size}...")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                # Calculate document hash
                doc_hash = get_document_hash(pdf_file)
                
                # Skip if already loaded
                if doc_hash in loaded_hashes:
                    print(f"[{i}/{len(pdf_files)}] Skipping {pdf_file.name}: already loaded")
                    skipped_count += 1
                    continue
                
                print(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
                
                # Extract metadata
                metadata = extract_arxiv_metadata(pdf_file)
                metadata["doc_hash"] = doc_hash
                
                # Load document
                try:
                    reader = SimpleDirectoryReader(
                        input_files=[str(pdf_file)],
                        filename_as_id=True
                    )
                    documents = reader.load_data()
                    
                    if not documents:
                        print(f"Warning: No content extracted from {pdf_file.name}")
                        continue
                    
                    # Add metadata to all documents
                    for doc in documents:
                        doc.metadata.update(metadata)
                    
                    
                    # Add to batch
                    batch_docs.extend(documents)
                    print(f"✓ Prepared: {pdf_file.name} ({len(documents)} chunks)")
                    
                    # Process batch when it's full or we're at the end
                    if len(batch_docs) >= batch_size or i == len(pdf_files):
                        current_batch += 1
                        print(f"\n--- Processing Batch {current_batch} ({len(batch_docs)} documents) ---")
                        
                        try:
                            # FIXED: Use direct vector store add instead of VectorStoreIndex.from_documents
                            # Convert documents to nodes and add directly to vector store
                            from llama_index.core.schema import TextNode
                            
                            text_nodes = []
                            for doc in batch_docs:
                                # Parse document into nodes
                                nodes = Settings.node_parser.get_nodes_from_documents([doc])
                                
                                # Generate embeddings for each node
                                for node in nodes:
                                    # Ensure node has all required metadata
                                    node.metadata.update(doc.metadata)
                                    # Generate embedding
                                    embedding = Settings.embed_model.get_text_embedding(node.text)
                                    node.embedding = embedding
                                    text_nodes.append(node)
                            
                            # Add nodes directly to vector store
                            node_ids = vector_store.add(text_nodes)
                            print(f"Processing {len(text_nodes)} text chunks...")
                            
                            # Count unique documents by their hash (one per PDF file)
                            unique_hashes = set([d.metadata.get("doc_hash") for d in batch_docs if d.metadata.get("doc_hash")])
                            loaded_count += len(unique_hashes)
                            loaded_hashes.update(unique_hashes)
                            
                            print(f"✓ Batch {current_batch} committed to Neo4j ({len(node_ids)} chunks inserted)")
                            
                            # Clear batch
                            batch_docs = []
                            
                        except Exception as batch_error:
                            print(f"✗ Error processing batch {current_batch}: {batch_error}")
                            # Try to process documents individually in this batch
                            print("Attempting individual document processing for this batch...")
                            individual_success = 0
                            for doc in batch_docs:
                                try:
                                    # Use direct vector store method for individual docs too
                                    nodes = Settings.node_parser.get_nodes_from_documents([doc])
                                    text_nodes = []
                                    for node in nodes:
                                        node.metadata.update(doc.metadata)
                                        embedding = Settings.embed_model.get_text_embedding(node.text)
                                        node.embedding = embedding
                                        text_nodes.append(node)
                                    
                                    vector_store.add(text_nodes)
                                    individual_success += 1
                                    if doc.metadata.get("doc_hash"):
                                        loaded_hashes.add(doc.metadata.get("doc_hash"))
                                except:
                                    pass
                            
                            loaded_count += individual_success
                            error_count += len(batch_docs) - individual_success
                            batch_docs = []
                            print(f"Individual processing: {individual_success} succeeded, {len(batch_docs) - individual_success} failed")
                    
                except Exception as e:
                    print(f"✗ Error processing {pdf_file.name}: {e}")
                    error_count += 1
                    continue
                    
            except Exception as e:
                print(f"✗ Error with {pdf_file.name}: {e}")
                error_count += 1
                continue
            
            # Progress update every 50 files
            if i % 50 == 0:
                print(f"\nProgress Update: {i}/{len(pdf_files)} files processed")
                print(f"Loaded: {loaded_count}, Skipped: {skipped_count}, Errors: {error_count}")
                print("-" * 50)
        
        # Final summary
        print(f"\n=== LOADING SUMMARY ===")
        print(f"Successfully loaded: {loaded_count} documents")
        print(f"Skipped (already loaded): {skipped_count} documents")
        print(f"Errors: {error_count} documents")
        print(f"Total documents in Neo4j: {len(loaded_hashes) + loaded_count}")
        
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        sys.exit(1)
        
    finally:
        neo4j_driver.close()


def main():
    """Main CLI function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load documents into Neo4j using LlamaIndex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables Required:
  NEO4J_URL=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=your_password
  USE_LOCAL_EMBEDDINGS=true (default: true, set to false to use OpenAI)
  OPENAI_API_KEY=your_openai_api_key (only required if USE_LOCAL_EMBEDDINGS=false)

Examples:
  %(prog)s                           # Load from ./papers directory (batch size 5)
  %(prog)s --papers-dir /path/to/pdfs  # Load from custom directory
  %(prog)s --batch-size 10           # Use larger batch size for faster processing
  %(prog)s --batch-size 1            # Process one document at a time (safer)
        """
    )
    
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("./papers"),
        help="Directory containing PDF files (default: ./papers)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of documents to process in each batch (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("ArXivist Document Loader for Neo4j")
    print("=" * 40)
    
    load_documents_to_neo4j(args.papers_dir, args.batch_size)


if __name__ == "__main__":
    main()