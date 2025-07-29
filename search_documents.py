#!/usr/bin/env python3
"""
ArXivist Document Search - Search through documents stored in Neo4j using vector similarity.
Supports both local embeddings and OpenAI embeddings based on configuration.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import neo4j


def get_env_variable(var_name: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(var_name, default)
    if not value:
        raise ValueError(f"Environment variable {var_name} is required")
    return value


def setup_local_embeddings():
    """Setup local embedding model only."""
    # Setup local embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=512
    )
    embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
    
    Settings.embed_model = embed_model
    return embed_model, embedding_dimension


def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search documents using vector similarity only (no LLM)."""
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    neo4j_url = get_env_variable("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = get_env_variable("NEO4J_USER", "neo4j")
    neo4j_password = get_env_variable("NEO4J_PASSWORD")
    
    print("Using local embeddings for search...")
    
    # Setup local embeddings only
    embed_model, embedding_dimension = setup_local_embeddings()
    
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
        
        # Setup Neo4j vector store
        vector_store = Neo4jVectorStore(
            username=neo4j_user,
            password=neo4j_password,
            url=neo4j_url,
            embedding_dimension=embedding_dimension,
            distance_strategy="cosine"
        )
        
        print(f"\nSearching for: '{query}'")
        print("=" * 50)
        
        # Generate query embedding
        query_embedding = embed_model.get_query_embedding(query)
        
        # Perform direct vector similarity search using VectorStoreQuery
        from llama_index.core.vector_stores import VectorStoreQuery
        
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k
        )
        
        search_results = vector_store.query(vector_store_query)
        
        # Format results
        results = []
        if hasattr(search_results, 'nodes'):
            for i, node in enumerate(search_results.nodes, 1):
                # Try to get score from different possible attributes
                score = 0.0
                if hasattr(node, 'score'):
                    score = node.score
                elif hasattr(search_results, 'similarities') and len(search_results.similarities) >= i:
                    score = search_results.similarities[i-1]
                
                result = {
                    'rank': i,
                    'score': score,
                    'text': node.text[:500] + "..." if len(node.text) > 500 else node.text,
                    'metadata': node.metadata
                }
                results.append(result)
        
        # Create a simple summary without LLM
        summary = f"Found {len(results)} relevant documents for query '{query}'"
        if results:
            summary += f". Top result has similarity score of {results[0]['score']:.4f}."
        
        return results, summary
        
    except Exception as e:
        print(f"Error during search: {e}")
        raise e
    finally:
        neo4j_driver.close()


def display_results(results: List[Dict[str, Any]], response: str):
    """Display search results in a formatted way."""
    if not results:
        print("No results found.")
        return
    
    print(f"\n📊 Search Summary:")
    print("-" * 50)
    print(response)
    print()
    
    print(f"📄 Found {len(results)} relevant documents:")
    print("-" * 50)
    
    for result in results:
        print(f"\n[{result['rank']}] Similarity Score: {result['score']:.4f}")
        
        # Display metadata
        metadata = result['metadata']
        if 'title' in metadata:
            print(f"📖 Title: {metadata['title']}")
        if 'arxiv_id' in metadata:
            print(f"🔗 ArXiv ID: {metadata['arxiv_id']}")
        if 'filename' in metadata:
            print(f"📁 File: {metadata['filename']}")
        
        print(f"📝 Content: {result['text']}")
        print("-" * 30)


def interactive_search():
    """Interactive search mode."""
    print("ArXivist Interactive Search")
    print("=" * 40)
    print("Type your queries (or 'quit' to exit)")
    print()
    
    while True:
        try:
            query = input("\n🔍 Search: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            results, response = search_documents(query)
            display_results(results, response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Search ArXiv documents stored in Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables Required:
  NEO4J_URL=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=your_password
  USE_LOCAL_EMBEDDINGS=true (default: true, set to false to use OpenAI)
  OPENAI_API_KEY=your_openai_api_key (only required if USE_LOCAL_EMBEDDINGS=false)

Examples:
  %(prog)s                                    # Interactive search mode
  %(prog)s -q "machine learning"              # Single query search
  %(prog)s -q "neural networks" -k 10         # Return top 10 results
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Search query (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="ArXivist Search 1.0.0"
    )
    
    args = parser.parse_args()
    
    print("ArXivist Document Search")
    print("=" * 40)
    
    if args.query:
        # Single query mode
        try:
            results, response = search_documents(args.query, args.top_k)
            display_results(results, response)
        except Exception as e:
            print(f"Search failed: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_search()


if __name__ == "__main__":
    main()