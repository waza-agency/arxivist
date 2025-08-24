#!/usr/bin/env python3
"""
Script to count papers already loaded in Neo4j.
"""

import os
from dotenv import load_dotenv
import neo4j


def get_env_variable(var_name: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(var_name, default)
    if not value:
        raise ValueError(f"Environment variable {var_name} is required")
    return value


def count_papers_in_neo4j():
    """Count unique papers and total chunks in Neo4j."""

    load_dotenv()

    neo4j_url = get_env_variable("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = get_env_variable("NEO4J_USER", "neo4j")
    neo4j_password = get_env_variable("NEO4J_PASSWORD", "your_password")

    print(f"Connecting to Neo4j at {neo4j_url}...")

    driver = neo4j.GraphDatabase.driver(
        neo4j_url,
        auth=(neo4j_user, neo4j_password)
    )

    try:
        with driver.session() as session:
            # Test connection
            session.run("RETURN 1")
            print("✓ Neo4j connection established\n")

            # Count unique papers by doc_hash
            result = session.run(
                "MATCH (n) WHERE n.doc_hash IS NOT NULL "
                "RETURN COUNT(DISTINCT n.doc_hash) as unique_papers"
            )
            unique_papers = result.single()["unique_papers"]

            # Count total chunks/nodes
            result = session.run(
                "MATCH (n) WHERE n.doc_hash IS NOT NULL "
                "RETURN COUNT(n) as total_chunks"
            )
            total_chunks = result.single()["total_chunks"]

            # Get some sample paper information
            result = session.run(
                "MATCH (n) WHERE n.doc_hash IS NOT NULL AND n.arxiv_id IS NOT NULL "
                "RETURN DISTINCT n.arxiv_id as arxiv_id, n.title as title, n.filename as filename "
                "LIMIT 10"
            )

            sample_papers = []


            # Display results
            print("=== NEO4J PAPER COUNT ===")
            print(f"Unique papers: {unique_papers}")
            print(f"Total chunks: {total_chunks}")

            if unique_papers > 0:
                print(f"Average chunks per paper: {total_chunks / unique_papers:.1f}")


    except Exception as e:
        print(f"Error: {e}")

    finally:
        driver.close()


if __name__ == "__main__":
    count_papers_in_neo4j()
