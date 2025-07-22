#!/usr/bin/env python3
"""
ArXivist - A CLI tool to download arXiv papers by search term.
Defaults to downloading "Machine Learning" papers and avoids re-downloading existing papers.
"""

import argparse
import os
import sys
from pathlib import Path
import arxiv
import hashlib
from typing import Set, Optional


def get_downloaded_papers(download_dir: Path) -> Set[str]:
    """Get set of already downloaded paper IDs from filenames."""
    downloaded = set()
    if download_dir.exists():
        for file in download_dir.glob("*.pdf"):
            # Extract arXiv ID from filename (assumes format: ID_title.pdf)
            if "_" in file.stem:
                arxiv_id = file.stem.split("_")[0]
                downloaded.add(arxiv_id)
    return downloaded


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename.strip()


def download_papers(query: str, max_results: Optional[int], download_dir: Path) -> None:
    """Download papers matching the query to the specified directory."""
    
    # Create download directory if it doesn't exist
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Get already downloaded papers
    downloaded_papers = get_downloaded_papers(download_dir)
    
    # Handle unlimited downloads
    if max_results is None:
        max_results = 1000000  # Set to very large number for unlimited downloads
        limit_text = "no limit (ALL papers)"
    else:
        limit_text = str(max_results)
    
    print(f"Searching for papers with query: '{query}'")
    print(f"Download directory: {download_dir}")
    print(f"Download limit: {limit_text}")
    print(f"Already downloaded: {len(downloaded_papers)} papers")
    
    # Create arXiv client and search
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    downloaded_count = 0
    skipped_count = 0
    
    try:
        for result in client.results(search):
            # Extract arXiv ID (remove version if present)
            arxiv_id = result.entry_id.split('/')[-1].split('v')[0]
            
            if arxiv_id in downloaded_papers:
                print(f"Skipping {arxiv_id}: already downloaded")
                skipped_count += 1
                continue
            
            # Create safe filename
            title = sanitize_filename(result.title)
            filename = f"{arxiv_id}_{title[:100]}.pdf"  # Limit title length
            filepath = download_dir / filename
            
            try:
                print(f"Downloading: {result.title}")
                result.download_pdf(dirpath=str(download_dir), filename=filename)
                downloaded_count += 1
                print(f"✓ Downloaded: {filename}")
                
            except Exception as e:
                print(f"✗ Failed to download {arxiv_id}: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
    except Exception as e:
        print(f"Error during search/download: {e}")
        sys.exit(1)
    
    print(f"\nDownload complete!")
    print(f"Downloaded: {downloaded_count} new papers")
    print(f"Skipped: {skipped_count} already downloaded papers")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Download arXiv papers by search term",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Download ALL Machine Learning papers
  %(prog)s --query "computer vision"          # Download ALL computer vision papers
  %(prog)s --query "neural networks" --max 50  # Download 50 neural network papers
  %(prog)s --dir ./papers                     # Download ALL papers to custom directory
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        default="cat:cs.LG OR cat:stat.ML",  # Machine Learning categories
        help="Search query (default: Machine Learning papers)"
    )
    
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=None,
        help="Maximum number of papers to download (default: no limit - downloads ALL papers)"
    )
    
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        default=Path("./papers"),
        help="Download directory (default: ./papers)"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="ArXivist 1.0.0"
    )
    
    args = parser.parse_args()
    
    print("ArXivist - arXiv Paper Downloader")
    print("=" * 40)
    
    download_papers(args.query, args.max, args.dir)


if __name__ == "__main__":
    main()