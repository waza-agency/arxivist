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
import time
from datetime import datetime, timedelta


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


def download_papers_batch(query: str, max_results: int, download_dir: Path, start_date: str = None, end_date: str = None) -> tuple[int, int]:
    """Download a batch of papers with optional date range. Returns (downloaded, skipped) counts."""
    
    # Get already downloaded papers
    downloaded_papers = get_downloaded_papers(download_dir)
    
    # Add date range to query if specified
    if start_date and end_date:
        query = f"({query}) AND submittedDate:[{start_date} TO {end_date}]"
        print(f"Searching {start_date} to {end_date}: '{query}'")
    else:
        print(f"Searching: '{query}'")
    
    # Create arXiv client with optimized settings
    client = arxiv.Client(
        page_size=1000,  # Larger page size for efficiency
        delay_seconds=3.0,  # Respect rate limits
        num_retries=3
    )
    
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
            
            try:
                print(f"Downloading: {result.title}")
                result.download_pdf(dirpath=str(download_dir), filename=filename)
                downloaded_count += 1
                print(f"✓ Downloaded: {filename}")
                
            except Exception as e:
                print(f"✗ Failed to download {arxiv_id}: {e}")
                continue
    
    except Exception as e:
        error_msg = str(e)
        if "unexpectedly empty" in error_msg.lower():
            print("Reached end of available papers for this batch")
        else:
            print(f"Error in batch: {e}")
            raise e
    
    return downloaded_count, skipped_count


def download_papers(query: str, max_results: Optional[int], download_dir: Path) -> None:
    """Download papers matching the query to the specified directory."""
    
    # Create download directory if it doesn't exist
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Get already downloaded papers
    downloaded_papers = get_downloaded_papers(download_dir)
    
    print(f"Searching for papers with query: '{query}'")
    print(f"Download directory: {download_dir}")
    print(f"Already downloaded: {len(downloaded_papers)} papers")
    
    total_downloaded = 0
    total_skipped = 0
    
    try:
        if max_results is None:
            # For unlimited downloads, use time-based chunking to bypass 30k API limit
            print("Download limit: no limit (ALL papers)")
            print("Using time-based chunking to access all papers...")
            
            # Start from 2007 (when arXiv started) to present
            start_year = 2007
            current_year = datetime.now().year
            
            interrupted = False
            for year in range(start_year, current_year + 1):
                if interrupted:
                    break
                    
                # Split each year into quarters to stay under 30k limit per batch
                quarters = [
                    (f"{year}01", f"{year}03"),  # Q1
                    (f"{year}04", f"{year}06"),  # Q2  
                    (f"{year}07", f"{year}09"),  # Q3
                    (f"{year}10", f"{year}12")   # Q4
                ]
                
                for start_month, end_month in quarters:
                    try:
                        print(f"\n--- Processing {year} Q{quarters.index((start_month, end_month))+1} ---")
                        downloaded, skipped = download_papers_batch(
                            query, 30000, download_dir, 
                            f"{start_month}01", f"{end_month}31"
                        )
                        total_downloaded += downloaded
                        total_skipped += skipped
                        
                        # Add delay between batches to be respectful
                        if downloaded > 0:
                            print(f"Batch complete: {downloaded} downloaded, {skipped} skipped")
                            time.sleep(5)
                            
                    except KeyboardInterrupt:
                        print(f"\nDownload interrupted by user")
                        interrupted = True
                        break
                    except Exception as e:
                        print(f"Error in batch {year} Q{quarters.index((start_month, end_month))+1}: {e}")
                        continue
        else:
            # For limited downloads, use simple approach
            print(f"Download limit: {max_results}")
            total_downloaded, total_skipped = download_papers_batch(query, max_results, download_dir)
    
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total downloaded: {total_downloaded} new papers")
    print(f"Total skipped: {total_skipped} already downloaded papers")
    print(f"Total papers in library: {len(get_downloaded_papers(download_dir))} papers")


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