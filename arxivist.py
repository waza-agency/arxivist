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
    """Get set of already downloaded paper IDs from filenames, excluding empty files."""
    downloaded = set()
    empty_files = []
    
    if download_dir.exists():
        for file in download_dir.glob("*.pdf"):
            # Check if file is empty (0 bytes)
            if file.stat().st_size == 0:
                empty_files.append(file.name)
                continue
                
            # Extract arXiv ID from filename (assumes format: ID_title.pdf)
            if "_" in file.stem:
                arxiv_id = file.stem.split("_")[0]
                downloaded.add(arxiv_id)
    
    # Report empty files found
    if empty_files:
        print(f"Found {len(empty_files)} empty files that will be re-downloaded:")
        for empty_file in empty_files[:5]:  # Show first 5
            print(f"  - {empty_file}")
        if len(empty_files) > 5:
            print(f"  ... and {len(empty_files) - 5} more")
    
    return downloaded


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename.strip()


def download_papers_batch(query: str, max_results: int, download_dir: Path, start_date: str = None, end_date: str = None) -> tuple[int, int, bool]:
    """Download a batch of papers with optional date range. Returns (downloaded, skipped, hit_limit) counts."""
    
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
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    downloaded_count = 0
    skipped_count = 0
    result_count = 0
    
    try:
        for result in client.results(search):
            result_count += 1
            
            # Extract arXiv ID (remove version if present)
            arxiv_id = result.entry_id.split('/')[-1].split('v')[0]
            
            if arxiv_id in downloaded_papers:
                print(f"Skipping {arxiv_id}: already downloaded")
                skipped_count += 1
                continue
            
            # Create safe filename
            title = sanitize_filename(result.title)
            filename = f"{arxiv_id}_{title[:100]}.pdf"  # Limit title length
            file_path = download_dir / filename
            
            # Remove existing empty file if present
            if file_path.exists() and file_path.stat().st_size == 0:
                print(f"Removing empty file: {filename}")
                file_path.unlink()
            
            try:
                print(f"Downloading: {result.title}")
                result.download_pdf(dirpath=str(download_dir), filename=filename)
                
                # Verify the download was successful (non-empty)
                if file_path.exists() and file_path.stat().st_size > 0:
                    downloaded_count += 1
                    print(f"✓ Downloaded: {filename} ({file_path.stat().st_size} bytes)")
                else:
                    print(f"✗ Download failed or file is empty: {filename}")
                    continue
                
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
    
    # Check if we hit the API limit
    hit_limit = result_count >= max_results
    if hit_limit:
        print(f"⚠️ Hit API limit ({max_results} results) - may need smaller time window")
    
    return downloaded_count, skipped_count, hit_limit


def download_papers_with_adaptive_windows(query: str, download_dir: Path, start_date: str, end_date: str, window_type: str = "month") -> tuple[int, int]:
    """Download papers with adaptive time window splitting when hitting API limits."""
    from datetime import datetime, timedelta
    import calendar
    
    total_downloaded = 0
    total_skipped = 0
    
    if window_type == "month":
        # Try monthly batch first
        downloaded, skipped, hit_limit = download_papers_batch(query, 25000, download_dir, start_date, end_date)
        total_downloaded += downloaded
        total_skipped += skipped
        
        if hit_limit:
            print(f"Month {start_date[:6]} hit limit, splitting into weeks...")
            # Split month into weeks
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            current_dt = start_dt
            while current_dt <= end_dt:
                week_end = min(current_dt + timedelta(days=6), end_dt)
                week_downloaded, week_skipped = download_papers_with_adaptive_windows(
                    query, download_dir, 
                    current_dt.strftime("%Y%m%d"), 
                    week_end.strftime("%Y%m%d"), 
                    "week"
                )
                total_downloaded += week_downloaded
                total_skipped += week_skipped
                current_dt = week_end + timedelta(days=1)
                
    elif window_type == "week":
        # Try weekly batch
        downloaded, skipped, hit_limit = download_papers_batch(query, 25000, download_dir, start_date, end_date)
        total_downloaded += downloaded
        total_skipped += skipped
        
        if hit_limit:
            print(f"Week {start_date}-{end_date} hit limit, splitting into days...")
            # Split week into days
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            current_dt = start_dt
            while current_dt <= end_dt:
                day_downloaded, day_skipped, _ = download_papers_batch(
                    query, 25000, download_dir,
                    current_dt.strftime("%Y%m%d"),
                    current_dt.strftime("%Y%m%d")
                )
                total_downloaded += day_downloaded
                total_skipped += day_skipped
                current_dt += timedelta(days=1)
                
    else:  # day
        downloaded, skipped, _ = download_papers_batch(query, 25000, download_dir, start_date, end_date)
        total_downloaded += downloaded
        total_skipped += skipped
    
    return total_downloaded, total_skipped


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
            
            # Start from present and go backwards to get newest papers first
            start_year = 2007
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            interrupted = False
            
            # Process years in reverse order (newest first)
            for year in range(current_year, start_year - 1, -1):
                if interrupted:
                    break
                
                # Determine the month range for current year
                if year == current_year:
                    # For current year, only go up to current month
                    end_month = current_month
                else:
                    end_month = 12
                
                # Use smaller batches for recent years (2020+) due to higher paper volume
                # Use monthly batches for 2020+ to avoid 30k limit, quarterly for older years
                if year >= 2020:
                    # Monthly batches for recent high-volume years with adaptive windows
                    for month in range(end_month, 0, -1):  # Reverse order within year
                        try:
                            month_str = f"{month:02d}"
                            print(f"\n--- Processing {year}-{month_str} (adaptive windows) ---")
                            downloaded, skipped = download_papers_with_adaptive_windows(
                                query, download_dir,
                                f"{year}{month_str}01", f"{year}{month_str}31"
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
                            print(f"Error in batch {year}-{month_str}: {e}")
                            continue
                else:
                    # Quarterly batches for older years (lower paper volume)
                    quarters = [
                        (f"{year}10", f"{year}12"),  # Q4 (reversed order)
                        (f"{year}07", f"{year}09"),  # Q3
                        (f"{year}04", f"{year}06"),  # Q2  
                        (f"{year}01", f"{year}03"),  # Q1
                    ]
                    
                    # Adjust quarters for partial years
                    if year == current_year:
                        quarters = [(f"{year}{str(((end_month-1)//3)*3+1).zfill(2)}", f"{year}{end_month:02d}")]
                    
                    for start_month, end_month_q in quarters:
                        try:
                            quarter_num = 4 - quarters.index((start_month, end_month_q))
                            print(f"\n--- Processing {year} Q{quarter_num} (quarterly batch) ---")
                            downloaded, skipped, _ = download_papers_batch(
                                query, 30000, download_dir, 
                                f"{start_month}01", f"{end_month_q}31"
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
                            print(f"Error in batch {year} Q{quarter_num}: {e}")
                            continue
        else:
            # For limited downloads, use simple approach
            print(f"Download limit: {max_results}")
            total_downloaded, total_skipped, _ = download_papers_batch(query, max_results, download_dir)
    
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