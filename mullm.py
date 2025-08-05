#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to extract text from a PDF document using pymupdf4llm, optimized
for speed using multi-processing.

This script processes pages in parallel to significantly speed up extraction,
measures the total time taken, calculates the average time per page, and
saves the extracted text in Markdown format.

Installation:
    pip install pymupdf4llm pymupdf

Usage:
    python extract_and_time.py /path/to/your/document.pdf
"""

import argparse
import pathlib
import sys
import time
import pymupdf
import pymupdf4llm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_page(pdf_path: str, page_num: int):
    """
    Extracts markdown text from a single page of a PDF.
    This function is designed to be run in a separate process.

    Args:
        pdf_path: The file path to the PDF document.
        page_num: The 0-based index of the page to process.

    Returns:
        A tuple containing the page number and its extracted markdown text.
    """
    try:
        # The 'pages' argument allows us to target a specific page.
        md_text = pymupdf4llm.to_markdown(pdf_path, pages=[page_num])
        return page_num, md_text
    except Exception as e:
        # Return error information for the parent process to handle
        return page_num, f"Error processing page {page_num}: {e}"


def extract_text_with_timing_parallel(pdf_path: str):
    """
    Extracts text from a PDF in parallel, measures performance, and returns the results.

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        A tuple containing:
        - The extracted text in Markdown format.
        - The total time taken for extraction.
        - The average time taken per page.
        - The total number of pages.
    """
    print(f"▶️ Starting parallel processing for: {pdf_path}")

    # --- Get page count first ---
    try:
        with pymupdf.open(pdf_path) as doc:
            num_pages = doc.page_count
    except Exception as e:
        print(f"❌ Could not open PDF to get page count: {e}")
        return None, 0, 0, 0
    
    if num_pages == 0:
        print("⚠️ Document has no pages.")
        return "", 0, 0, 0

    # --- Time the entire extraction process ---
    t0 = time.perf_counter()

    page_results = [None] * num_pages  # Pre-allocate list to store results in order

    # Use a ProcessPoolExecutor to run page processing in parallel
    with ProcessPoolExecutor() as executor:
        # Create a future for each page
        futures = {executor.submit(process_page, pdf_path, i): i for i in range(num_pages)}
        
        # As each future completes, store its result
        for future in as_completed(futures):
            page_num, md_text = future.result()
            page_results[page_num] = md_text

    # --- End of timing ---
    total_time = time.perf_counter() - t0

    # --- Combine results and calculate stats ---
    full_markdown_text = "\n\n".join(page_results)
    avg_time_per_page = total_time / num_pages

    return full_markdown_text, total_time, avg_time_per_page, num_pages


def main():
    """
    Main function to handle command-line arguments and orchestrate the process.
    """
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF and measure performance using pymupdf4llm."
    )
    parser.add_argument(
        "pdf_file",
        type=pathlib.Path,
        help="The path to the PDF file to process."
    )
    args = parser.parse_args()
    pdf_path = args.pdf_file

    if not pdf_path.exists() or not pdf_path.is_file():
        print(f"❌ Error: File not found or is not a file: {pdf_path}")
        sys.exit(1)

    # --- Run the extraction using the new parallel function ---
    extracted_text, total_duration, avg_per_page, page_count = extract_text_with_timing_parallel(str(pdf_path))

    if extracted_text is None:
        sys.exit(1)

    # --- Print performance metrics ---
    print("\n--- Performance Report (Parallel) ---")
    print(f"📄 Total Pages:   {page_count}")
    print(f"⏱️  Total Time:    {total_duration:.3f} seconds")
    print(f"📊 Avg/Page:      {avg_per_page:.4f} seconds")
    print("-------------------------------------\n")

    output_path = pdf_path.with_suffix(".md")
    
    try:
        output_path.write_bytes(extracted_text.encode('utf-8'))
        print(f"✅ Successfully saved extracted text to: {output_path}")
    except IOError as e:
        print(f"❌ Error saving file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
