#!/usr/bin/env python3
"""
Helper script to batch-convert PDF files to Markdown using PyMuPDF4LLM.

This script scans the 'pdfs' directory for PDF files and converts them to
markdown format, saving the output with the same base name (e.g., file.pdf -> file.md).
"""
import pymupdf.layout
import pymupdf4llm
from pathlib import Path
import sys

def extract_markdown_from_pdf(pdf_path, output_path=None):
    """
    Extract markdown from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional custom output path. If None, uses same name with .md extension
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Processing: {pdf_path}")
        
        # Convert PDF to markdown
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        
        # Determine output path
        if output_path is None:
            output_path = Path(pdf_path).with_suffix('.md')
        
        # Write markdown to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_text)
        
        print(f"  ✓ Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def batch_extract(directory="pdfs"):
    """
    Batch extract markdown from all PDFs in a directory.
    
    Args:
        directory: Directory containing PDF files
    """
    pdf_dir = Path(directory)
    
    if not pdf_dir.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{directory}'")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s)\n")
    
    # Process each PDF
    success_count = 0
    for pdf_file in pdf_files:
        if extract_markdown_from_pdf(pdf_file):
            success_count += 1
        print()
    
    # Summary
    print(f"Completed: {success_count}/{len(pdf_files)} files processed successfully")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Process specific file or directory from command line
        path = Path(sys.argv[1])
        
        if path.is_file() and path.suffix.lower() == '.pdf':
            # Single file
            extract_markdown_from_pdf(path)
        elif path.is_dir():
            # Directory
            batch_extract(str(path))
        else:
            print(f"Error: '{path}' is not a valid PDF file or directory")
    else:
        # Default: process all PDFs in the 'pdfs' directory
        batch_extract("pdfs")

if __name__ == "__main__":
    main()

