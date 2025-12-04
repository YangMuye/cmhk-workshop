#!/usr/bin/env python3
"""
Convert PDF pages to PNG images at 300 DPI using PyMuPDF (fitz).

Usage:
  python pdf_to_png.py input.pdf -o out_dir
  python pdf_to_png.py path/to/dir/with/pdfs -o out_dir --overwrite

Requires:
  pip install pymupdf opencv-python
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import cv2
import fitz  # type: ignore[import-not-found]  # PyMuPDF
import numpy as np


def render_pdf_pages(pdf_path: Path, dpi: int = 300) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Render each PDF page to a BGR numpy array suitable for OpenCV.
    Yields (page_number, image).
    """
    scale = dpi / 72.0  # PyMuPDF default resolution is 72 DPI
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yield idx, img


def save_png(image: np.ndarray, path: Path, overwrite: bool) -> None:
    """
    Save a numpy image array to a PNG file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"Exists (use --overwrite): {path}")
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write: {path}")


def iter_pdfs(target: Path) -> Iterator[Path]:
    """
    Yield PDF files from a single file or recursively from a directory.
    """
    if target.is_file() and target.suffix.lower() == ".pdf":
        yield target
        return
    if target.is_dir():
        for pdf in sorted(target.rglob("*.pdf")):
            if pdf.is_file():
                yield pdf
        return
    raise FileNotFoundError(f"Not a PDF file or directory: {target}")


def convert_pdf_to_pngs(
    input_path: Path,
    output_dir: Optional[Path],
    overwrite: bool,
    dpi: int = 300,
) -> None:
    """
    Convert one or many PDFs to PNG images, one subfolder per PDF.
    """
    for pdf_path in iter_pdfs(input_path):
        print(f"Processing: {pdf_path}")
        base_dir = (output_dir or pdf_path.parent) / pdf_path.stem
        count = 0
        for page_num, image in render_pdf_pages(pdf_path, dpi=dpi):
            out_path = base_dir / f"{pdf_path.stem}_p{page_num:04d}.png"
            try:
                save_png(image, out_path, overwrite)
                count += 1
            except FileExistsError as e:
                print(f"  [SKIP] {e}")
            except Exception as e:
                print(f"  [ERROR] Page {page_num}: {e}")
        print(f"  [OK] Saved {count} page(s) to {base_dir}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF pages to PNG images at 300 DPI (PyMuPDF).")
    parser.add_argument("input", type=Path, help="PDF file or directory containing PDFs")
    parser.add_argument("-o", "--output-dir", type=Path, help="Directory to store images (default: alongside PDF)")
    parser.add_argument("--dpi", type=int, default=300, help="Rendering DPI (default: 300)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    convert_pdf_to_pngs(args.input, args.output_dir, args.overwrite, dpi=args.dpi)


if __name__ == "__main__":
    main()

