#!/usr/bin/env python3
"""
OCR-based PDF text extraction utility.

This script renders PDF pages to images (via PyMuPDF), applies grayscale and
adaptive thresholding with OpenCV to enhance text regions, then extracts text
using Tesseract OCR (`pytesseract`). The resulting plain-text output is saved
alongside each PDF by default.

Prerequisites:
    * Tesseract OCR binary must be installed and discoverable on PATH.
      Windows installers: https://github.com/UB-Mannheim/tesseract/wiki
    * Run `pip install opencv-python pytesseract` (declared in project deps).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

import cv2
import fitz  # type: ignore[import-not-found]
import numpy as np
import pytesseract


def render_pdf_pages(pdf_path: Path, dpi: int = 300) -> Iterator[np.ndarray]:
    """
    Yield rendered images (as BGR numpy arrays) for each page in a PDF.

    Args:
        pdf_path: PDF document to render.
        dpi: Target resolution for rendering (affects OCR quality).

    Yields:
        OpenCV-compatible BGR arrays per page.
    """
    scale = dpi / 72  # PyMuPDF default DPI is 72

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
            )

            image = np.frombuffer(pix.samples, dtype=np.uint8)
            image = image.reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            yield image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Convert image to black-and-white to boost OCR accuracy.

    Steps:
        - Convert to grayscale
        - Apply Gaussian blur for noise reduction
        - Adaptive threshold to isolate text foreground
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return bw


def ocr_image(image: np.ndarray, lang: str = "eng") -> str:
    """
    Run Tesseract OCR on a preprocessed image.

    Args:
        image: OpenCV image (grayscale or binary).
        lang: Language(s) for Tesseract (default English).
    """
    return pytesseract.image_to_string(image, lang=lang, config="--psm 6")


def iter_pdf_files(target: Path) -> Iterator[Path]:
    """
    Yield PDF files from a file or directory recursively.
    """
    if target.is_file() and target.suffix.lower() == ".pdf":
        yield target
        return

    if target.is_dir():
        for pdf_file in sorted(target.rglob("*.pdf")):
            if pdf_file.is_file():
                yield pdf_file
        return

    raise FileNotFoundError(f"'{target}' is not a valid PDF file or directory")


def extract_text_from_pdf(
    pdf_path: Path,
    dpi: int = 300,
    lang: str = "eng",
) -> str:
    """
    Extract OCR text from a PDF.
    """
    page_texts: list[str] = []
    for page_index, image in enumerate(render_pdf_pages(pdf_path, dpi), start=1):
        processed = preprocess_image(image)
        text = ocr_image(processed, lang=lang)
        page_texts.append(text.strip())
        print(f"  Page {page_index}: {len(text.strip())} characters extracted")

    return "\n\n".join(page_texts)


def write_output(text: str, output_path: Path, overwrite: bool) -> None:
    """
    Write text to disk respecting overwrite preference.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"'{output_path}' exists (use --overwrite to replace it)",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def process_target(
    target: Path,
    output_dir: Optional[Path],
    overwrite: bool,
    dpi: int,
    lang: str,
) -> None:
    """
    Process a single file or directory of PDFs.
    """
    pdf_files = list(iter_pdf_files(target))
    if not pdf_files:
        print(f"No PDF files found in '{target}'")
        return

    print(f"Found {len(pdf_files)} PDF file(s)\n")

    for pdf_path in pdf_files:
        try:
            print(f"Processing: {pdf_path}")
            text = extract_text_from_pdf(pdf_path, dpi=dpi, lang=lang)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to extract from '{pdf_path}': {exc}")
            continue

        if output_dir:
            if target.is_dir():
                relative = pdf_path.relative_to(target)
                output_path = (output_dir / relative).with_suffix(".txt")
            else:
                output_path = (output_dir / pdf_path.name).with_suffix(".txt")
        else:
            output_path = pdf_path.with_suffix(".txt")

        try:
            write_output(text, output_path, overwrite=overwrite)
        except FileExistsError as exc:
            print(f"[SKIP] {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Could not write to '{output_path}': {exc}")
        else:
            print(f"[OK] Saved to: {output_path}\n")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Configure CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs using OpenCV preprocessing and Tesseract OCR.",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to a PDF file or directory containing PDFs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Optional directory to write output `.txt` files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rendering DPI for PDF pages (higher improves OCR, defaults to 300).",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Language(s) passed to Tesseract (default: 'eng').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing `.txt` files.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    try:
        process_target(
            target=args.input,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            dpi=args.dpi,
            lang=args.lang,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

