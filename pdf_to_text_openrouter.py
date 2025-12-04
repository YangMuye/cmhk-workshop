#!/usr/bin/env python3
"""
PDF → Text via OpenRouter (rednote)

Renders each PDF page to a 300 DPI PNG in memory and sends the image
to OpenRouter using the specified vision-capable model (default: 'rednote').
The model response is aggregated and written to a single .txt file per PDF.

Requirements:
  - Environment variable OPENROUTER_API_KEY must be set
  - Dependencies: pymupdf, opencv-python, requests

Usage:
  python pdf_to_text_openrouter.py input.pdf -o out_texts
  python pdf_to_text_openrouter.py pdfs/ -o out_texts --max-pages 10
  python pdf_to_text_openrouter.py input.pdf --model rednote --dpi 300
"""
from __future__ import annotations

import argparse
import base64
import os
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import cv2
import fitz  # type: ignore[import-not-found]
import numpy as np
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def require_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")
    return api_key


def render_pdf_pages(pdf_path: Path, dpi: int = 300) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Render each page to BGR numpy array (OpenCV).
    """
    scale = dpi / 72.0
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yield idx, img


def encode_png_base64(image: np.ndarray) -> str:
    """
    Encode a BGR image to PNG and return base64 string (no data URL prefix).
    """
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG")
    return base64.b64encode(buf).decode("utf-8")


def openrouter_vision_ocr(
    api_key: str,
    model: str,
    image_b64: str,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> str:
    """
    Call OpenRouter chat completions with a single image and return text.
    Uses OpenAI-compatible schema with 'image_url' content using base64 data URL.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://example.com",
        "X-Title": "PDF OCR Pipeline",
        "Content-Type": "application/json",
    }
    data_url = f"data:image/png;base64,{image_b64}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an OCR assistant. Extract and return only the plain text content from the provided image. Do not include any additional commentary.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    }
                ],
            },
        ],
        "temperature": 0.0,
    }

    for attempt in range(1, max_retries + 1):
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)
        if resp.status_code == 200:
            js = resp.json()
            try:
                return js["choices"][0]["message"]["content"].strip()
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Unexpected OpenRouter response shape: {e} | {js}")

        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            time.sleep(backoff_seconds * attempt)
            continue
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

    raise RuntimeError("Max retries exceeded")


def iter_pdfs(target: Path) -> Iterator[Path]:
    if target.is_file() and target.suffix.lower() == ".pdf":
        yield target
        return
    if target.is_dir():
        for pdf in sorted(target.rglob("*.pdf")):
            if pdf.is_file():
                yield pdf
        return
    raise FileNotFoundError(f"Not a PDF file or directory: {target}")


def extract_with_openrouter_for_pdf(
    pdf_path: Path,
    api_key: str,
    model: str,
    dpi: int,
    max_pages: Optional[int],
) -> str:
    """
    Render pages and call OpenRouter for each. Return aggregated text.
    """
    texts: list[str] = []
    for page_num, image in render_pdf_pages(pdf_path, dpi=dpi):
        if max_pages is not None and page_num > max_pages:
            break
        print(f"  Page {page_num}: rendering and sending to model...")
        b64 = encode_png_base64(image)
        text = openrouter_vision_ocr(api_key, model, b64)
        texts.append(text)
    return "\n\n".join(texts)


def write_output(text: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"'{output_path}' exists (use --overwrite to replace)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def process_target(
    input_path: Path,
    output_dir: Optional[Path],
    model: str,
    dpi: int,
    max_pages: Optional[int],
    overwrite: bool,
) -> None:
    api_key = require_api_key()
    for pdf_path in iter_pdfs(input_path):
        print(f"Processing: {pdf_path}")
        txt = extract_with_openrouter_for_pdf(
            pdf_path=pdf_path,
            api_key=api_key,
            model=model,
            dpi=dpi,
            max_pages=max_pages,
        )
        out_path = (output_dir or pdf_path.parent) / (pdf_path.stem + ".txt")
        try:
            write_output(txt, out_path, overwrite=overwrite)
        except FileExistsError as e:
            print(f"  [SKIP] {e}")
        else:
            print(f"  [OK] Saved to: {out_path}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert PDF pages to text using OpenRouter vision model (rednote).",
    )
    p.add_argument("input", type=Path, help="PDF file or directory containing PDFs")
    p.add_argument("-o", "--output-dir", type=Path, help="Directory to store .txt outputs")
    p.add_argument("--dpi", type=int, default=300, help="Rendering DPI (default: 300)")
    p.add_argument("--model", type=str, default="rednote", help="OpenRouter model ID (default: rednote)")
    p.add_argument("--max-pages", type=int, help="Process at most N pages per PDF")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    process_target(
        input_path=args.input,
        output_dir=args.output_dir,
        model=args.model,
        dpi=args.dpi,
        max_pages=args.max_pages,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

