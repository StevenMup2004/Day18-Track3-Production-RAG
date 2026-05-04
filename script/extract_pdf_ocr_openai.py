"""OCR scanned PDFs to Markdown using OpenAI Vision.

Usage example:
  python scripts/extract_pdf_ocr_openai.py ^
    --pdf data/Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf ^
    --out data/Nghi_dinh_13_2023_extracted_ocr.md ^
    --model gpt-4o-mini ^
    --zoom 2.0
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI


def parse_pages(spec: str | None, total_pages: int) -> list[int]:
    """Parse page selection string into 0-based page indices."""
    if not spec:
        return list(range(total_pages))

    picked: set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left) if left else 1
            end = int(right) if right else total_pages
            for p in range(start, end + 1):
                if 1 <= p <= total_pages:
                    picked.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= total_pages:
                picked.add(p - 1)

    return sorted(picked)


def render_page_png(doc: fitz.Document, page_index: int, zoom: float) -> bytes:
    """Render one page into PNG bytes."""
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


def ocr_page(client: OpenAI, model: str, png_bytes: bytes, page_no: int) -> str:
    """OCR a single page with OpenAI Vision."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    prompt = (
        "Trich xuat van ban tu trang PDF nay theo kieu OCR.\n"
        "Yeu cau:\n"
        "1) Tra ve duy nhat noi dung van ban, khong mo ta them.\n"
        "2) Giu cau truc muc: tieu de, dieu, khoan, gach dau dong neu co.\n"
        "3) Neu trang khong doc duoc thi tra ve [UNREADABLE].\n"
        f"4) Day la trang so {page_no}."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Ban la OCR engine chinh xac cao cho van ban tieng Viet.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR scanned PDF to markdown using OpenAI Vision.")
    parser.add_argument("--pdf", required=True, help="Path to input PDF.")
    parser.add_argument("--out", required=True, help="Path to output markdown file.")
    parser.add_argument("--model", default=os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini"))
    parser.add_argument("--zoom", type=float, default=2.0, help="Render zoom factor for page images.")
    parser.add_argument("--pages", default=None, help="Page selection, e.g. '1-5,8,10-12'. Default: all.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    load_dotenv(root.parent / ".env")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is missing. Put it in .env.", file=sys.stderr)
        return 1

    pdf_path = Path(args.pdf).resolve()
    out_path = Path(args.out).resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)
    doc = fitz.open(str(pdf_path))
    indices = parse_pages(args.pages, len(doc))
    if not indices:
        print("No pages selected.", file=sys.stderr)
        return 1

    lines: list[str] = [
        f"# OCR EXTRACT - {pdf_path.stem}",
        "",
        f"- Source PDF: {pdf_path.name}",
        f"- Model: {args.model}",
        f"- Total pages in PDF: {len(doc)}",
        f"- Extracted pages: {len(indices)}",
        "",
    ]

    for i, page_idx in enumerate(indices, start=1):
        page_no = page_idx + 1
        print(f"[{i}/{len(indices)}] OCR page {page_no}...")
        png = render_page_png(doc, page_idx, zoom=args.zoom)
        text = ocr_page(client, args.model, png, page_no=page_no)
        lines.append(f"## Page {page_no}")
        lines.append("")
        lines.append(text if text else "[UNREADABLE]")
        lines.append("")
        if args.sleep > 0:
            time.sleep(args.sleep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
