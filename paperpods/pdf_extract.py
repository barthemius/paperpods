from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

MATH_LINE_PATTERN = re.compile(r"(=|\\|/|\^|_|\b(sum|int|lim|argmax|argmin)\b)")


@dataclass(slots=True)
class ExtractedPaper:
    title: str
    text: str
    formula_snippets: list[str]


def extract_pdf_text(pdf_path: Path, *, max_chars: int = 60_000, max_formula_snippets: int = 20) -> ExtractedPaper:
    reader = PdfReader(str(pdf_path))
    raw_text_parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            raw_text_parts.append(text.strip())

    full_text = "\n\n".join(raw_text_parts).strip()
    if not full_text:
        raise ValueError("Could not extract text from PDF.")

    truncated_text = full_text[:max_chars]
    lines = [line.strip() for line in truncated_text.splitlines() if line.strip()]
    formula_snippets: list[str] = []
    for line in lines:
        if MATH_LINE_PATTERN.search(line) and len(line) < 180:
            formula_snippets.append(line)
        if len(formula_snippets) >= max_formula_snippets:
            break

    title = _infer_title(pdf_path=pdf_path, extracted_text=truncated_text)
    return ExtractedPaper(title=title, text=truncated_text, formula_snippets=formula_snippets)


def _infer_title(*, pdf_path: Path, extracted_text: str) -> str:
    first_lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
    if first_lines:
        candidate = first_lines[0]
        if 5 <= len(candidate) <= 200:
            return candidate
    return pdf_path.stem.replace("_", " ").replace("-", " ").title()

