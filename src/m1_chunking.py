"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, va structure-aware chunking.
So sanh voi basic chunking (baseline) de thay improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re, math
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Best-effort PDF extraction with optional dependencies."""
    for module_name in ("pypdf", "PyPDF2"):
        try:
            if module_name == "pypdf":
                from pypdf import PdfReader  # type: ignore
            else:
                from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(path)
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            text = "\n".join(pages).strip()
            if text:
                return text
        except Exception:
            continue
    return ""


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown/text/pdf files from data/."""
    docs = []

    for pattern in ("*.md", "*.txt"):
        for fp in sorted(glob.glob(os.path.join(data_dir, pattern))):
            with open(fp, encoding="utf-8") as f:
                docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    pdf_docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            pdf_docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
    docs.extend(pdf_docs)

    if not docs:
        # Keep pipeline runnable in fully offline/broken extraction situations.
        docs.append(
            {
                "text": "Khong the trich xuat du lieu tu thu muc data trong moi truong hien tai.",
                "metadata": {"source": "fallback_empty_data"},
            }
        )

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    embeddings = None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)
    except Exception:
        embeddings = None

    def lexical_similarity(a: str, b: str) -> float:
        ta = {w for w in re.findall(r"\w+", a.lower()) if len(w) > 1}
        tb = {w for w in re.findall(r"\w+", b.lower()) if len(w) > 1}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(min(len(ta), len(tb)), 1)

    def cosine(v1, v2) -> float:
        dot = sum(float(x) * float(y) for x, y in zip(v1, v2))
        n1 = math.sqrt(sum(float(x) * float(x) for x in v1))
        n2 = math.sqrt(sum(float(y) * float(y) for y in v2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    chunks = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        if embeddings is not None:
            sim = cosine(embeddings[i - 1], embeddings[i])
        else:
            sim = lexical_similarity(sentences[i - 1], sentences[i])
        if sim < threshold:
            chunks.append(
                Chunk(
                    text=" ".join(current_group).strip(),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
                )
            )
            current_group = []
        current_group.append(sentences[i])

    if current_group:
        chunks.append(
            Chunk(
                text=" ".join(current_group).strip(),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            )
        )
    return [c for c in chunks if c.text]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [], []

    parents: list[Chunk] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) > parent_size and current:
            pid = f"parent_{len(parents)}"
            parents.append(
                Chunk(
                    text=current.strip(),
                    metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
                )
            )
            current = para
        else:
            current = candidate
    if current.strip():
        pid = f"parent_{len(parents)}"
        parents.append(
            Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            )
        )

    children: list[Chunk] = []
    stride = max(1, int(child_size * 0.8))
    for parent in parents:
        pid = parent.metadata.get("parent_id", "")
        ptext = parent.text
        if len(ptext) <= child_size:
            children.append(
                Chunk(
                    text=ptext,
                    metadata={**metadata, "chunk_type": "child", "child_index": 0},
                    parent_id=pid,
                )
            )
            continue
        child_index = 0
        for start in range(0, len(ptext), stride):
            snippet = ptext[start:start + child_size].strip()
            if not snippet:
                continue
            children.append(
                Chunk(
                    text=snippet,
                    metadata={**metadata, "chunk_type": "child", "child_index": child_index},
                    parent_id=pid,
                )
            )
            child_index += 1
            if start + child_size >= len(ptext):
                break

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    header_pattern = re.compile(r"^#{1,3}\s+.+$", flags=re.MULTILINE)
    matches = list(header_pattern.finditer(text))

    if not matches:
        return [
            Chunk(
                text=text.strip(),
                metadata={**metadata, "section": "full_document", "strategy": "structure"},
            )
        ] if text.strip() else []

    chunks: list[Chunk] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        header = match.group(0).strip()
        if section_text:
            chunks.append(
                Chunk(
                    text=section_text,
                    metadata={**metadata, "section": header, "strategy": "structure", "chunk_index": i},
                )
            )

    # Include leading text before first header as preface section if any.
    first_start = matches[0].start()
    preface = text[:first_start].strip()
    if preface:
        chunks.insert(
            0,
            Chunk(
                text=preface,
                metadata={**metadata, "section": "preface", "strategy": "structure", "chunk_index": -1},
            ),
        )
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    def summarize(chunks: list[Chunk]) -> dict:
        lengths = [len(c.text) for c in chunks if c.text]
        if not lengths:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        return {
            "num_chunks": len(lengths),
            "avg_length": round(sum(lengths) / len(lengths), 2),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    basic_chunks: list[Chunk] = []
    semantic_chunks: list[Chunk] = []
    structure_chunks: list[Chunk] = []
    hierarchical_parents: list[Chunk] = []
    hierarchical_children: list[Chunk] = []

    for doc in documents:
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        basic_chunks.extend(chunk_basic(text, metadata=meta))
        semantic_chunks.extend(chunk_semantic(text, metadata=meta))
        parents, children = chunk_hierarchical(text, metadata=meta)
        hierarchical_parents.extend(parents)
        hierarchical_children.extend(children)
        structure_chunks.extend(chunk_structure_aware(text, metadata=meta))

    results = {
        "basic": summarize(basic_chunks),
        "semantic": summarize(semantic_chunks),
        "hierarchical": {
            "num_parents": len(hierarchical_parents),
            "num_children": len(hierarchical_children),
            **summarize(hierarchical_children),
        },
        "structure": summarize(structure_chunks),
    }

    print("\nStrategy Comparison")
    print("-" * 72)
    print(f"{'Strategy':<14} {'Chunks':>12} {'Avg Len':>10} {'Min':>8} {'Max':>8}")
    print("-" * 72)
    print(
        f"{'basic':<14} {results['basic']['num_chunks']:>12} "
        f"{results['basic']['avg_length']:>10} {results['basic']['min_length']:>8} {results['basic']['max_length']:>8}"
    )
    print(
        f"{'semantic':<14} {results['semantic']['num_chunks']:>12} "
        f"{results['semantic']['avg_length']:>10} {results['semantic']['min_length']:>8} {results['semantic']['max_length']:>8}"
    )
    h = results["hierarchical"]
    print(
        f"{'hierarchical':<14} {str(h['num_parents']) + 'p/' + str(h['num_children']) + 'c':>12} "
        f"{h['avg_length']:>10} {h['min_length']:>8} {h['max_length']:>8}"
    )
    print(
        f"{'structure':<14} {results['structure']['num_chunks']:>12} "
        f"{results['structure']['avg_length']:>10} {results['structure']['min_length']:>8} {results['structure']['max_length']:>8}"
    )
    print("-" * 72)

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
