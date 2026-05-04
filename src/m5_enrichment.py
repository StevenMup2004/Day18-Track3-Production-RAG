"""
Module 5: Enrichment Pipeline
==============================
Lam giau chunks TRUOC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _call_openai(system_prompt: str, user_prompt: str, max_tokens: int = 180) -> str | None:
    """Optional LLM helper. Returns None when API/module is unavailable."""
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except Exception:
        return None


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    if not text.strip():
        return ""

    llm_summary = _call_openai(
        "Tom tat doan van sau trong 2-3 cau ngan gon bang tieng Viet.",
        text,
        max_tokens=120,
    )
    if llm_summary:
        return llm_summary

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return text[:180].strip()
    summary = " ".join(sentences[:2]).strip()
    if not summary.endswith((".", "!", "?")):
        summary += "."
    return summary[:300]


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    text = text.strip()
    if not text:
        return []

    llm_questions = _call_openai(
        f"Duoc giao mot doan van. Tao {n_questions} cau hoi ma doan van co the tra loi. Moi dong mot cau hoi.",
        text,
        max_tokens=180,
    )
    if llm_questions:
        lines = [ln.strip() for ln in llm_questions.splitlines() if ln.strip()]
        cleaned = [re.sub(r"^[0-9\-\.\)\s]+", "", ln) for ln in lines]
        cleaned = [q if q.endswith("?") else f"{q}?" for q in cleaned if q]
        return cleaned[:n_questions]

    lower = text.lower()
    questions: list[str] = []
    if "nghỉ phép" in lower:
        questions.append("Nhan vien duoc nghi phep bao nhieu ngay moi nam?")
        questions.append("Dieu kien ap dung chinh sach nghi phep la gi?")
    if "mật khẩu" in lower or "mat khau" in lower:
        questions.append("Mat khau can thay doi theo chu ky nao?")
    if "thử việc" in lower or "thu viec" in lower:
        questions.append("Thoi gian thu viec duoc quy dinh bao lau?")

    if not questions:
        # Generic fallback from first noun-like phrase.
        phrase = " ".join(re.findall(r"[0-9A-Za-zÀ-ỹ]+", text, flags=re.UNICODE)[:8]).strip()
        questions = [
            f"Noi dung chinh cua doan van ve '{phrase}' la gi?",
            "Doan van nay tra loi cho cau hoi nao?",
            "Thong tin quan trong nhat trong doan van la gi?",
        ]
    return questions[:max(1, n_questions)]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    if not text:
        return text

    prompt_input = f"Tai lieu: {document_title}\n\nDoan van:\n{text}"
    llm_context = _call_openai(
        "Viet 1 cau ngan mo ta doan van nay nam o dau trong tai lieu va noi ve chu de gi.",
        prompt_input,
        max_tokens=80,
    )
    if llm_context:
        return f"{llm_context}\n\n{text}"

    title = document_title.strip() or "tai lieu"
    first_sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    context = f"Trich tu {title}, phan noi dung lien quan den: {first_sentence[:120]}".strip()
    if not context.endswith("."):
        context += "."
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    text = text.strip()
    if not text:
        return {}

    llm_json = _call_openai(
        'Trich xuat metadata tu doan van. Tra ve JSON {"topic":"...","entities":["..."],"category":"policy|hr|it|finance|general","language":"vi|en"}',
        text,
        max_tokens=140,
    )
    if llm_json:
        try:
            parsed = json.loads(llm_json)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    lower = text.lower()
    if any(k in lower for k in ["nghỉ phép", "nghi phep", "chính sách", "quy định", "quy dinh"]):
        category = "policy"
    elif any(k in lower for k in ["nhân viên", "thu viec", "thử việc", "hr", "phúc lợi", "phuc loi"]):
        category = "hr"
    elif any(k in lower for k in ["mật khẩu", "mat khau", "vpn", "it", "bảo mật", "bao mat"]):
        category = "it"
    elif any(k in lower for k in ["doanh thu", "chi phí", "chi phi", "lợi nhuận", "loi nhuan", "tài chính", "tai chinh"]):
        category = "finance"
    else:
        category = "general"

    entities = re.findall(r"\b[A-ZĐ][a-zà-ỹA-ZÀ-Ỹ0-9]+\b", text)
    topic_tokens = re.findall(r"[0-9A-Za-zÀ-ỹ]+", text, flags=re.UNICODE)[:8]
    topic = " ".join(topic_tokens)
    language = "vi" if re.search(r"[à-ỹÀ-ỸđĐ]", text) else "en"

    return {
        "topic": topic.strip(),
        "entities": sorted(set(entities))[:10],
        "category": category,
        "language": language,
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    for chunk in chunks:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        if not text:
            continue

        use_full = "full" in methods
        summary = summarize_chunk(text) if (use_full or "summary" in methods) else ""
        questions = generate_hypothesis_questions(text) if (use_full or "hyqa" in methods) else []
        enriched_text = contextual_prepend(text, metadata.get("source", "")) if (use_full or "contextual" in methods) else text
        auto_meta = extract_metadata(text) if (use_full or "metadata" in methods) else {}

        if questions:
            enriched_text = (
                f"{enriched_text}\n\nCau hoi lien quan:\n"
                + "\n".join(f"- {q}" for q in questions)
            )
        if summary and (use_full or "summary" in methods):
            enriched_text = f"Tom tat: {summary}\n\n{enriched_text}"

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text or text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**metadata, **auto_meta},
                method="+".join(methods),
            )
        )

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
