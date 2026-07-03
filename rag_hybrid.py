from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from langchain_core.documents import Document


_WORD_RE = re.compile(r"[0-9A-Za-z\u0900-\u097F]+", re.UNICODE)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return _WORD_RE.findall(normalized)


def char_ngrams(text: str, n: int = 3) -> List[str]:
    normalized = normalize_text(text).replace(" ", "")
    if len(normalized) < n:
        return [normalized] if normalized else []
    return [normalized[i : i + n] for i in range(len(normalized) - n + 1)]


def doc_fingerprint(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    page_number = str(doc.metadata.get("page_number", ""))
    digest = hashlib.sha1(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()
    return f"{source}|{page_number}|{digest}"


def format_context(docs: Iterable[Document], max_chars_per_doc: int = 1200) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown source")
        page_number = doc.metadata.get("page_number")
        page_label = f" | page {page_number}" if page_number is not None else ""
        content = doc.page_content.strip()
        if max_chars_per_doc and len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc].rstrip() + "..."
        blocks.append(f"[Document {i} | {source}{page_label}]\n{content}")
    return "\n\n".join(blocks)


def contains_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text or ""))


def latin_letter_ratio(text: str) -> float:
    text = text or ""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for ch in letters if _LATIN_RE.match(ch))
    return latin / len(letters)


def numbers_in_text(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text or ""))


def validate_marathi_answer(answer: str, context: str, require_marathi: bool = True) -> tuple[bool, str]:
    answer = (answer or "").strip()
    if not answer:
        return False, "रिकामे उत्तर"

    if require_marathi and not contains_devanagari(answer):
        return False, "उत्तर मराठीत नाही"

    # Allow proper nouns / organization names, but reject answers that are
    # mostly English or transliterated noise.
    if latin_letter_ratio(answer) > 0.6:
        return False, "उत्तरात खूप इंग्रजी आहे"

    return True, "ok"


def refusal_message() -> str:
    return "माझ्याकडे दिलेल्या दस्तऐवजांवर आधारित पुरेशी माहिती नाही."


def _split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    pieces = re.split(r"(?<=[।.!?])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def _sentence_score(sentence: str, query_tokens: Counter[str], query_numbers: set[str]) -> float:
    sent_tokens = Counter(tokenize(sentence))
    if not sent_tokens:
        return 0.0

    overlap = sum(min(count, sent_tokens.get(token, 0)) for token, count in query_tokens.items())
    token_score = overlap / max(1, sum(query_tokens.values()))

    sent_numbers = numbers_in_text(sentence)
    number_score = 0.0
    if query_numbers:
        number_score = 1.0 if query_numbers & sent_numbers else 0.0

    # Slightly favor sentences with Devanagari content in Marathi docs.
    dev_bonus = 0.1 if contains_devanagari(sentence) else 0.0
    return token_score * 0.75 + number_score * 0.15 + dev_bonus


def extractive_marathi_answer(question: str, docs: List[Document], min_score: float = 0.12) -> str:
    """Return a short extractive answer built from the best matching sentences."""
    if not docs:
        return refusal_message()

    q_tokens = Counter(tokenize(question))
    q_numbers = numbers_in_text(question)

    scored_sentences = []
    for doc in docs:
        sentences = _split_sentences(doc.page_content)
        for sentence in sentences:
            score = _sentence_score(sentence, q_tokens, q_numbers)
            if score > 0:
                scored_sentences.append((score, doc, sentence))

    if not scored_sentences:
        return refusal_message()

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    top_score = scored_sentences[0][0]
    if top_score < min_score:
        return refusal_message()

    chosen = []
    used = set()
    for score, doc, sentence in scored_sentences:
        fp = doc_fingerprint(doc) + "|" + sentence[:80]
        if fp in used:
            continue
        used.add(fp)
        source = doc.metadata.get("source", "unknown source")
        page_number = doc.metadata.get("page_number")
        page_info = f" (पान {page_number})" if page_number is not None else ""
        chosen.append(f"- {sentence.strip()} [{source}{page_info}]")
        if len(chosen) >= 2:
            break

    if not chosen:
        return refusal_message()

    return "दस्तऐवजानुसार:\n" + "\n".join(chosen)


def extractive_marathi_answer_clean(question: str, docs: List[Document], min_score: float = 0.12) -> str:
    """Clean Marathi extractive answer for the app entry points."""
    if not docs:
        return "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही."

    q_tokens = Counter(tokenize(question))
    q_numbers = numbers_in_text(question)

    scored_sentences = []
    for doc in docs:
        for sentence in re.split(r"(?<=[।.!?])\s+|\n+", doc.page_content or ""):
            sentence = sentence.strip()
            if not sentence:
                continue
            score = _sentence_score(sentence, q_tokens, q_numbers)
            if score > 0:
                scored_sentences.append((score, doc, sentence))

    if not scored_sentences:
        return "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही."

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    if scored_sentences[0][0] < min_score:
        return "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही."

    picked = []
    seen = set()
    for score, doc, sentence in scored_sentences:
        fingerprint = doc_fingerprint(doc) + "|" + sentence[:80]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        source = doc.metadata.get("source", "unknown source")
        page_number = doc.metadata.get("page_number")
        page_info = f", पान {page_number}" if page_number is not None else ""
        picked.append(f"- {sentence} [{source}{page_info}]")
        if len(picked) >= 2:
            break

    if not picked:
        return "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही."

    return "दस्तऐवजानुसार:\n" + "\n".join(picked)


def extractive_marathi_answer_strict(question: str, docs: List[Document], min_score: float = 0.18) -> str:
    """Stricter Marathi answer selector that refuses weak matches."""
    if not docs:
        return "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही."

    q_tokens = Counter(tokenize(question))
    q_numbers = numbers_in_text(question)
    q_content_tokens = [t for t in q_tokens if len(t) >= 2]

    scored_sentences = []
    for doc in docs:
        doc_hybrid_score = float(doc.metadata.get("hybrid_score") or 0.0)
        doc_lexical_score = float(doc.metadata.get("lexical_score") or 0.0)
        if doc_hybrid_score and doc_hybrid_score < 0.18:
            continue
        if doc_lexical_score and doc_lexical_score < 0.08 and doc_hybrid_score < 0.22:
            continue

        sentences = re.split(r"(?<=[।.!?])\s+|\n+", doc.page_content or "")
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sent_tokens = Counter(tokenize(sentence))
            overlap = sum(1 for token in q_content_tokens if sent_tokens.get(token))
            num_overlap = bool(q_numbers and numbers_in_text(sentence) & q_numbers)
            if overlap < 2 and not num_overlap:
                continue
            score = _sentence_score(sentence, q_tokens, q_numbers)
            if score > 0:
                scored_sentences.append((score, doc, sentence))

    if not scored_sentences:
        return "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही."

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    if scored_sentences[0][0] < min_score:
        return "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही."

    picked = []
    seen = set()
    for score, doc, sentence in scored_sentences:
        fingerprint = doc_fingerprint(doc) + "|" + sentence[:80]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        source = doc.metadata.get("source", "unknown source")
        page_number = doc.metadata.get("page_number")
        page_info = f", पान {page_number}" if page_number is not None else ""
        picked.append(f"- {sentence} [{source}{page_info}]")
        if len(picked) >= 2:
            break

    if not picked:
        return "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही."

    return "दस्तऐवजानुसार:\n" + "\n".join(picked)


def load_documents_from_vector_store(db) -> List[Document]:
    payload = db.get()
    documents = payload.get("documents", []) or []
    metadatas = payload.get("metadatas", []) or []

    docs: List[Document] = []
    for text, metadata in zip(documents, metadatas):
        docs.append(
            Document(
                page_content=text or "",
                metadata=metadata or {},
            )
        )
    return docs


def build_marathi_rewrite_prompt(user_question: str, conversation_history: str) -> str:
    return f"""तुम्ही संभाषणातील मागील संदर्भ वापरून प्रश्न एकटाच (standalone) आणि शोधता येईल असा पुन्हा लिहा.
फक्त पुनर्लिखित प्रश्न द्या, इतर काही नाही.
प्रश्न मराठीतच ठेवा.

संभाषण इतिहास:
{conversation_history}

नवीन प्रश्न:
{user_question}
"""


def build_marathi_answer_prompt(user_question: str, context: str) -> str:
    return f"""फक्त खाली दिलेल्या दस्तऐवजांवर आधारित उत्तर द्या.
उत्तर मराठीत आणि देवनागरी लिपीत द्या.
इंग्रजी टाळा, फक्त आवश्यक नामे/संस्था नावं अपवाद असू शकतात.
संक्षिप्त, थेट आणि तथ्यात्मक उत्तर द्या.
जर उत्तर दस्तऐवजांमध्ये स्पष्टपणे नसेल, तर नेमके असे लिहा:
"माझ्याकडे दिलेल्या दस्तऐवजांवर आधारित पुरेशी माहिती नाही."

कडक नियम:
1. फक्त दिलेल्या दस्तऐवजांमधील माहिती वापरा.
2. अंदाज, कल्पना, किंवा बाह्य ज्ञान वापरू नका.
3. उत्तरात शक्य असल्यास दस्तऐवजातील अचूक शब्द किंवा अतिशय जवळचा मजकूर वापरा.
4. संदर्भात माहिती कमकुवत किंवा अपुरी असेल तर उत्तर देऊ नका.

प्रश्न:
{user_question}

दस्तऐवज:
{context}
"""


@dataclass
class _ScoredDoc:
    doc: Document
    semantic: float = 0.0
    lexical: float = 0.0

    @property
    def combined(self) -> float:
        if self.semantic and self.lexical:
            return 0.5 * self.semantic + 0.5 * self.lexical
        if self.semantic:
            return 0.8 * self.semantic
        return 0.7 * self.lexical


class HybridRetriever:
    @classmethod
    def from_vector_store(cls, db) -> "HybridRetriever":
        return cls(load_documents_from_vector_store(db))

    def __init__(self, docs: List[Document]):
        self._docs = docs
        self._lexical_index = []
        for doc in docs:
            norm = normalize_text(doc.page_content)
            token_counts = Counter(tokenize(norm))
            tri_set = set(char_ngrams(norm, 3))
            self._lexical_index.append((doc, norm, token_counts, tri_set))

    def _lexical_score(self, query: str, doc_idx: int) -> float:
        q_norm = normalize_text(query)
        q_tokens = Counter(tokenize(q_norm))
        q_tris = set(char_ngrams(q_norm, 3))
        if not q_tokens and not q_tris:
            return 0.0

        doc, norm, d_tokens, d_tris = self._lexical_index[doc_idx]

        token_score = 0.0
        if q_tokens:
            overlap = sum(min(count, d_tokens.get(token, 0)) for token, count in q_tokens.items())
            token_score = overlap / max(1, sum(q_tokens.values()))

        tri_score = 0.0
        if q_tris and d_tris:
            tri_score = len(q_tris & d_tris) / max(1, len(q_tris | d_tris))

        phrase_bonus = 0.0
        for token in q_tokens:
            if len(token) >= 3 and token in norm:
                phrase_bonus += 0.05
        phrase_bonus = min(0.2, phrase_bonus)

        return min(1.0, 0.7 * token_score + 0.3 * tri_score + phrase_bonus)

    def retrieve(
        self,
        db,
        query: str,
        top_k: int = 4,
        semantic_k: int = 12,
        lexical_k: int = 12,
        min_score: float = 0.12,
    ) -> Tuple[List[Document], List[dict]]:
        semantic_hits = []
        try:
            semantic_hits = db.similarity_search_with_relevance_scores(query, k=semantic_k)
        except Exception:
            semantic_hits = []

        scored = {}

        for doc, score in semantic_hits:
            key = doc_fingerprint(doc)
            scored[key] = _ScoredDoc(
                doc=doc,
                semantic=max(0.0, min(1.0, float(score))),
                lexical=0.0,
            )

        lexical_scores = []
        for idx, (doc, _, _, _) in enumerate(self._lexical_index):
            lexical_scores.append((idx, self._lexical_score(query, idx)))

        lexical_scores.sort(key=lambda item: item[1], reverse=True)
        for idx, score in lexical_scores[:lexical_k]:
            doc = self._lexical_index[idx][0]
            key = doc_fingerprint(doc)
            entry = scored.get(key)
            if entry is None:
                entry = _ScoredDoc(doc=doc)
                scored[key] = entry
            entry.lexical = max(entry.lexical, score)

        ranked = sorted(scored.values(), key=lambda item: item.combined, reverse=True)
        filtered = [item for item in ranked if item.combined >= min_score]

        selected = filtered[:top_k]
        docs = []
        debug = []
        for item in selected:
            doc = Document(
                page_content=item.doc.page_content,
                metadata={
                    **item.doc.metadata,
                    "hybrid_score": round(item.combined, 4),
                    "semantic_score": round(item.semantic, 4),
                    "lexical_score": round(item.lexical, 4),
                },
            )
            docs.append(doc)
            debug.append(
                {
                    "source": doc.metadata.get("source", "unknown source"),
                    "page_number": doc.metadata.get("page_number"),
                    "hybrid_score": doc.metadata["hybrid_score"],
                    "semantic_score": doc.metadata["semantic_score"],
                    "lexical_score": doc.metadata["lexical_score"],
                }
            )

        return docs, debug
