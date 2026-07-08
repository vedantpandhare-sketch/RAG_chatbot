import re
import hashlib
import unicodedata
from collections import Counter
from typing import List, Tuple, Iterable
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


def doc_fingerprint(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    page_number = str(doc.metadata.get("page_number", ""))
    digest = hashlib.sha1(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()
    return f"{source}|{page_number}|{digest}"


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


def extractive_marathi_answer_strict(question: str, docs: List[Document], min_score: float = 0.12) -> str:
    """
    Marathi answer selector that extracts the best matching sentences.
    
    Key behaviours:
    - Searches both doc.page_content (small chunk) AND parent_chunk metadata
      (the larger medium chunk stored during ingestion) to maximise recall on
      short OCR fragments.
    - Requires at least 1 content-token overlap (not 2) so short chunks are
      not silently discarded.
    - Falls back to a refusal message only when no sentence reaches min_score.
    """
    if not docs:
        return "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही."

    q_tokens = Counter(tokenize(question))
    q_numbers = numbers_in_text(question)
    q_content_tokens = [t for t in q_tokens if len(t) >= 2]

    scored_sentences = []
    for doc in docs:
        # Build the text pool: small chunk + parent_chunk if available
        texts_to_search = [doc.page_content or ""]
        parent_chunk = doc.metadata.get("parent_chunk", "")
        if parent_chunk and parent_chunk != doc.page_content:
            texts_to_search.append(parent_chunk)
        combined_text = "\n".join(texts_to_search)

        sentences = re.split(r"(?<=[।.!?])\s+|\n+", combined_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 8:
                continue
            sent_tokens = Counter(tokenize(sentence))
            overlap = sum(1 for token in q_content_tokens if sent_tokens.get(token))
            num_overlap = bool(q_numbers and numbers_in_text(sentence) & q_numbers)
            # Lowered threshold: need at least 1 matching token (was 2)
            if overlap < 1 and not num_overlap:
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
        source = doc.metadata.get("source_pdf", doc.metadata.get("source", "unknown source"))
        page_number = doc.metadata.get("page_number")
        page_info = f", पान {page_number}" if page_number is not None else ""
        picked.append(f"- {sentence} [{source}{page_info}]")
        if len(picked) >= 3:  # return up to 3 sentences for richer answers
            break

    if not picked:
        return "मला या प्रश्नासाठी दस्तऐवजांमध्ये पुरेशी माहिती सापडली नाही."

    return "दस्तऐवजानुसार:\n" + "\n".join(picked)



def validate_marathi_answer(answer: str, context: str, require_marathi: bool = True) -> Tuple[bool, str]:
    answer = (answer or "").strip()
    if not answer:
        return False, "रिकामे उत्तर"

    if require_marathi and not contains_devanagari(answer):
        return False, "उत्तर मराठीत नाही"

    if latin_letter_ratio(answer) > 0.6:
        return False, "उत्तरात खूप इंग्रजी आहे"

    return True, "ok"


def refusal_message() -> str:
    return "माझ्याकडे दिलेल्या दस्तऐवजांवर आधारित पुरेशी माहिती नाही."


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
