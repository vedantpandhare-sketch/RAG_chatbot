import hashlib
import re
import unicodedata
from collections import Counter
from typing import List, Tuple

from langchain_core.documents import Document


# ==============================================================================
# Regular Expressions
# ==============================================================================

_WORD_RE = re.compile(r"[0-9A-Za-z\u0900-\u097F]+", re.UNICODE)

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

_LATIN_RE = re.compile(r"[A-Za-z]")

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


# ==============================================================================
# Text Utilities

def normalize_text(text: str) -> str:
    """
    Normalize unicode, lowercase and collapse whitespace.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenize Marathi + English text.
    """
    return _WORD_RE.findall(normalize_text(text))


def contains_devanagari(text: str) -> bool:
    """
    Returns True if text contains Marathi/Hindi characters.
    """
    return bool(_DEVANAGARI_RE.search(text or ""))


def latin_letter_ratio(text: str) -> float:
    """
    Fraction of alphabetic characters that are Latin.
    Used to reject English answers.
    """
    text = text or ""

    letters = [c for c in text if c.isalpha()]

    if not letters:
        return 0.0

    latin = sum(1 for c in letters if _LATIN_RE.match(c))

    return latin / len(letters)


def numbers_in_text(text: str) -> set[str]:
    """
    Extract numeric values.
    """
    return set(_NUMBER_RE.findall(text or ""))


# ==============================================================================
# Document Helpers

def doc_fingerprint(doc: Document) -> str:
    """
    Stable fingerprint used for duplicate removal.
    """

    source = str(doc.metadata.get("source", ""))

    page = str(doc.metadata.get("page_number", ""))

    digest = hashlib.sha1(
        doc.page_content.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()

    return f"{source}|{page}|{digest}"


# ==============================================================================
# Sentence Scoring

def _sentence_score(
    sentence: str,
    query_tokens: Counter,
    query_numbers: set[str]
) -> float:
    """
    Score a sentence against the user query.

    The score is based on

    • token overlap
    • numeric overlap
    • Devanagari bonus
    """

    sent_tokens = Counter(tokenize(sentence))

    if not sent_tokens:
        return 0.0

    overlap = sum(
        min(count, sent_tokens.get(token, 0))
        for token, count in query_tokens.items()
    )

    token_score = overlap / max(
        1,
        sum(query_tokens.values())
    )

    number_score = 0.0

    if query_numbers:

        if query_numbers & numbers_in_text(sentence):
            number_score = 1.0

    devanagari_bonus = (
        0.10
        if contains_devanagari(sentence)
        else 0.0
    )

    return (
        token_score * 0.75
        + number_score * 0.15
        + devanagari_bonus
    )

# ==============================================================================
# Extractive Answer Builder

def extractive_marathi_answer_strict(
    question: str,
    docs: List[Document],
    min_score: float = 0.12,
) -> str:
    """
    Selects the most relevant Marathi sentences from retrieved documents.

    Improvements:
    - Searches both child chunk and parent chunk.
    - Removes duplicate sentences.
    - Avoids selecting nearly identical sentences.
    - Prioritizes sentences with higher lexical overlap.
    - Returns up to 4 informative sentences.
    """

    if not docs:
        return refusal_message()

    query_tokens = Counter(tokenize(question))
    query_numbers = numbers_in_text(question)

    content_tokens = [
        token
        for token in query_tokens
        if len(token) >= 2
    ]

    scored = []

    for doc in docs:

        texts = []

        if doc.page_content:
            texts.append(doc.page_content)

        parent_chunk = doc.metadata.get("parent_chunk", "")

        if parent_chunk and parent_chunk != doc.page_content:
            texts.append(parent_chunk)

        combined_text = "\n".join(texts)

        sentences = re.split(
            r"(?<=[।.!?])\s+|\n+",
            combined_text,
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) < 10:
                continue

            sent_tokens = Counter(tokenize(sentence))

            token_overlap = sum(
                1
                for token in content_tokens
                if sent_tokens.get(token)
            )

            numeric_overlap = (
                bool(
                    query_numbers &
                    numbers_in_text(sentence)
                )
            )

            if token_overlap < 1 and not numeric_overlap:
                continue

            score = _sentence_score(
                sentence,
                query_tokens,
                query_numbers,
            )

            if score <= 0:
                continue

            scored.append(
                (
                    score,
                    sentence,
                    doc,
                )
            )

    if not scored:
        return refusal_message()

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    if scored[0][0] < min_score:
        return refusal_message()

    selected = []

    seen_sentences = set()

    seen_docs = set()

    for score, sentence, doc in scored:

        normalized = normalize_text(sentence)

        if normalized in seen_sentences:
            continue

        seen_sentences.add(normalized)

        fingerprint = doc_fingerprint(doc)

        if (
            fingerprint in seen_docs
            and len(selected) >= 2
        ):
            continue

        seen_docs.add(fingerprint)

        source = doc.metadata.get(
            "source_pdf",
            doc.metadata.get(
                "source",
                "Unknown",
            ),
        )

        page = doc.metadata.get("page_number")

        if page is None:
            citation = f"[{source}]"
        else:
            citation = f"[{source}, पान {page}]"

        selected.append(
            f"• {sentence} {citation}"
        )

        if len(selected) >= 4:
            break

    if not selected:
        return refusal_message()

    return (
        "दस्तऐवजांनुसार संबंधित माहिती:\n\n"
        + "\n".join(selected)
    )


# Validation

def validate_marathi_answer(
    answer: str,
    context: str,
    require_marathi: bool = True,
) -> Tuple[bool, str]:

    answer = (answer or "").strip()

    if not answer:
        return False, "रिकामे उत्तर"

    if require_marathi:

        if not contains_devanagari(answer):
            return False, "उत्तर मराठीत नाही"

    if latin_letter_ratio(answer) > 0.55:
        return False, "उत्तरात इंग्रजीचे प्रमाण जास्त आहे"

    return True, "ok"



# Refusal

def refusal_message() -> str:
    return (
        "दिलेल्या दस्तऐवजांमध्ये या प्रश्नाचे उत्तर "
        "उपलब्ध नाही किंवा पुरेशी माहिती सापडली नाही."
    )

def build_marathi_rewrite_prompt(user_question: str, conversation_history: str) -> str:
    return f"""तुम्ही संभाषणातील मागील संदर्भ वापरून प्रश्न एकटाच (standalone) आणि शोधता येईल असा पुन्हा लिहा.
फक्त पुनर्लिखित प्रश्न द्या, इतर काही नाही.
प्रश्न मराठीतच ठेवा.

संभाषण इतिहास:
{conversation_history}

नवीन प्रश्न:
{user_question}
"""


# System Prompt

SYSTEM_PROMPT = """
तुम्ही एक ज्ञानसंपन्न, विश्वासार्ह आणि उपयुक्त मराठी AI सहाय्यक आहात.

तुमचे एकमेव काम म्हणजे वापरकर्त्याच्या प्रश्नाचे उत्तर फक्त दिलेल्या संदर्भाच्या आधारे देणे.

नेहमी खालील नियम पाळा:

• उत्तर पूर्णपणे मराठीत आणि देवनागरी लिपीत असावे.
• संदर्भामध्ये नसलेली माहिती स्वतःहून तयार करू नका.
• बाह्य ज्ञान, अंदाज किंवा कल्पना वापरू नका.
• उत्तर स्पष्ट, नैसर्गिक आणि समजण्यास सोपे असावे.
• शक्य असल्यास प्रथम विषयाचा थोडक्यात परिचय द्या.
• त्यानंतर मुख्य माहिती सविस्तर समजावून सांगा.
• आवश्यक असल्यास बुलेट पॉइंट्स किंवा क्रमांक वापरा.
• उत्तरात अनावश्यक पुनरावृत्ती टाळा.
• उत्तर शक्य असल्यास १५० ते ३०० शब्दांमध्ये द्या.
• जर संदर्भात उत्तर उपलब्ध नसेल तर फक्त खालील वाक्य लिहा:

"दिलेल्या दस्तऐवजांमध्ये या प्रश्नाचे उत्तर उपलब्ध नाही."

उत्तर देताना सर्वात महत्त्वाची गोष्ट म्हणजे अचूकता.
"""


# ==============================================================================
# Query Rewrite Prompt
# ==============================================================================

def build_marathi_rewrite_prompt(
    user_question: str,
    conversation_history: str,
) -> str:
    """
    Converts a follow-up question into a standalone search query.
    """

    return f"""
तुम्ही शोध प्रणालीसाठी प्रश्न पुन्हा लिहिणारे सहाय्यक आहात.

तुमचे काम:

• संभाषणातील मागील संदर्भ समजून घ्या.
• नवीन प्रश्न स्वतंत्र (standalone) स्वरूपात लिहा.
• प्रश्नाचा मूळ अर्थ बदलू नका.
• नवीन माहिती जोडू नका.
• फक्त पुनर्लिखित प्रश्न द्या.
• उत्तरात कोणतेही स्पष्टीकरण लिहू नका.

संभाषण:

{conversation_history}

नवीन प्रश्न:

{user_question}
"""


# ==============================================================================
# Answer Prompt
# ==============================================================================

def build_marathi_answer_prompt(
    user_question: str,
    context: str,
) -> str:
    """
    Builds the final prompt sent to the LLM.
    """

    return f"""
खाली दिलेल्या संदर्भाचा काळजीपूर्वक अभ्यास करा.

==============================
संदर्भ
==============================

{context}

==============================
प्रश्न
==============================

{user_question}

==============================
उत्तर लिहिण्याची पद्धत
==============================

१. विषयाचा थोडक्यात परिचय द्या.

२. प्रश्नाचे सविस्तर उत्तर द्या.

३. आवश्यक असल्यास महत्त्वाचे मुद्दे बुलेट पॉइंट्समध्ये द्या.

४. शेवटी छोटासा निष्कर्ष द्या.

फक्त संदर्भातील माहिती वापरा.
"""


# ==============================================================================
# Convenience Helper
# ==============================================================================

def get_system_prompt() -> str:
    """
    Returns the system prompt.

    This wrapper makes future prompt changes easier.
    """

    return SYSTEM_PROMPT