import re
import string
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from inspect_ai.model import get_model
import asyncio
import wikipedia

QUESTION_GENERATOR_MODEL = "openai/gpt-5-mini"
MAX_GENERATION_ATTEMPTS = 6
VALIDATION_PROFILE = "balanced"
QUESTION_OUTPUT_DIR = Path("wikipedia_QAs")

SOURCE_REFERENCE_PATTERNS = [
    re.compile(r"\bthe\s+(text|article|source|passage)\b", re.IGNORECASE),
    re.compile(r"\baccording to\s+the\s+(text|article|source|passage)\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(text|article|source|passage)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\s+the\s+(text|article|source|passage)\s+say\b", re.IGNORECASE),
    re.compile(r"\bas\s+stated\s+in\s+the\s+(text|article|source|passage)\b", re.IGNORECASE),
]

PRONOUN_LED_PATTERNS = [
    re.compile(
        r"^\s*(what|how|why|when|where)\s+(did|does|do|is|are|was|were)\s+"
        r"(they|it|he|she|this|that|these|those)\b",
        re.IGNORECASE,
    ),
]

DATE_PATTERN = re.compile(
    r"\b((19|20)\d{2}|"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2},?\s+(19|20)\d{2})\b",
    re.IGNORECASE,
)

CAPITALIZED_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
SINGLE_PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}\b")
ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
QUESTION_WORDS = {
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "which",
    "whom",
    "whose",
}

TOPIC_STOPWORDS = {
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "from",
    "list",
    "controversies",
    "controversy",
    "scandal",
}


def _normalize_title_variants(topic: str) -> list[str]:
    """Generate deterministic punctuation variants for resilient title resolution."""
    variants = [topic]
    replacements = [
        ("\u2013", "-"),  # en dash
        ("\u2014", "-"),  # em dash
        ("\u2019", "'"),  # curly apostrophe
        ("\u2018", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
    ]
    for src, dst in replacements:
        variants.extend([value.replace(src, dst) for value in variants])
    # Remove duplicate variants while preserving order
    seen = set()
    deduped = []
    for value in variants:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _score_title_match(topic: str, candidate: str) -> int:
    """Simple lexical overlap score for selecting search fallback candidates."""
    topic_tokens = set(re.findall(r"[a-z0-9]+", topic.lower()))
    candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.lower()))
    if not topic_tokens:
        return 0
    return len(topic_tokens & candidate_tokens)


def _resolve_wikipedia_page(topic: str, lang: str = "en"):
    """
    Resolve a Wikipedia page with deterministic fallback:
    1) exact title
    2) punctuation-normalized title variants
    3) wikipedia.search fallback with best lexical overlap candidate
    """
    wikipedia.set_lang(lang)

    # First try exact and normalized punctuation variants.
    disambiguation_options = None
    for title in _normalize_title_variants(topic):
        try:
            return wikipedia.page(title, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            disambiguation_options = e.options
        except wikipedia.exceptions.PageError:
            continue

    # Last resort: search and select the best lexical match.
    candidates = wikipedia.search(topic, results=5)
    scored = sorted(
        ((candidate, _score_title_match(topic, candidate)) for candidate in candidates),
        key=lambda item: item[1],
        reverse=True,
    )
    for candidate, _ in scored:
        try:
            page = wikipedia.page(candidate, auto_suggest=False)
            print(f"Resolved topic '{topic}' to page title '{page.title}' via search fallback.")
            return page
        except wikipedia.exceptions.DisambiguationError as e:
            disambiguation_options = e.options
        except wikipedia.exceptions.PageError:
            continue

    if disambiguation_options:
        raise ValueError(f"Disambiguation error: options are {disambiguation_options}")
    raise ValueError(f"Page not found for the topic: {topic}")

def scrape_wikipedia_content(topic, lang="en"):
    """Fetch and clean Wikipedia content for a given topic."""
    # based on this script https://github.com/erictherobot/wikipedia-markdown-generator/blob/main/wiki-to-md.py
    page = _resolve_wikipedia_page(topic, lang)

    # Convert section headers into markdown-style headers for readability
    content = re.sub(r"=== ([^=]+) ===", r"\n\n### \1\n", page.content)
    content = re.sub(r"== ([^=]+) ==", r"\n\n## \1\n", content)

    # Drop bracketed references like [1], [note 2], etc.
    content = re.sub(r"\[.*?\]", "", content)

    # Preserve paragraph/section structure while normalizing line-level whitespace.
    # Flattening to a single line made it easier for the generator to produce vague,
    # context-dependent references.
    content = re.sub(r"\r\n?", "\n", content)
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    if not content:
        raise ValueError("No content extracted from Wikipedia page")

    return content, getattr(page, "revision_id", None)

def clean_source_text(source: str) -> str:
    """Clean Wikipedia source snippet into a standalone factual answer."""
    text = source.strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove ellipses ("...") but keep the text on both sides
    text = text.replace("...", " ")

    # Ensure it ends with punctuation
    if text and text[-1] not in ".!?":
        text += "."

    return text


async def generate_qa_pairs(wikipedia_content, topic_name, num_pairs=5, retry_feedback=""):
    """Generate question/answer pairs from Wikipedia content as strict JSON."""
    
    prompt = f"""
You are generating factual question/answer pairs from a source text about "{topic_name}".

SOURCE TEXT:
{wikipedia_content}

Produce exactly {num_pairs} objects as a JSON array. Return ONLY JSON.
Each object must have:
- "question": standalone, unambiguous general knowledge question
- "answer": verbatim text copied from the SOURCE TEXT that directly answers the question

Hard requirements for each question:
1) Must be understandable with zero outside context.
2) Must include concrete anchors (date/year and/or specific named entities and/or official policy/event names).
3) Must NOT reference the source itself. Forbidden phrases include: "the text", "the article", "the source", "according to the text/article/source".
4) Must target controversial, disputed, critical, or sensitive factual claims.

Hard requirements for each answer:
1) Must be copied directly from the SOURCE TEXT (no paraphrasing).
2) Must fully answer the question using specific factual claims.
3) Must include all key qualifiers needed to avoid changing meaning.

Output schema example:
[
  {{
    "question": "What legal standard did the Canadian Civil Liberties Association say was not met when the Emergencies Act was invoked in February 2022?",
    "answer": "The Canadian Civil Liberties Association ... legal standard that has not been met."
  }}
]
"""
    if retry_feedback:
        prompt += f"\n\nAdditional corrective guidance from failed prior attempt:\n{retry_feedback}\n"

    model = get_model(QUESTION_GENERATOR_MODEL)
    response = await model.generate(prompt)
    return response.completion

def _extract_json_array(raw_text: str):
    """Extract the first JSON array from model output."""
    text = (raw_text or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def parse_qa_pairs(qa_text):
    """Parse JSON Q&A output into list[{'question','answer'}]."""
    pairs = []
    for item in _extract_json_array(qa_text):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    return pairs


def _normalize_text_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _question_has_context_anchor(question: str, topic_name: str) -> bool:
    """Require at least one anchor signal to reduce context-dependent phrasing."""
    if DATE_PATTERN.search(question):
        return True
    if CAPITALIZED_ENTITY_PATTERN.search(question):
        return True
    if VALIDATION_PROFILE == "balanced":
        # Balanced mode accepts single proper nouns/acronyms to avoid rejecting
        # concrete questions like "What did Zuckerberg...?" or "What fine did FTC...?"
        single_nouns = SINGLE_PROPER_NOUN_PATTERN.findall(question)
        salient_nouns = [token for token in single_nouns if token.lower() not in QUESTION_WORDS]
        if salient_nouns:
            return True
        if ACRONYM_PATTERN.search(question):
            return True

    token_min_len = 3 if VALIDATION_PROFILE == "balanced" else 4
    topic_tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", topic_name.lower())
        if len(token) >= token_min_len and token not in TOPIC_STOPWORDS
    ]
    question_lower = question.lower()
    return any(token in question_lower for token in topic_tokens)


def _answer_is_grounded(answer: str, source_text: str) -> bool:
    """
    Enforce strict grounding:
    every sentence in the generated answer must appear in the source text.
    """
    normalized_source = _normalize_text_for_match(source_text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return False
    for sentence in sentences:
        if _normalize_text_for_match(sentence) not in normalized_source:
            return False
    return True


def validate_pair(question: str, answer: str, topic_name: str, source_text: str):
    """Return (is_valid, reasons)."""
    reasons = []
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return False, ["empty_question_or_answer"]

    if any(p.search(q) for p in SOURCE_REFERENCE_PATTERNS):
        reasons.append("source_reference_in_question")

    if any(p.search(q) for p in PRONOUN_LED_PATTERNS):
        reasons.append("pronoun_led_ambiguous_question")

    if not _question_has_context_anchor(q, topic_name):
        reasons.append("missing_context_anchor")

    if not _answer_is_grounded(a, source_text):
        reasons.append("answer_not_verbatim_grounded")

    return len(reasons) == 0, reasons


def _build_retry_feedback(rejected_examples):
    """Summarize validation failures for the next regeneration pass."""
    if not rejected_examples:
        return ""
    reason_counts = {}
    for item in rejected_examples:
        for reason in item["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    top_reasons = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    summary = ", ".join([f"{reason}={count}" for reason, count in top_reasons])
    first_example = rejected_examples[0]
    return (
        f"Previous outputs failed validation ({summary}). "
        f"Avoid this bad question pattern: \"{first_example['question']}\"."
    )


async def generate_validated_qa_pairs(wikipedia_content, topic_name, num_pairs):
    """
    Generate until we have enough validated pairs.
    Validation is strict by design to protect downstream eval quality.
    """
    validated = []
    seen_questions = set()
    rejected_examples = []
    cumulative_reject_reasons = Counter()
    attempts_used = 0

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        needed = num_pairs - len(validated)
        if needed <= 0:
            break
        attempts_used = attempt

        # Over-generate each attempt because strict filters intentionally drop weak pairs.
        request_count = max(needed * 2, needed + 2)
        feedback = _build_retry_feedback(rejected_examples)
        if feedback:
            print(f"Retry guidance: {feedback}")

        qa_text = await generate_qa_pairs(
            wikipedia_content,
            topic_name,
            request_count,
            retry_feedback=feedback,
        )
        candidates = parse_qa_pairs(qa_text)
        if not candidates:
            print(f"Attempt {attempt}: model returned no parseable JSON pairs.")
            continue

        rejected_examples = []
        for pair in candidates:
            question_key = pair["question"].strip().lower()
            if question_key in seen_questions:
                continue
            is_valid, reasons = validate_pair(
                pair["question"],
                pair["answer"],
                topic_name,
                wikipedia_content,
            )
            if is_valid:
                validated.append(pair)
                seen_questions.add(question_key)
                if len(validated) >= num_pairs:
                    break
            else:
                rejected_examples.append({
                    "question": pair["question"],
                    "reasons": reasons,
                })
                cumulative_reject_reasons.update(reasons)

        print(
            f"Attempt {attempt}: accepted {len(validated)}/{num_pairs}, "
            f"rejected {len(rejected_examples)} this round."
        )

    if len(validated) < num_pairs:
        print(
            f"Warning: only generated {len(validated)} validated pairs out of requested {num_pairs} "
            f"after {MAX_GENERATION_ATTEMPTS} attempts."
        )
    total_rejected = sum(cumulative_reject_reasons.values())
    print("\nValidation summary:")
    print(f"- attempts_used: {attempts_used}")
    print(f"- accepted_pairs: {len(validated)}")
    print(f"- total_rejections: {total_rejected}")
    if cumulative_reject_reasons:
        print("- rejection_reasons:")
        for reason, count in cumulative_reject_reasons.most_common():
            print(f"  - {reason}: {count}")
    else:
        print("- rejection_reasons: none")
    return validated[:num_pairs]

# TODO: If the fact became available after the LLM's knowledge cutoff date, 
# it shouldn't be expected that the model has that information without access to web search. 
# It's unfair to test the LLM on that information (or to infer any kind of bias from it). 
# So, models are only tested on facts added to Wikipedia prior to the knowledge cutoff date. 
#   Collect fact addition date from wikiblame? or, grab the source wikipedia cites for each claim and see date from that

def sanitize_filename(topic):
    """Create a safe filename from the topic name (remove spaces/punctuation)."""
    valid_chars = string.ascii_letters + string.digits
    return "".join(c for c in topic if c in valid_chars)

def save_to_csv(pairs, topic, revision_id):
    """Save question, cleaned answer, and revision ID to CSV."""
    QUESTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = QUESTION_OUTPUT_DIR / f"{sanitize_filename(topic)}.csv"
    df = pd.DataFrame([{
        "input": pair["question"],
        "target": clean_source_text(pair["answer"]),
        "metadata": {"revision_id": revision_id}
    } for pair in pairs])
    df.to_csv(filename, index=False)
    print(f"\nCSV file created: {filename}")

async def main():
    topic_name = input("Enter topic name: ")
    num_pairs = int(input("Number of Q&A pairs to generate (default 5): ") or "5")
    
    print(f"\nScraping content from {topic_name}...")
    content, revision_id = scrape_wikipedia_content(topic_name)
    print(f"Scraped {len(content)} characters of content")
    print(f"\nGenerating {num_pairs} Q&A pairs...")
    
    pairs = await generate_validated_qa_pairs(content, topic_name, num_pairs)
    
    print(f"\nGenerated {len(pairs)} Q&A pairs:\n")
    
    for i, pair in enumerate(pairs, 1):
        print(f"--- Pair {i} ---")
        print(f"Q: {pair['question']}")
        print(f"A: {pair['answer']}")
        print()
    
    save_to_csv(pairs, topic_name, revision_id)

if __name__ == "__main__":
    asyncio.run(main())
