"""Text chunking utilities for long text synthesis."""

import re
from typing import List

# Common abbreviations that shouldn't trigger sentence splits
ABBREVIATIONS = {
    # Titles
    'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Jr.', 'Sr.', 'Rev.',
    # Academic
    'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'B.S.', 'M.S.',
    # Common
    'vs.', 'etc.', 'e.g.', 'i.e.', 'a.m.', 'p.m.', 'approx.',
    # Geographic
    'U.S.', 'U.K.', 'Mt.', 'St.', 'Ave.', 'Blvd.', 'Rd.',
    # Ordinals
    'No.', 'Fig.', 'Vol.', 'Ch.', 'Pt.',
}


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, respecting abbreviations.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    if not text:
        return []

    # Split on sentence-ending punctuation followed by space and capital
    pattern = r'(?<=[.!?])\s+(?=[A-Z"])'
    raw_parts = re.split(pattern, text)

    # Rejoin incorrectly split abbreviations
    sentences = []
    buffer = ""

    for part in raw_parts:
        if buffer:
            buffer += " " + part
        else:
            buffer = part

        # Check if buffer ends with an abbreviation
        buffer_stripped = buffer.rstrip()
        ends_with_abbrev = any(
            buffer_stripped.endswith(abbr) for abbr in ABBREVIATIONS
        )

        if not ends_with_abbrev:
            sentences.append(buffer.strip())
            buffer = ""

    if buffer:
        sentences.append(buffer.strip())

    return sentences


def chunk_text(text: str, max_chars: int = 250) -> List[str]:
    """
    Split text into chunks respecting sentence boundaries.

    Useful for processing long text that might degrade TTS quality
    or exceed model context limits.

    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk (default 250)

    Returns:
        List of text chunks
    """
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Handle sentences longer than max_chars
        if not current_chunk and len(sentence) > max_chars:
            # Split long sentence on commas or include as-is
            chunks.append(sentence)
            continue

        # Try adding sentence to current chunk
        test_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence

        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def estimate_chunk_count(text: str, max_chars: int = 250) -> int:
    """
    Estimate how many chunks text will produce.

    Args:
        text: Input text
        max_chars: Maximum characters per chunk

    Returns:
        Estimated number of chunks
    """
    return len(chunk_text(text, max_chars))
