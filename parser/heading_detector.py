# heading_detector.py
import re

MAIN_HEADINGS = [
    "abstract",
    "keywords",
    "introduction",
    "background",
    "literature review",
    "related work",
    "theoretical framework",
    "problem statement",
    "research questions",
    "objectives",
    "hypothesis",
    "methodology",
    "methods",
    "method",
    "materials and methods",
    "experimental setup",
    "data collection",
    "data analysis",
    "results",
    "findings",
    "discussion",
    "interpretation",
    "implications",
    "limitations",
    "future work",
    "conclusion",
    "summary",
    "recommendations",
    "acknowledgments",
    "funding",
    "conflict of interest",
    "ethical considerations",
    "appendix",
    "supplementary materials",
    "references",
    "bibliography",
    "glossary",
    "abbreviations",
    "notation",
    "author contributions",
    "data availability",
    "code availability",
    "figures",
    "tables",
    "index",
]

HEADING_REGEX = re.compile(
    r"^(?:\s*\d+(?:\.\d+)*\s*)?"          # optional numbering (e.g., 1.2.3)
    r"(" + "|".join(re.escape(h) for h in MAIN_HEADINGS) + r")" # heading keywords
    r"\s*[:\-\.]?\s*$",                   # optional trailing punctuation
    flags=re.IGNORECASE
)

EXCEPTION_SHORT_SECTIONS = {"keywords", "acknowledgments", "funding"}

def looks_like_numbered_heading(line):
    return re.match(r"^\s*\d+(\.\d+)*\.?\s*", line) is not None

def detect_headings_from_lines(text_lines):
    """
    Detect headings from extracted PDF lines with font info.

    Args:
        text_lines: list of tuples (page_num, y0, line_text, is_bold)

    Returns:
        List of (start_offset, end_offset, heading_name)
    """
    headings = []
    offset = 0

    for _, _, line_text, is_bold_line in text_lines:
        stripped_line = line_text.strip()
        line_len = len(stripped_line) + 1  # +1 for newline in full text concat

        if not stripped_line:
            offset += line_len
            continue

        lowered = stripped_line.lower()
        # Only consider if bold or numbered heading (your heuristic)
        if not (is_bold_line or looks_like_numbered_heading(stripped_line)):
            offset += line_len
            continue

        # Remove numbering from start, trailing punctuation
        text_no_num = re.sub(r"^\s*\d+(\.\d+)*\.?\s*", "", lowered).strip()
        text_no_num = re.sub(r"[\:\.\-]+$", "", text_no_num).strip()

        if len(text_no_num.split()) > 10:
            offset += line_len
            continue

        # Match using old regex for known headings
        if HEADING_REGEX.match(text_no_num):
            # Extract heading keyword only
            match = HEADING_REGEX.match(text_no_num)
            heading_word = match.group(1).lower()
            headings.append((offset, offset + line_len, heading_word))

        offset += line_len

    headings.sort(key=lambda x: x[0])
    return headings

def chunk_text_by_headings(text, headings):
    """
    Chunk text by detected headings.

    Args:
        text (str): full extracted text.
        headings (list): list of (start, end, heading_name)

    Returns:
        dict of {heading_name: section_text}
    """
    if not headings:
        return {"full_text": text.strip()}

    sections = {}
    for i, (start, end, heading) in enumerate(headings):
        section_start = end
        section_end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        section_text = text[section_start:section_end].strip()

        key = heading
        if key in sections:
            count = 2
            while f"{key}_{count}" in sections:
                count += 1
            key = f"{key}_{count}"

        if len(section_text.split()) < 5 and heading not in EXCEPTION_SHORT_SECTIONS:
            continue

        sections[key] = section_text

    return sections
