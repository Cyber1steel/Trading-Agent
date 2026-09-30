import re


_MULTIPLE_HORIZONTAL_SPACE = re.compile(r"[ \t]{2,}")
_EXCESSIVE_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize common extraction whitespace without rewriting content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u00ad", "").replace("\x00", "")

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        leading = line[: len(line) - len(line.lstrip(" \t"))]
        content = line[len(leading) :].rstrip(" \t")
        cleaned_lines.append(leading + _MULTIPLE_HORIZONTAL_SPACE.sub(" ", content))

    cleaned = "\n".join(cleaned_lines)
    cleaned = _EXCESSIVE_BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()
