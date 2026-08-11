"""Small deterministic semantic chunker used before a MinerU adapter is configured."""

from __future__ import annotations

import re


def chunk_markdown(content: str, max_chars: int = 650) -> list[tuple[list[str], str]]:
    """Preserve Markdown heading context while making independently useful chunks."""

    heading_path: list[str] = []
    blocks: list[tuple[list[str], str]] = []
    current: list[str] = []
    for line in content.splitlines():
        if line.startswith("#"):
            if current:
                blocks.append((heading_path.copy(), "\n".join(current).strip()))
                current = []
            level = len(line) - len(line.lstrip("#"))
            title = line[level:].strip()
            heading_path = heading_path[: level - 1] + [title]
        elif line.strip():
            current.append(line.strip())
    if current:
        blocks.append((heading_path.copy(), "\n".join(current).strip()))

    chunks: list[tuple[list[str], str]] = []
    for path, block in blocks:
        sentences = re.split(r"(?<=[。！？.!?])\s*", block)
        buffer = ""
        for sentence in sentences:
            if not sentence:
                continue
            if buffer and len(buffer) + len(sentence) + 1 > max_chars:
                chunks.append((path, buffer))
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            chunks.append((path, buffer))
    return chunks
