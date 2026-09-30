from collections.abc import Sequence

from app.knowledge.models import DocumentChunk, SourceMetadata, TextSegment


def chunk_text(
    text: str,
    source: SourceMetadata,
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
    segments: Sequence[TextSegment] | None = None,
) -> list[DocumentChunk]:
    """Split text into ordered character windows with traceable source ranges."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if not text.strip():
        return []

    segment_list = list(segments) if segments is not None else [TextSegment(start=0, end=len(text))]
    chunks: list[DocumentChunk] = []

    for segment in segment_list:
        if segment.end > len(text) or segment.start > segment.end:
            raise ValueError("Text segment is outside the cleaned text")
        segment_start = segment.start
        segment_end = segment.end

        while segment_start < segment_end:
            start = segment_start
            while start < segment_end and text[start].isspace():
                start += 1
            if start >= segment_end:
                break

            end = min(start + chunk_size, segment_end)
            if end < segment_end:
                boundary_floor = start + max(1, int(chunk_size * 0.7))
                preferred_boundary = max(text.rfind(" ", boundary_floor, end), text.rfind("\n", boundary_floor, end))
                if preferred_boundary > start:
                    end = preferred_boundary
            while end > start and text[end - 1].isspace():
                end -= 1
            if end <= start:
                end = min(start + chunk_size, segment_end)

            chunk_index = len(chunks)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source.source_id}:{chunk_index:06d}",
                    source_id=source.source_id,
                    source_type=source.source_type,
                    title=source.title,
                    file_path=source.file_path,
                    source_url=source.source_url,
                    page_number=segment.page_number,
                    chunk_index=chunk_index,
                    char_start=start,
                    char_end=end,
                    text=text[start:end],
                )
            )
            if end >= segment_end:
                break

            next_start = max(start + 1, end - overlap)
            while next_start < end and text[next_start].isspace():
                next_start += 1
            segment_start = next_start

    return chunks
