import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.knowledge.chunker import chunk_text
from app.knowledge.cleaner import clean_text
from app.knowledge.loaders.base import DocumentLoadError
from app.knowledge.loaders.markdown import MarkdownLoader
from app.knowledge.loaders.pdf import PDFLoader
from app.knowledge.loaders.text import TextLoader
from app.knowledge.models import LoadOptions, SourceMetadata, TextSegment
from app.knowledge.pipeline import (
    DocumentIngestionError,
    process_document,
    write_processed_document,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    page_ids = [4 + index * 2 for index in range(len(page_texts))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, page_text in enumerate(page_texts):
        page_id = 4 + index * 2
        content_id = page_id + 1
        escaped = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii") if page_text else b""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    path.write_bytes(pdf)


def _source() -> SourceMetadata:
    return SourceMetadata(
        source_id="source-123",
        source_type="txt",
        title="Fictional Notes",
        file_path="data/raw/notes.txt",
        source_url="https://example.invalid/notes",
    )


def test_text_loader_preserves_utf8_and_source_metadata(tmp_path: Path) -> None:
    source_path = tmp_path / "market_notes.txt"
    source_path.write_bytes("Café price notes\n".encode("utf-8"))

    loaded = TextLoader().load(
        source_path,
        LoadOptions(author="A. Example", source_url="https://example.invalid/source"),
    )

    assert loaded.raw_text == "Café price notes\n"
    assert loaded.source.title == "Market notes"
    assert loaded.source.author == "A. Example"
    assert loaded.source.source_url == "https://example.invalid/source"
    assert loaded.source.file_path == str(source_path.resolve())
    assert loaded.source.source_type == "txt"


def test_text_loader_replaces_invalid_utf8_bytes(tmp_path: Path) -> None:
    source_path = tmp_path / "invalid.txt"
    source_path.write_bytes(b"valid\xfftext")

    assert "\ufffd" in TextLoader().load(source_path).raw_text


def test_markdown_loader_preserves_markdown_content(tmp_path: Path) -> None:
    markdown = "# Market Notes\n\n- **Price** stayed near 1.25.\n"
    source_path = tmp_path / "market_notes.md"
    source_path.write_text(markdown, encoding="utf-8", newline="")

    loaded = MarkdownLoader().load(source_path)

    assert loaded.raw_text == markdown
    assert loaded.source.source_type == "markdown"
    assert loaded.source.title == "Market notes"


def test_pdf_loader_extracts_each_page_and_keeps_page_numbers(tmp_path: Path) -> None:
    source_path = tmp_path / "lesson.pdf"
    _write_pdf(source_path, ["First page note", "Second page note"])

    loaded = PDFLoader().load(source_path)

    assert "First page note" in loaded.raw_text
    assert "Second page note" in loaded.raw_text
    assert [page.page_number for page in loaded.pages] == [1, 2]
    assert loaded.source.file_path == str(source_path.resolve())
    assert loaded.source.title == "Lesson"


def test_pdf_loader_fails_clearly_for_image_only_pdf(tmp_path: Path) -> None:
    source_path = tmp_path / "scan.pdf"
    _write_pdf(source_path, [""])

    with pytest.raises(DocumentLoadError, match="no extractable text.*OCR"):
        PDFLoader().load(source_path)


def test_cleaner_normalizes_whitespace_and_safe_extraction_artifacts() -> None:
    original = "\u00a0Price   paused. \u00ad\x00\r\n\r\n\r\nVolume stayed quiet.  \r\n"

    cleaned = clean_text(original)

    assert cleaned == "Price paused.\n\nVolume stayed quiet."
    assert clean_text(original) == cleaned
    assert "Price paused" in cleaned and "Volume stayed quiet" in cleaned


def test_chunker_preserves_order_overlap_metadata_and_offsets() -> None:
    text = "alpha bravo charlie delta echo foxtrot"

    chunks = chunk_text(text, _source(), chunk_size=18, overlap=5)

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text for chunk in chunks)
    assert all(text[chunk.char_start : chunk.char_end] == chunk.text for chunk in chunks)
    assert chunks[0].text.startswith("alpha")
    assert chunks[0].source_id == "source-123"
    assert chunks[0].source_url == "https://example.invalid/notes"
    assert chunks[1].char_start < chunks[0].char_end


def test_chunker_preserves_page_attribution() -> None:
    text = "First page words\n\nSecond page words"
    chunks = chunk_text(
        text,
        _source(),
        chunk_size=20,
        overlap=2,
        segments=[
            TextSegment(start=0, end=16, page_number=1),
            TextSegment(start=18, end=len(text), page_number=2),
        ],
    )

    assert [chunk.page_number for chunk in chunks] == [1, 2]


def test_chunker_rejects_invalid_size_and_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("some text", _source(), chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", _source(), chunk_size=10, overlap=10)


def test_pipeline_processes_txt_end_to_end_and_preserves_optional_metadata(tmp_path: Path) -> None:
    source_path = tmp_path / "lesson.txt"
    source_path.write_text("\n  Price   paused.\n\n\nVolume was quiet.  ", encoding="utf-8")

    document = process_document(
        source_path,
        chunk_size=16,
        overlap=3,
        options=LoadOptions(
            author="A. Example",
            source_url="https://example.invalid/lesson",
            published_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        ),
    )

    assert document.source.author == "A. Example"
    assert document.source.source_url == "https://example.invalid/lesson"
    assert document.source.published_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert document.cleaned_characters < document.extracted_characters
    assert len(document.chunks) > 1
    assert all(chunk.source_id == document.source.source_id for chunk in document.chunks)


def test_pipeline_rejects_unsupported_empty_and_missing_sources(tmp_path: Path) -> None:
    unsupported = tmp_path / "lesson.html"
    unsupported.write_text("<p>not supported</p>", encoding="utf-8")
    with pytest.raises(DocumentIngestionError, match="Unsupported file extension"):
        process_document(unsupported)

    empty = tmp_path / "empty.txt"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(DocumentIngestionError, match="contains no text"):
        process_document(empty)

    with pytest.raises(DocumentIngestionError, match="does not exist"):
        process_document(tmp_path / "missing.txt")


def test_pipeline_writes_traceable_json(tmp_path: Path) -> None:
    source_path = tmp_path / "market notes.txt"
    source_path.write_text("A fictional market note.", encoding="utf-8")
    document = process_document(source_path)

    output_path = write_processed_document(document, tmp_path / "processed")
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name.startswith("market-notes-")
    assert data["source"]["source_id"] == document.source.source_id
    assert data["chunks"][0]["source_id"] == document.source.source_id
    assert data["chunks"][0]["file_path"] == str(source_path.resolve())
    assert "embedding" not in data["chunks"][0]


def test_ingestion_script_processes_the_sample_document(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "ingest_document.py"
    sample = PROJECT_ROOT / "data" / "raw" / "sample_trading_notes.txt"
    output_dir = tmp_path / "processed"
    result = subprocess.run(
        [sys.executable, str(script), str(sample), "--output-dir", str(output_dir)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Chunks:" in result.stdout
    output_files = list(output_dir.glob("*.json"))
    assert len(output_files) == 1
    output = json.loads(output_files[0].read_text(encoding="utf-8"))
    assert output["source"]["source_type"] == "txt"
    assert output["chunks"]
