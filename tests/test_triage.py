from pathlib import Path

import fitz

from parsing_agent.config import WorkflowConfig
from parsing_agent.models import DocumentSource
from parsing_agent.triage import triage_document


class _SimplePdfPage:
    rect = fitz.Rect(0, 0, 600, 800)

    def find_tables(self):
        return type("Finder", (), {"tables": []})()

    def get_text(self, mode: str):
        assert mode == "blocks"
        return [
            (70, 80, 520, 120, "Simple heading\n", 0, 0),
            (70, 140, 520, 300, "Simple body text\n", 1, 0),
        ]


class _ComplexPdfPage:
    rect = fitz.Rect(0, 0, 600, 800)

    def find_tables(self):
        table = type("Table", (), {"bbox": (70, 220, 520, 420)})()
        return type("Finder", (), {"tables": [table]})()

    def get_text(self, mode: str):
        assert mode == "blocks"
        return [
            (70, 80, 260, 120, "Left column\n", 0, 0),
            (320, 80, 520, 120, "Right column\n", 1, 0),
            (70, 480, 520, 620, "<image>", 2, 1),
        ]


class _MultiColumnPdfPage:
    rect = fitz.Rect(0, 0, 600, 800)

    def find_tables(self):
        return type("Finder", (), {"tables": []})()

    def get_text(self, mode: str):
        assert mode == "blocks"
        return [
            (70, 80, 260, 120, "Left column\n", 0, 0),
            (320, 80, 520, 120, "Right column\n", 1, 0),
            (70, 160, 260, 220, "Left body\n", 2, 0),
            (320, 160, 520, 220, "Right body\n", 3, 0),
        ]


class _FakeDocument:
    def __init__(self, pages):
        self._pages = pages
        self.page_count = len(pages)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def load_page(self, index: int):
        return self._pages[index]


def _pdf_source(page_count: int) -> DocumentSource:
    return DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="triage-test",
        extracted_text="sample extracted text",
        page_count=page_count,
    )


def test_triage_routes_simple_pdf_to_source_text(monkeypatch) -> None:
    monkeypatch.setattr("parsing_agent.triage.fitz.open", lambda path: _FakeDocument([_SimplePdfPage()]))

    decision = triage_document(_pdf_source(page_count=1), WorkflowConfig())

    assert decision.complexity == "simple"
    assert decision.selected_parsers == ["source-text"]
    assert "sample_pages_plain_text_only" in decision.reasons


def test_triage_routes_complex_pdf_to_opendataloader_with_layout_support(monkeypatch) -> None:
    monkeypatch.setattr(
        "parsing_agent.triage.fitz.open",
        lambda path: _FakeDocument([_ComplexPdfPage(), _ComplexPdfPage()]),
    )

    decision = triage_document(_pdf_source(page_count=20), WorkflowConfig())

    assert decision.complexity == "complex"
    assert decision.selected_parsers == ["opendataloader-pdf", "layout-first-pdf"]
    assert "sample_pages_show_tables" in decision.reasons
    assert "sample_pages_show_images" in decision.reasons


def test_triage_uses_configured_pdf_parser_roles_for_complex_pdf(monkeypatch) -> None:
    monkeypatch.setattr(
        "parsing_agent.triage.fitz.open",
        lambda path: _FakeDocument([_ComplexPdfPage(), _ComplexPdfPage()]),
    )
    config = WorkflowConfig(
        parser_names=["custom-primary-pdf", "custom-support-pdf", "mock"],
        pdf_parser_roles={
            "custom-primary-pdf": "primary",
            "custom-support-pdf": "support",
            "mock": "auxiliary",
        },
    )

    decision = triage_document(_pdf_source(page_count=20), config)

    assert decision.complexity == "complex"
    assert decision.selected_parsers == ["custom-primary-pdf", "custom-support-pdf"]


def test_triage_skips_layout_support_when_sample_shows_only_multi_column_text(monkeypatch) -> None:
    monkeypatch.setattr(
        "parsing_agent.triage.fitz.open",
        lambda path: _FakeDocument([_MultiColumnPdfPage(), _MultiColumnPdfPage()]),
    )

    decision = triage_document(_pdf_source(page_count=20), WorkflowConfig())

    assert decision.complexity == "complex"
    assert decision.selected_parsers == ["opendataloader-pdf"]
    assert "sample_pages_show_multi_column_layout" in decision.reasons
