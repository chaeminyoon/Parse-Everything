from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.ingestion import build_document_source
from parsing_agent.ocr import OcrResult, extract_text_from_surya_result, should_run_ocr


def test_should_run_ocr_for_pdf_with_too_little_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    config = WorkflowConfig(ocr_enabled=True, ocr_min_text_characters=50)

    assert should_run_ocr(
        media_type="application/pdf",
        path=pdf_path,
        extracted_text="",
        config=config,
    )


def test_should_not_run_ocr_when_disabled(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    config = WorkflowConfig(ocr_enabled=False, ocr_min_text_characters=50)

    assert not should_run_ocr(
        media_type="application/pdf",
        path=pdf_path,
        extracted_text="",
        config=config,
    )


def test_should_run_ocr_for_image_inputs(tmp_path: Path) -> None:
    config = WorkflowConfig(ocr_enabled=True, ocr_min_text_characters=50)

    for file_name, media_type in [
        ("scan.png", "image/png"),
        ("photo.jpg", "image/jpeg"),
        ("page.jpeg", "image/jpeg"),
        ("fax.tiff", "image/tiff"),
    ]:
        image_path = tmp_path / file_name
        image_path.write_bytes(b"\x89PNG-fake")

        assert should_run_ocr(
            media_type=media_type,
            path=image_path,
            extracted_text=None,
            config=config,
        ), file_name


def test_should_not_run_ocr_for_image_when_disabled(tmp_path: Path) -> None:
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"\x89PNG-fake")

    config = WorkflowConfig(ocr_enabled=False)

    assert not should_run_ocr(
        media_type="image/png",
        path=image_path,
        extracted_text=None,
        config=config,
    )


def test_build_document_source_applies_ocr_for_image(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"\x89PNG-fake")
    ocr_dir = tmp_path / "ocr"

    monkeypatch.setattr(
        "parsing_agent.ingestion.run_ocr",
        lambda **kwargs: OcrResult(
            applied=True,
            provider="surya",
            text="이미지에서 추출된 본문",
            metadata={"applied": True, "provider": "surya", "ocr_block_count": 4},
            artifacts={"ocr_text": ocr_dir / "ocr_text.md"},
        ),
    )

    source = build_document_source(
        image_path,
        run_id="image-ocr-test",
        config=WorkflowConfig(ocr_enabled=True),
        artifact_dir=ocr_dir,
    )

    assert source.extracted_text == "이미지에서 추출된 본문"
    assert source.ocr_metadata["applied"] is True
    assert source.page_count is None


def test_extract_text_from_surya_result_handles_html_blocks() -> None:
    result = {
        "pages": [
            {
                "text_lines": [
                    {"html": "<h1>제1장 요약문</h1>"},
                    {"html": "<table><tr><th>항목</th><th>값</th></tr><tr><td>대기질</td><td>양호</td></tr></table>"},
                ]
            }
        ]
    }

    text = extract_text_from_surya_result(result)

    assert "<!-- page 1 -->" in text
    assert "제1장 요약문" in text
    assert "대기질" in text
    assert "| 항목 | 값 |" in text
    assert "| --- | --- |" in text
    assert "| 대기질 | 양호 |" in text


def test_build_document_source_applies_ocr_when_pdf_text_is_empty(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    ocr_dir = tmp_path / "ocr"

    monkeypatch.setattr(
        "parsing_agent.ingestion.extract_source_text",
        lambda path, media_type: ("", 3),
    )
    monkeypatch.setattr(
        "parsing_agent.ingestion.run_ocr",
        lambda **kwargs: OcrResult(
            applied=True,
            provider="surya",
            text="OCR body",
            metadata={
                "applied": True,
                "provider": "surya",
                "ocr_page_count": 3,
                "ocr_block_count": 12,
            },
            artifacts={"ocr_text": ocr_dir / "ocr_text.md"},
        ),
    )

    source = build_document_source(
        pdf_path,
        run_id="ocr-test",
        config=WorkflowConfig(ocr_enabled=True, ocr_min_text_characters=50),
        artifact_dir=ocr_dir,
    )

    assert source.extracted_text == "OCR body"
    assert source.ocr_metadata["applied"] is True
    assert source.ocr_metadata["provider"] == "surya"
    assert source.ocr_metadata["ocr_block_count"] == 12
    assert source.ocr_artifacts["ocr_text"].endswith("ocr_text.md")
