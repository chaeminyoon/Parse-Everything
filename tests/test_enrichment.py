from __future__ import annotations

import base64

from parsing_agent.config import WorkflowConfig
from parsing_agent.enrichment import MarkdownImageCaptionEnricher
from parsing_agent.models import DocumentSource, ParseCandidate

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnR6zsAAAAASUVORK5CYII="
)


def test_markdown_image_caption_enricher_replaces_relative_placeholder(tmp_path, monkeypatch) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    image_path = assets_dir / "chart.png"
    image_path.write_bytes(_PNG_BYTES)
    source_path = tmp_path / "parsed.md"
    source_path.write_text("![image chart](assets/chart.png)", encoding="utf-8")

    source = DocumentSource(
        path=source_path,
        media_type="text/markdown",
        size_bytes=source_path.stat().st_size,
        run_id="enrichment-test",
        extracted_text=source_path.read_text(encoding="utf-8"),
    )
    candidate = ParseCandidate(
        parser_name="mock",
        content="Before\n\n![image chart](assets/chart.png)\n\nAfter",
        format_name="md",
        source_path=source_path,
    )

    payloads: list[dict[str, object]] = []

    def fake_post_response(payload, config, timeout_seconds):
        del config, timeout_seconds
        payloads.append(payload)
        return {"output_text": "Industrial site layout overview"}

    monkeypatch.setattr("parsing_agent.enrichment._post_response", fake_post_response)

    enriched = MarkdownImageCaptionEnricher(
        WorkflowConfig(
            judge_api_key="test-key",
            post_selection_image_captioning_enabled=True,
            post_selection_image_caption_model="gpt-test",
        )
    ).enrich(source, candidate)

    assert "![image chart]" not in enriched.content
    assert "Image: Industrial site layout overview" in enriched.content
    assert enriched.metadata["image_caption_enrichment_count"] == 1
    assert enriched.metadata["image_caption_enrichment_paths"] == [str(image_path)]
    assert payloads and payloads[0]["model"] == "gpt-test"


def test_markdown_image_caption_enricher_uses_embedded_image_data_when_file_is_missing(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "parsed.md"
    source_path.write_text("![image chart](assets/chart.png)", encoding="utf-8")
    source = DocumentSource(
        path=source_path,
        media_type="text/markdown",
        size_bytes=source_path.stat().st_size,
        run_id="enrichment-test",
        extracted_text=source_path.read_text(encoding="utf-8"),
    )
    candidate = ParseCandidate(
        parser_name="mock",
        content="![image chart](assets/chart.png)",
        format_name="md",
        metadata={"embedded_image_data_urls": {"assets/chart.png": "data:image/png;base64,AAAA"}},
        source_path=source_path,
    )

    monkeypatch.setattr(
        "parsing_agent.enrichment._post_response",
        lambda payload, config, timeout_seconds: {"output_text": "Embedded chart summary"},
    )

    enriched = MarkdownImageCaptionEnricher(
        WorkflowConfig(
            judge_api_key="test-key",
            post_selection_image_captioning_enabled=True,
            post_selection_image_caption_model="gpt-test",
        )
    ).enrich(source, candidate)

    assert enriched.content == "Image: Embedded chart summary"
    assert enriched.metadata["image_caption_enrichment_paths"] == ["assets/chart.png"]


def test_markdown_image_caption_enricher_leaves_remote_placeholder_unchanged(tmp_path) -> None:
    source_path = tmp_path / "parsed.md"
    source_path.write_text("![image chart](https://example.com/chart.png)", encoding="utf-8")
    source = DocumentSource(
        path=source_path,
        media_type="text/markdown",
        size_bytes=source_path.stat().st_size,
        run_id="enrichment-test",
        extracted_text=source_path.read_text(encoding="utf-8"),
    )
    candidate = ParseCandidate(
        parser_name="mock",
        content="![image chart](https://example.com/chart.png)",
        format_name="md",
        source_path=source_path,
    )

    enriched = MarkdownImageCaptionEnricher(
        WorkflowConfig(
            judge_api_key="test-key",
            post_selection_image_captioning_enabled=True,
            post_selection_image_caption_model="gpt-test",
        )
    ).enrich(source, candidate)

    assert enriched.content == candidate.content
    assert enriched.metadata == candidate.metadata
