from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
import re
import urllib.error
import urllib.request

from langsmith import tracing_context

from parsing_agent.config import WorkflowConfig
from parsing_agent.models import DocumentSource, ParseCandidate

_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target><[^>]+>|[^)]+)\)")


def _extract_response_text(payload: dict[str, object]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()
    fragments: list[str] = []
    for item in payload.get("output", []) if isinstance(payload.get("output"), list) else []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                fragments.append(content["text"])
    return "\n".join(fragment for fragment in fragments if fragment).strip()


def _post_response(payload: dict[str, object], config: WorkflowConfig, timeout_seconds: float) -> dict[str, object]:
    request = urllib.request.Request(
        f"{config.judge_base_url.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.judge_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with tracing_context(enabled=False):
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _clean_target(target: str) -> str:
    cleaned = target.strip()
    if cleaned.startswith("<") and cleaned.endswith(">"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _resolve_image_path(target: str, *, candidate: ParseCandidate, source: DocumentSource) -> Path | None:
    cleaned = _clean_target(target)
    if not cleaned or "://" in cleaned or cleaned.startswith("data:"):
        return None
    image_path = Path(cleaned)
    if image_path.is_absolute():
        return image_path
    bases: list[Path] = []
    if candidate.source_path is not None:
        bases.append(candidate.source_path.parent)
    bases.append(source.path.parent)
    for base in bases:
        resolved = (base / image_path).resolve()
        if resolved.exists():
            return resolved
    return (bases[0] / image_path).resolve() if bases else None


def _resolve_embedded_image_data_url(target: str, candidate: ParseCandidate) -> str | None:
    embedded_images = candidate.metadata.get("embedded_image_data_urls")
    if not isinstance(embedded_images, dict):
        return None
    return embedded_images.get(_clean_target(target))


def _image_path_to_data_url(image_path: Path) -> str | None:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        return None
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class MarkdownImageCaptionEnricher:
    def __init__(self, config: WorkflowConfig) -> None:
        self._config = config

    def enrich(self, source: DocumentSource, candidate: ParseCandidate) -> ParseCandidate:
        if not self._is_enabled(candidate):
            return candidate

        replacements = 0
        resolved_images: list[str] = []

        def replace_match(match: re.Match[str]) -> str:
            nonlocal replacements
            if replacements >= max(self._config.post_selection_image_caption_max_images, 0):
                return match.group(0)
            embedded_data_url = _resolve_embedded_image_data_url(match.group("target"), candidate)
            image_path = _resolve_image_path(match.group("target"), candidate=candidate, source=source)
            if embedded_data_url is None and (image_path is None or not image_path.exists()):
                return match.group(0)
            caption = self._caption_image(
                match.group("alt"),
                image_path=image_path,
                embedded_data_url=embedded_data_url,
            )
            if caption is None:
                return match.group(0)
            replacements += 1
            resolved_images.append(str(image_path) if image_path is not None and image_path.exists() else _clean_target(match.group("target")))
            return f"Image: {caption}"

        enriched_content = _MARKDOWN_IMAGE_RE.sub(replace_match, candidate.content)
        if replacements == 0 or enriched_content == candidate.content:
            return candidate
        return ParseCandidate(
            parser_name=candidate.parser_name,
            content=enriched_content,
            format_name=candidate.format_name,
            metadata={
                **candidate.metadata,
                "image_caption_enrichment_count": replacements,
                "image_caption_enrichment_paths": resolved_images,
            },
            source_path=candidate.source_path,
            repaired_from=candidate.repaired_from,
        )

    def _is_enabled(self, candidate: ParseCandidate) -> bool:
        if not self._config.post_selection_image_captioning_enabled:
            return False
        if not self._config.judge_api_key or not self._config.post_selection_image_caption_model:
            return False
        return candidate.format_name.lower() == "md"

    def _caption_image(
        self,
        alt_text: str,
        *,
        image_path: Path | None = None,
        embedded_data_url: str | None = None,
    ) -> str | None:
        data_url = embedded_data_url or (None if image_path is None else _image_path_to_data_url(image_path))
        if data_url is None:
            return None
        prompt = (
            "Write a single-line caption for the image as it should appear in a parsed markdown document. "
            "Stay factual, concise, and avoid speculation. "
            f"If the existing alt text is useful, consider it: {alt_text or '(none)'}."
        )
        payload = {
            "model": self._config.post_selection_image_caption_model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ],
                }
            ],
            "max_output_tokens": 120,
        }
        try:
            response = _post_response(payload, self._config, self._config.post_selection_image_caption_timeout_seconds)
            caption = _extract_response_text(response)
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
            return None
        caption = " ".join(caption.split())
        return caption or None


def build_default_image_caption_enricher(config: WorkflowConfig) -> MarkdownImageCaptionEnricher:
    return MarkdownImageCaptionEnricher(config)
