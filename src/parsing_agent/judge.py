from __future__ import annotations

import base64
import json
from typing import Any
from urllib import request

import fitz
from langsmith import tracing_context

from parsing_agent.config import WorkflowConfig
from parsing_agent.interfaces import CandidateJudge
from parsing_agent.monitoring import load_judge_prompt_hints
from parsing_agent.models import DocumentSource, EvaluationMetrics, JudgeResult, ParseCandidate, load_document_source_text

_SYSTEM_PROMPT = """You are judging the quality of a parsed document against its source.
Return strict JSON with this schema:
{
  "overall_score": number between 0 and 1,
  "coverage_score": number between 0 and 1,
  "structure_score": number between 0 and 1,
  "table_score": number between 0 and 1,
  "hallucination_risk": number between 0 and 1,
  "editorial_readiness": number between 0 and 1,
  "notes": ["short note", "..."],
  "issues": ["specific issue", "..."]
}
`hallucination_risk` is penalty-compatible: 0 means low risk, 1 means high risk.
`issues` may be omitted or an empty list.
`overall_score` should reflect content fidelity, structural preservation, formatting usefulness, and penalize hallucination risk.
Do not include any prose outside the JSON object."""
def _extract_message_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        raise ValueError("LLM judge response did not include choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        joined_text = "\n".join(part for part in text_parts if part)
        if not joined_text:
            raise ValueError("LLM judge response content list included no text blocks.")
        return joined_text
    raise ValueError("LLM judge response content format was not recognized.")


def _post_chat_completion(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with tracing_context(enabled=False):
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _post_response(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with tracing_context(enabled=False):
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _extract_response_text(response_payload: dict[str, Any]) -> str:
    output_items = response_payload.get("output") or []
    text_parts: list[str] = []
    for item in output_items:
        if item.get("type") != "message":
            continue
        for content_item in item.get("content") or []:
            if content_item.get("type") == "output_text":
                text = content_item.get("text")
                if text:
                    text_parts.append(str(text))
    if text_parts:
        return "\n".join(text_parts)
    raise ValueError("LLM judge multimodal response did not include output_text content.")


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _coerce_optional_score(payload: dict[str, Any], field_name: str) -> float | None:
    value = payload.get(field_name)
    if value is None:
        return None
    return _clamp(float(value))


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _is_pdf_source(source: DocumentSource) -> bool:
    return source.media_type == "application/pdf" or source.path.suffix.lower() == ".pdf"


def _render_pdf_page_data_urls(source: DocumentSource, max_pages: int) -> list[tuple[int, str]]:
    if not _is_pdf_source(source):
        return []
    page_limit = max(0, min(source.page_count or max_pages, max_pages))
    if page_limit <= 0:
        return []
    images: list[tuple[int, str]] = []
    with fitz.open(source.path) as document:
        for page_index in range(min(page_limit, document.page_count)):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
            encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
            images.append((page_index + 1, f"data:image/png;base64,{encoded}"))
    return images


class OpenAICompatibleJudge(CandidateJudge):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        enable_multimodal_grounding: bool = True,
        grounding_max_pages: int = 2,
        prompt_hints: list[str] | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._enable_multimodal_grounding = enable_multimodal_grounding
        self._grounding_max_pages = grounding_max_pages
        self._prompt_hints = list(prompt_hints or [])

    def judge(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
    ) -> JudgeResult:
        source_text = load_document_source_text(source)
        prompt = self._build_prompt(source_text, candidate.content, metrics)
        grounding_pages = _render_pdf_page_data_urls(source, self._grounding_max_pages) if self._enable_multimodal_grounding else []
        if grounding_pages:
            response_payload = _post_response(
                url=f"{self._base_url}/responses",
                api_key=self._api_key,
                payload={
                    "model": self._model,
                    "input": [
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": _SYSTEM_PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt}]
                            + [
                                {"type": "input_image", "image_url": image_url, "detail": "high"}
                                for _, image_url in grounding_pages
                            ],
                        },
                    ],
                },
                timeout_seconds=self._timeout_seconds,
            )
            verdict = json.loads(_extract_response_text(response_payload))
        else:
            response_payload = _post_chat_completion(
                url=f"{self._base_url}/chat/completions",
                api_key=self._api_key,
                payload={
                    "model": self._model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout_seconds=self._timeout_seconds,
            )
            verdict = json.loads(_extract_message_content(response_payload))
        overall_score = _coerce_optional_score(verdict, "overall_score")
        if overall_score is None:
            overall_score = _clamp(float(verdict["score"]))
        return JudgeResult(
            overall_score=overall_score,
            coverage_score=_coerce_optional_score(verdict, "coverage_score"),
            structure_score=_coerce_optional_score(verdict, "structure_score"),
            table_score=_coerce_optional_score(verdict, "table_score"),
            hallucination_risk=_coerce_optional_score(verdict, "hallucination_risk"),
            editorial_readiness=_coerce_optional_score(verdict, "editorial_readiness"),
            notes=_coerce_string_list(verdict.get("notes")),
            issues=_coerce_string_list(verdict.get("issues")),
            metadata={
                "transport": "responses" if grounding_pages else "chat_completions",
                "grounding_enabled": bool(grounding_pages),
                "grounding_pages": [page_number for page_number, _ in grounding_pages],
            },
        )

    def _build_prompt(
        self,
        source_text: str,
        candidate_text: str,
        metrics: EvaluationMetrics,
    ) -> str:
        tuning_text = ""
        if self._prompt_hints:
            tuning_text = "\n".join(f"- {hint}" for hint in self._prompt_hints)
            tuning_text = f"\n\nExtra review instructions from prior failures:\n{tuning_text}"
        return (
            "Judge the parser output against the source.\n\n"
            f"Deterministic metrics: coverage={metrics.text_coverage:.3f}, "
            f"similarity={metrics.normalized_similarity:.3f}, "
            f"structure={metrics.structure_retention:.3f}, "
            f"table={metrics.table_preservation:.3f}, "
            f"empty_penalty={metrics.empty_block_penalty:.3f}, "
            f"repeat_penalty={metrics.repetition_penalty:.3f}"
            f"{tuning_text}\n\n"
            f"Source text:\n{source_text}\n\n"
            f"Candidate text:\n{candidate_text}"
        )


def build_default_judge(config: WorkflowConfig) -> CandidateJudge | None:
    if config.judge_weight <= 0:
        return None
    if not config.judge_model or not config.judge_api_key:
        return None
    return OpenAICompatibleJudge(
        model=config.judge_model,
        api_key=config.judge_api_key,
        base_url=config.judge_base_url,
        timeout_seconds=config.judge_timeout_seconds,
        enable_multimodal_grounding=config.judge_multimodal_grounding_enabled,
        grounding_max_pages=config.judge_grounding_max_pages,
        prompt_hints=load_judge_prompt_hints(config),
    )
