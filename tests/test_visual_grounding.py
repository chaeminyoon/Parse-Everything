"""visual repair 환각 방어 검증: 페이지 재배치, 콘텐츠 그라운딩 게이트.

골든 라벨 1호가 잡아낸 실제 사고 경로를 재현한다: judge가 인쇄 쪽번호를
PDF 페이지 번호로 착각 → 라벨 없는 페이지에서 아무 표나 crop → 환각 표가
라벨 자리에 삽입.
"""

import json
from pathlib import Path

import fitz
import pytest

from parsing_agent.models import DocumentSource, ParseCandidate
from parsing_agent.repair import HeuristicRepairer
from parsing_agent.visual_repair import (
    OpenAIVisualTableRecoverer,
    VisualRepairTask,
    VisualTableRecovery,
    _recovery_grounding_ratio,
)


def _make_pdf(tmp_path, *, label_page: int = 1, pages: int = 2, label: str = "표 9.1-1"):
    path = tmp_path / "grounding-test.pdf"
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 60), f"본문 페이지 {index + 1} 내용이 여기에 충분히 길게 들어간다" * 2)
        if index + 1 == label_page:
            page.insert_text((72, 200), label)
            page.insert_text((72, 230), "구분 수질 대기 소음 항목 값 기준")
    document.save(path)
    document.close()
    return path


def _recoverer() -> OpenAIVisualTableRecoverer:
    return OpenAIVisualTableRecoverer(model="gpt-test", api_key="test-key")


def _fake_response(markdown: str) -> dict:
    body = json.dumps(
        {"table_label": "표 9.1-1", "page_number": 1, "confidence": 0.9, "markdown": markdown},
        ensure_ascii=False,
    )
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": body}]}]}


# --- 페이지 재배치 -------------------------------------------------------------


def test_wrong_page_number_is_relocated_to_label_page(tmp_path, monkeypatch) -> None:
    # 라벨은 1페이지에 있는데 task는 (judge 오답처럼) 2페이지를 들고 온다.
    pdf = _make_pdf(tmp_path, label_page=1, pages=2)
    captured = {}

    def fake_post(*, url, api_key, payload, timeout_seconds):
        captured["prompt"] = payload["input"][0]["content"][0]["text"]
        return _fake_response("| 구분 | 수질 |\n| --- | --- |\n| 대기 | 소음 |")

    monkeypatch.setattr("parsing_agent.visual_repair._post_response", fake_post)

    recovery = _recoverer()._recover_single_table(pdf, "", "표 9.1-1", page_number=2)

    assert recovery is not None
    assert "Source page number: 1" in captured["prompt"]


def test_label_missing_everywhere_aborts_without_api_call(tmp_path, monkeypatch) -> None:
    pdf = _make_pdf(tmp_path, label_page=1, pages=2)

    def exploding_post(**kwargs):
        raise AssertionError("라벨이 없으면 vision API를 호출하면 안 된다")

    monkeypatch.setattr("parsing_agent.visual_repair._post_response", exploding_post)

    recovery = _recoverer()._recover_single_table(pdf, "", "표 99.9-9", page_number=2)

    assert recovery is None


# --- 그라운딩 비율 -------------------------------------------------------------


def test_grounding_ratio_high_when_cells_exist_in_reference() -> None:
    reference = "구 분 수질 대기 소음 항목 값 기준 이 페이지에는 표의 원본 텍스트가 충분히 길게 존재하며 조사지점과 조사주기도 함께 기재되어 있다"
    markdown = "| 구분 | 수질 |\n| --- | --- |\n| 대기 | 소음 |"

    ratio = _recovery_grounding_ratio(reference, markdown)

    assert ratio == 1.0


def test_grounding_ratio_low_for_alien_content() -> None:
    reference = "구 분 수질 대기 소음 항목 값 기준 이 페이지에는 표의 원본 텍스트가 충분히 길게 존재하며 조사지점과 조사주기도 함께 기재되어 있다"
    markdown = "| 정수장 | 시설용량 |\n| --- | --- |\n| 급수지역 | 취수장 |"

    ratio = _recovery_grounding_ratio(reference, markdown)

    assert ratio == 0.0


def test_grounding_none_when_reference_text_missing() -> None:
    # 스캔 페이지: crop 텍스트가 거의 없으면 판정 불가
    assert _recovery_grounding_ratio("", "| a | b |\n| --- | --- |\n| c | d |") is None
    assert _recovery_grounding_ratio("짧음", "| a | b |") is None


def test_grounding_handles_html_recovery() -> None:
    reference = "구 분 수질 대기 소음 항목 값 기준 원본 텍스트가 충분히 길게 존재하며 조사지점과 조사주기도 함께 기재되어 있다"
    html = "<table><tr><td>구분</td><td>수질</td></tr><tr><td>대기</td><td>소음</td></tr></table>"

    assert _recovery_grounding_ratio(reference, html) == 1.0


# --- apply_chunk_repair 게이트 --------------------------------------------------


class _FixedRecoverer:
    def __init__(self, grounding):
        self._grounding = grounding

    def recover_task(self, source, content, task):
        return VisualTableRecovery(
            table_label=task.table_label,
            page_number=task.page_number,
            confidence=0.9,
            markdown="| 구분 | 항목 |\n| --- | --- |\n| 수질 | BOD |",
            notes=[],
            crop_method="test",
            bbox=None,
            grounding=self._grounding,
        )


def _pdf_source(run_id: str) -> DocumentSource:
    return DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id=run_id,
        extracted_text="원문",
        page_count=8,
    )


@pytest.mark.parametrize(
    ("grounding", "expected_reason"),
    [(0.1, "content_mismatch"), (None, None), (0.9, None)],
)
def test_apply_chunk_repair_gates_on_grounding(grounding, expected_reason) -> None:
    repairer = HeuristicRepairer(visual_table_recoverer=_FixedRecoverer(grounding))
    candidate = ParseCandidate(
        parser_name="p",
        content="<!-- page 3 -->\n표 9.1-1 캡션이 있는 본문",
        format_name="md",
    )
    task = VisualRepairTask(task_id="t1", table_label="표 9.1-1", page_number=3, issue_types=("missing_header",))
    rejections: list[dict] = []

    result = repairer.apply_chunk_repair(_pdf_source(f"g-{grounding}"), candidate, task, rejection_sink=rejections)

    if expected_reason == "content_mismatch":
        assert result is None
        assert rejections[0]["reason"] == "content_mismatch"
        assert rejections[0]["grounding"] == 0.1
    else:
        # grounding이 None(판정 불가)이거나 충분히 높으면 패치가 진행된다
        assert result is not None
