"""visual repair 패치 폴백(라벨/페이지 앵커 삽입)과 거부 사유 로깅 검증."""

from pathlib import Path
from types import SimpleNamespace

from parsing_agent.config import WorkflowConfig
from parsing_agent.models import DocumentSource, ParseCandidate
from parsing_agent.repair import HeuristicRepairer
from parsing_agent.visual_repair import (
    VisualRepairTask,
    VisualTableRecovery,
    insert_table_after_anchor,
    replace_table_block,
)
from parsing_agent.workflow import WorkflowRunner

RECOVERED_TABLE = "\n".join(
    [
        "| 구분 | 조사항목 | 조사주기 |",
        "| --- | --- | --- |",
        "| 수질 | BOD | 분기 1회 |",
        "| 대기 | PM10 | 월 1회 |",
    ]
)

UNNUMBERED_LABEL = "사후환경영향조사계획(평가서 요약) 표"


def _source(run_id: str, page_count: int = 8) -> DocumentSource:
    return DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id=run_id,
        extracted_text="원문",
        page_count=page_count,
    )


# --- 번호 없는 라벨 처리 -------------------------------------------------------


def test_replace_table_block_matches_unnumbered_label_text() -> None:
    content = "\n".join(
        [
            "본문입니다.",
            "사후환경영향조사계획(평가서 요약) 표",
            "구분 조사항목 조사주기 수질 BOD 분기",
            "대기 PM10 월 1회 소음 Leq 분기",
            "",
            "다음 본문입니다.",
        ]
    )

    transformed = replace_table_block(content, UNNUMBERED_LABEL, RECOVERED_TABLE)

    assert transformed != content
    assert "| 구분 | 조사항목 | 조사주기 |" in transformed
    assert "다음 본문입니다." in transformed


def test_insert_table_after_label_text_anchor() -> None:
    content = "\n".join(
        [
            "본문입니다.",
            "사후환경영향조사계획(평가서 요약) 표",
            "다음 본문입니다.",
        ]
    )

    transformed = insert_table_after_anchor(content, UNNUMBERED_LABEL, RECOVERED_TABLE)

    lines = transformed.splitlines()
    label_index = lines.index("사후환경영향조사계획(평가서 요약) 표")
    assert "| 구분 | 조사항목 | 조사주기 |" in lines[label_index + 1 : label_index + 4]
    assert "다음 본문입니다." in transformed


def test_insert_table_after_page_marker_anchor() -> None:
    content = "\n".join(
        [
            "<!-- page 6 -->",
            "페이지 본문입니다.",
        ]
    )

    transformed = insert_table_after_anchor(content, "관련 없는 라벨", RECOVERED_TABLE, page_number=6)

    lines = transformed.splitlines()
    assert lines[0] == "<!-- page 6 -->"
    assert "| 구분 | 조사항목 | 조사주기 |" in lines[1:4]


def test_insert_table_without_anchor_is_noop() -> None:
    content = "앵커가 없는 본문입니다."

    assert insert_table_after_anchor(content, "관련 없는 라벨", RECOVERED_TABLE) == content


# --- apply_chunk_repair: 거부 사유 + 삽입 폴백 ----------------------------------


class _FakeRecoverer:
    def __init__(self, confidence: float = 0.9, markdown: str = RECOVERED_TABLE):
        self._confidence = confidence
        self._markdown = markdown

    def recover_task(self, source, content, task):
        return VisualTableRecovery(
            table_label=task.table_label,
            page_number=task.page_number,
            confidence=self._confidence,
            markdown=self._markdown,
            notes=[],
            crop_method="test",
            bbox=None,
        )


def _task(label: str = UNNUMBERED_LABEL, page: int = 6) -> VisualRepairTask:
    return VisualRepairTask(task_id="t1", table_label=label, page_number=page, issue_types=("merged_cell_loss",))


def test_apply_chunk_repair_records_low_confidence_rejection() -> None:
    repairer = HeuristicRepairer(visual_table_recoverer=_FakeRecoverer(confidence=0.2))
    candidate = ParseCandidate(parser_name="p", content="본문", format_name="md")
    rejections: list[dict] = []

    result = repairer.apply_chunk_repair(_source("rej-conf"), candidate, _task(), rejection_sink=rejections)

    assert result is None
    assert rejections[0]["reason"] == "low_confidence"
    assert rejections[0]["confidence"] == 0.2
    assert rejections[0]["table_label"] == UNNUMBERED_LABEL


def test_apply_chunk_repair_records_patch_target_not_found() -> None:
    repairer = HeuristicRepairer(visual_table_recoverer=_FakeRecoverer())
    candidate = ParseCandidate(parser_name="p", content="라벨도 페이지 마커도 없는 본문", format_name="md")
    rejections: list[dict] = []

    result = repairer.apply_chunk_repair(_source("rej-patch"), candidate, _task(), rejection_sink=rejections)

    assert result is None
    assert rejections[0]["reason"] == "patch_target_not_found"


def test_apply_chunk_repair_falls_back_to_insertion_when_no_table_block() -> None:
    repairer = HeuristicRepairer(visual_table_recoverer=_FakeRecoverer())
    # 라벨 라인은 있지만 뒤에 교체할 표 블록이 전혀 없는 후보 (실제 협의내용
    # 문서에서 관찰된 케이스: 파서가 표를 리스트 라인으로 렌더링).
    candidate = ParseCandidate(
        parser_name="p",
        content="\n".join(
            [
                "<!-- page 6 -->",
                "- ※ 협의내용 반영결과 통보시 사후환경영향조사계획에 수정 반영 제출",
            ]
        ),
        format_name="md",
    )
    rejections: list[dict] = []

    result = repairer.apply_chunk_repair(_source("insert-fallback"), candidate, _task(), rejection_sink=rejections)

    assert result is not None
    repaired_candidate, action = result
    assert "| 구분 | 조사항목 | 조사주기 |" in repaired_candidate.content
    assert "inserting after" in action.description
    assert rejections == []


# --- workflow: 거부 사유가 리포트까지 흐르는지 -----------------------------------


def test_repair_chunk_node_propagates_rejections() -> None:
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False),
        repairer=HeuristicRepairer(visual_table_recoverer=_FakeRecoverer(confidence=0.1)),
    )

    result = runner._repair_chunk_node(
        {
            "source": _source("node-rejections"),
            "task": _task(),
            "candidate": ParseCandidate(parser_name="p", content="본문", format_name="md"),
        }
    )

    chunk_result = result["repair_task_results"][0]
    assert chunk_result.candidate is None
    assert chunk_result.rejections[0]["reason"] == "low_confidence"


def test_repair_chunk_node_records_chunk_exception_reason() -> None:
    class _Exploding(HeuristicRepairer):
        def apply_chunk_repair(self, source, candidate, task, rejection_sink=None):
            raise RuntimeError("vision api down")

    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False),
        repairer=_Exploding(),
    )

    result = runner._repair_chunk_node(
        {
            "source": _source("node-exc"),
            "task": SimpleNamespace(task_id="t9"),
            "candidate": ParseCandidate(parser_name="p", content="본문", format_name="md"),
        }
    )

    chunk_result = result["repair_task_results"][0]
    assert chunk_result.candidate is None
    assert chunk_result.rejections[0]["reason"] == "chunk_exception"
    assert "RuntimeError" in chunk_result.rejections[0]["error"]


def test_text_label_anchor_prefers_body_over_toc() -> None:
    # 같은 라벨이 목차와 본문에 모두 등장하면 본문(표가 뒤따르거나 마지막 등장)을 고른다
    content = "\n".join(
        [
            "목차",
            "2. 사후환경영향조사계획(평가서 요약) 표 ......... 6",
            "본문 시작",
            "사후환경영향조사계획(평가서 요약) 표",
            "본문 끝",
        ]
    )

    transformed = insert_table_after_anchor(content, UNNUMBERED_LABEL, RECOVERED_TABLE)

    lines = transformed.splitlines()
    body_index = lines.index("사후환경영향조사계획(평가서 요약) 표")
    assert "| 구분 | 조사항목 | 조사주기 |" in lines[body_index + 1 : body_index + 4]
    # 목차 줄 다음에는 삽입되지 않았다
    toc_index = next(i for i, l in enumerate(lines) if l.startswith("2."))
    assert "| 구분" not in (lines[toc_index + 1] if toc_index + 1 < len(lines) else "")
