"""P1 품질 개선 검증: 한국어 줄 병합, 표 열 구조 일관성 메트릭."""

from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import DeterministicEvaluator, calculate_table_structure_consistency
from parsing_agent.models import DocumentSource, ParseCandidate
from parsing_agent.repair import _merge_wrapped_lines, _should_merge_lines


# --- 한국어 줄 병합 -----------------------------------------------------------


def test_korean_wrapped_sentence_lines_are_merged() -> None:
    left = "환경영향평가는 사업의 시행으로 인하여 자연환경과 생활환경에"
    right = "미치는 영향을 사전에 조사·예측·평가하는 제도이다."

    assert _should_merge_lines(left, right) is True
    merged = _merge_wrapped_lines(f"{left}\n{right}")
    assert merged.splitlines()[0] == f"{left} {right}"


def test_korean_sentence_terminal_line_is_not_merged() -> None:
    left = "본 사업은 항만 재개발을 목적으로 시행되는 광역 기반시설 사업이다"
    right = "사업 대상 지역은 전라남도 광양시 일원이다."

    assert _should_merge_lines(left, right) is False


def test_short_korean_heading_line_is_not_merged() -> None:
    heading = "지역개황"
    body = "사업 대상 지역의 지형 및 지질 현황은 다음과 같다."

    assert _should_merge_lines(heading, body) is False


def test_korean_merge_ignores_table_lines() -> None:
    left = "| 구분 | 조사항목 | 조사지점 | 조사주기 |"
    right = "수질 조사는 분기별로 수행하며 지점별 측정 항목은 다음과"

    assert _should_merge_lines(left, right) is False


# --- 표 열 구조 일관성 --------------------------------------------------------


def test_table_structure_consistency_is_perfect_for_wellformed_table() -> None:
    text = "\n".join(
        [
            "| 구분 | 항목 |",
            "| --- | --- |",
            "| 수질 | BOD |",
            "| 대기 | PM10 |",
        ]
    )
    assert calculate_table_structure_consistency(text) == 1.0


def test_table_structure_consistency_penalizes_inconsistent_columns() -> None:
    text = "\n".join(
        [
            "| 구분 | 항목 | 지점 |",
            "| --- | --- | --- |",
            "| 수질 | BOD |",
            "| 대기 | PM10 |",
            "| 소음 | Leq |",
        ]
    )
    score = calculate_table_structure_consistency(text)
    assert score < 1.0


def test_table_structure_consistency_defaults_to_one_without_tables() -> None:
    assert calculate_table_structure_consistency("표 없는 일반 본문입니다.") == 1.0


def test_evaluator_notes_inconsistent_table_structure() -> None:
    source = DocumentSource(
        path=Path("sample.txt"),
        media_type="text/plain",
        size_bytes=0,
        run_id="table-consistency",
        extracted_text="구분 항목 지점 수질 BOD 대기 PM10 소음 Leq",
    )
    candidate = ParseCandidate(
        parser_name="text-fallback",
        content="\n".join(
            [
                "| 구분 | 항목 | 지점 |",
                "| --- | --- | --- |",
                "| 수질 | BOD |",
                "| 대기 | PM10 |",
                "| 소음 | Leq |",
            ]
        ),
        format_name="md",
    )
    evaluator = DeterministicEvaluator(WorkflowConfig(judge_weight=0))

    metrics = evaluator.evaluate(source, candidate)

    assert any("column structure inconsistent" in note for note in metrics.notes)
