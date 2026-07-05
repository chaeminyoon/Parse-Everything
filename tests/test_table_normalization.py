"""골든 라벨 1·2호가 지목한 표 결함의 수리 검증.

- 수리 후 남은 표 잔재(같은 내용이 두 번) 제거
- 세로 병합 해제로 비어버린 선두 분류 열에 값 복제
- 하나로 붙어버린 통합표를 별개 표로 분리
"""

from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.models import DocumentSource, EvaluationMetrics, ParseCandidate
from parsing_agent.repair import (
    apply_table_normalizations,
    _classify_repair_directives,
    _fill_merged_leading_cells,
    _has_duplicate_table_blocks,
    _has_fused_table_blocks,
    _has_merged_leading_gaps,
    _remove_duplicate_table_blocks,
    _split_fused_tables,
)

# 골든 라벨 2호 사례 축소판: 그림 캡션 위에 표 잔재, 아래에 온전한 표
DUPLICATED = "\n".join(
    [
        "| 준설공 (수역시설) | 면적 : 864,940㎡ |  |",
        "| 준설공 (수역시설) | 준설량 : 4,546,086㎥ |  |",
        "| 부대공 | 1식 |  |",
        "",
        "(그림 2.4-2) 사업지구 전경사진",
        "",
        "<표 2.4-1> 사업규모",
        "",
        "| 구분 | 내용 | 규격 |",
        "| --- | --- | --- |",
        "| 접안시설 | 작업 돌핀 | 48.0m |",
        "| 준설공 (수역시설) | 면적 : 864,940㎡ 준설량 : 4,546,086㎥ | |",
        "| 부대공 | 1식 | |",
    ]
)


def test_duplicate_fragment_block_is_removed() -> None:
    assert _has_duplicate_table_blocks(DUPLICATED)

    cleaned = _remove_duplicate_table_blocks(DUPLICATED)

    # 잔재(위쪽 3행)는 사라지고 본 표와 캡션은 남는다
    assert cleaned.count("864,940") == 1
    assert "(그림 2.4-2)" in cleaned
    assert "| 접안시설 | 작업 돌핀 | 48.0m |" in cleaned


def test_distinct_tables_are_not_treated_as_duplicates() -> None:
    text = "\n".join(
        [
            "| 구분 | 값 |",
            "| --- | --- |",
            "| 수질 | BOD |",
            "",
            "| 항목 | 기준 |",
            "| --- | --- |",
            "| 소음 | 65dB |",
        ]
    )
    assert not _has_duplicate_table_blocks(text)
    assert _remove_duplicate_table_blocks(text) == text


# 골든 라벨 2호 표 2.4-2 사례: 병합 해제로 비어버린 선두 열
MERGED_GAPS = "\n".join(
    [
        "| 구분 | 구간 | 대상선박 | 준설량 | 비고 |",
        "| --- | --- | --- | --- | --- |",
        "| 1단계 | 박지구간 | 10만DWT급 | 2,082,718 | 항로폭 520m |",
        "|  | 항로구간 |  | 2,359,864 |  |",
    ]
)


def test_merged_leading_cells_are_filled_from_above() -> None:
    assert _has_merged_leading_gaps(MERGED_GAPS)

    filled = _fill_merged_leading_cells(MERGED_GAPS)

    assert "| 1단계 | 항로구간 |" in filled
    # 뒤쪽 열(대상선박·비고)은 건드리지 않는다 — 원래 비어 있을 수 있다
    lines = filled.splitlines()
    last = [cell.strip() for cell in lines[-1].split("|")[1:-1]]
    assert last[2] == ""
    assert last[4] == ""


def test_fill_does_not_touch_fully_labeled_rows() -> None:
    text = "\n".join(
        [
            "| 구분 | 값 |",
            "| --- | --- |",
            "| 수질 | BOD |",
            "| 대기 | PM10 |",
        ]
    )
    assert not _has_merged_leading_gaps(text)
    assert _fill_merged_leading_cells(text) == text


# 통합표: 하나의 파이프 블록 안에 캡션과 두 번째 헤더가 끼어 있는 경우
FUSED = "\n".join(
    [
        "| 구분 | 값 |",
        "| --- | --- |",
        "| 수질 | BOD |",
        "| <표 2.4-2> 수역시설 준설계획 |  |",
        "| 구간 | 준설량 |",
        "| --- | --- |",
        "| 박지구간 | 2,082,718 |",
    ]
)


def test_fused_table_is_split_at_embedded_caption() -> None:
    assert _has_fused_table_blocks(FUSED)

    split = _split_fused_tables(FUSED)
    lines = split.splitlines()

    # 캡션이 표 밖의 일반 텍스트 줄로 나온다
    assert "<표 2.4-2> 수역시설 준설계획" in lines
    caption_index = lines.index("<표 2.4-2> 수역시설 준설계획")
    assert lines[caption_index - 1] == ""
    assert lines[caption_index + 1] == ""
    # 두 표가 분리돼 각자 헤더를 가진다
    assert "| 구분 | 값 |" in lines
    assert "| 구간 | 준설량 |" in lines


def test_second_header_signature_splits_block() -> None:
    text = "\n".join(
        [
            "| 구분 | 값 |",
            "| --- | --- |",
            "| 수질 | BOD |",
            "| 항목 | 기준 |",
            "| --- | --- |",
            "| 소음 | 65dB |",
        ]
    )
    assert _has_fused_table_blocks(text)

    split = _split_fused_tables(text)

    blocks = [b for b in split.split("\n\n") if b.strip()]
    assert len(blocks) == 2
    assert blocks[1].startswith("| 항목 | 기준 |")


# 채점 루프 밖(finalize) 정규화로 연결됐는지


def test_apply_table_normalizations_reports_applied_transforms() -> None:
    combined = DUPLICATED + "\n\n" + MERGED_GAPS + "\n\n" + FUSED

    normalized, applied = apply_table_normalizations(combined)

    assert applied == [
        "remove_duplicate_table_blocks",
        "split_fused_tables",
        "fill_merged_leading_cells",
    ]
    assert normalized.count("864,940") == 1
    assert "| 1단계 | 항로구간 |" in normalized


def test_normalizations_stay_out_of_the_scored_repair_loop() -> None:
    # 현재 채점기가 이 정규화들을 감점하므로 루프 안에서 돌면 롤백당한다.
    # 반드시 finalize 단계 전용으로 남아야 한다.
    source = DocumentSource(
        path=Path("sample.pdf"),
        media_type="application/pdf",
        size_bytes=0,
        run_id="table-normalization",
        extracted_text="원문",
    )
    candidate = ParseCandidate(parser_name="p", content=DUPLICATED, format_name="md")
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.9,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.9,
    )

    routes = {d.route_name for d in _classify_repair_directives(source, candidate, metrics)}

    assert "remove_duplicate_table_blocks" not in routes


def test_finalize_node_applies_normalizations_and_reports_them() -> None:
    from parsing_agent.config import WorkflowConfig
    from parsing_agent.workflow import WorkflowRunner

    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))
    source = DocumentSource(
        path=Path("sample.txt"),
        media_type="text/plain",
        size_bytes=0,
        run_id="finalize-normalization",
        extracted_text="원문",
    )
    candidate = ParseCandidate(parser_name="p", content=DUPLICATED, format_name="md")
    metrics = EvaluationMetrics(
        text_coverage=0.9,
        normalized_similarity=0.9,
        structure_retention=0.9,
        table_preservation=0.9,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=0.9,
    )

    result = runner._finalize_output_node(
        {"source": source, "candidate": candidate, "metrics": metrics}
    )["result"]

    assert result.best_candidate.content.count("864,940") == 1
    assert result.report["monitoring"]["post_loop_normalizations"] == ["remove_duplicate_table_blocks"]
