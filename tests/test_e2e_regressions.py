"""그래프 E2E 검증에서 발견된 갭들의 회귀 테스트.

1. 본문 누락 시 LLM 전용 repair target 생성
2. externalize된 source가 inspect/repair 노드에서 materialize되는지
3. no-op 휴리스틱 스텝도 attempted route로 기록돼 LLM 승격이 가능한지
"""

from pathlib import Path

from parsing_agent.config import WorkflowConfig
from parsing_agent.models import DocumentSource, EvaluationMetrics, ParseCandidate
from parsing_agent.repair import HeuristicRepairer, RepairTarget, identify_repair_targets
from parsing_agent.workflow import WorkflowRunner


def _source(run_id: str, text: str) -> DocumentSource:
    return DocumentSource(
        path=Path("sample.txt"),
        media_type="text/plain",
        size_bytes=0,
        run_id=run_id,
        extracted_text=text,
    )


def _metrics(coverage: float) -> EvaluationMetrics:
    return EvaluationMetrics(
        text_coverage=coverage,
        normalized_similarity=coverage,
        structure_retention=0.9,
        table_preservation=0.9,
        empty_block_penalty=0.0,
        repetition_penalty=0.0,
        total_score=coverage,
    )


def test_low_coverage_produces_llm_repair_target() -> None:
    source = _source("cov-llm", "본문 첫 문장이다.\n본문 둘째 문장이다.\n본문 셋째 문장이다.")
    candidate = ParseCandidate(parser_name="p", content="본문 첫 문장이다.", format_name="md")

    targets = identify_repair_targets(source, candidate, _metrics(0.4))

    llm_targets = [t for t in targets if t.repairability == "llm"]
    assert len(llm_targets) == 1
    assert llm_targets[0].issue_type == "text_coverage_missing_content"
    assert llm_targets[0].route_name == "llm_restore_missing_content"
    assert llm_targets[0].expected_gain > 0


def test_high_coverage_does_not_produce_llm_repair_target() -> None:
    source = _source("cov-ok", "본문 첫 문장이다.")
    candidate = ParseCandidate(parser_name="p", content="본문 첫 문장이다.", format_name="md")

    targets = identify_repair_targets(source, candidate, _metrics(0.95))

    assert not [t for t in targets if t.repairability == "llm"]


def test_inspect_node_materializes_externalized_source() -> None:
    runner = WorkflowRunner(config=WorkflowConfig(judge_weight=0, langsmith_tracing=False))
    full_text = "\n".join(
        [
            "제1장 사업개요",
            "본문 내용이다.",
            "제2장 저감방안",
            "저감 내용이다.",
        ]
    )
    source = runner._externalize_source_text(_source("inspect-materialize", full_text))
    assert source.extracted_text is None
    candidate = ParseCandidate(parser_name="p", content="제1장 사업개요\n본문 내용이다.", format_name="md")

    result = runner._inspect_quality_issues_node(
        {"source": source, "candidate": candidate, "metrics": _metrics(0.5)}
    )

    # extracted_text가 materialize돼야 누락 구조 라인(제2장)이 감지된다.
    route_names = {t.route_name for t in result["repair_targets"]}
    assert "recover_missing_source_lines" in route_names


def test_repair_node_records_attempted_routes_even_without_actions() -> None:
    runner = WorkflowRunner(
        config=WorkflowConfig(judge_weight=0, langsmith_tracing=False),
        repairer=HeuristicRepairer(),
    )
    candidate = ParseCandidate(parser_name="p", content="깨끗한 본문이다.", format_name="md")
    target = RepairTarget(
        target_kind="text",
        issue_type="line_repetition_noise",
        route_name="deduplicate_lines",
        description="dedupe",
    )
    routed = runner._route_repair_strategy_node({"repair_targets": [target]})

    result = runner._repair_candidate_node(
        {
            "source": _source("noop-attempt", "깨끗한 본문이다."),
            "candidate": candidate,
            "metrics": _metrics(0.5),
            "repair_plan": routed["repair_plan"],
            "iteration_count": 0,
        }
    )

    # 고칠 게 없어 액션은 0개지만 route는 시도된 것으로 기록돼야 한다.
    assert result["repairs"] == []
    assert "deduplicate_lines" in result["attempted_repair_routes"]
