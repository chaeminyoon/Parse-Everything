from __future__ import annotations

from math import isclose

from parsing_agent.config import WorkflowConfig
from parsing_agent.evaluation import (
    DeterministicEvaluator,
    TABLE_ISSUE_MISSING_HEADER,
    TABLE_ISSUE_NUMERIC_TOKEN_BREAK,
    TABLE_ISSUE_SPLIT_MULTIPAGE,
    TABLE_ISSUE_TEXT_DUPLICATION,
    calculate_structure_retention,
    calculate_table_preservation,
)
from parsing_agent.models import DocumentSource, JudgeResult, ParseCandidate


def _build_source(tmp_path, media_type: str, extracted_text: str, suffix: str) -> DocumentSource:
    path = tmp_path / f"source{suffix}"
    path.write_text("", encoding="utf-8")
    return DocumentSource(
        path=path,
        media_type=media_type,
        size_bytes=0,
        run_id="test-run",
        extracted_text=extracted_text,
    )


class _IssueJudge:
    def __init__(self, issues: list[str]) -> None:
        self._issues = issues

    def judge(self, source, candidate, metrics) -> JudgeResult:
        del source, candidate, metrics
        return JudgeResult(overall_score=0.8, issues=list(self._issues))


def test_pdf_structure_retention_uses_inline_pdf_cues_in_flattened_text(tmp_path) -> None:
    source_text = (
        "\uC81C4\uC7A5 \uC9C0\uC5ED\uAC1C\uD669 4.1 \uC0AC\uC5C5\uC9C0\uAD6C\uC758 \uC9C0\uB9AC\uC801 \uD2B9\uC131 "
        "4.2 \uD1A0\uC9C0\uC774\uC6A9 \uD604\uD669 \uAC00. \uD589\uC815\uAD6C\uC5ED \uB098. \uC9C0\uBAA9\uBCC4 \uD1A0\uC9C0\uC774\uC6A9 \uD604\uD669"
    )
    candidate_text = """# \uC81C4\uC7A5 \uC9C0\uC5ED\uAC1C\uD669
## 4.1 \uC0AC\uC5C5\uC9C0\uAD6C\uC758 \uC9C0\uB9AC\uC801 \uD2B9\uC131

## 4.2 \uD1A0\uC9C0\uC774\uC6A9 \uD604\uD669

### \uAC00. \uD589\uC815\uAD6C\uC5ED

### \uB098. \uC9C0\uBAA9\uBCC4 \uD1A0\uC9C0\uC774\uC6A9 \uD604\uD669
"""
    legacy_score = calculate_structure_retention(source_text, candidate_text)
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert legacy_score == 0.0
    assert metrics.structure_retention == 1.0


def test_pdf_structure_retention_ignores_dotted_body_values(tmp_path) -> None:
    source_text = (
        "\uC81C4\uC7A5 \uC9C0\uC5ED\uAC1C\uD669 4.1 \uAC1C\uC694 "
        "\uBA74\uC801 512.3 63.0 4.2 \uD604\uD669"
    )
    candidate_text = """# \uC81C4\uC7A5 \uC9C0\uC5ED\uAC1C\uD669
## 4.1 \uAC1C\uC694

## 4.2 \uD604\uD669
"""
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert metrics.structure_retention == 1.0


def test_pdf_structure_retention_counts_repeated_subsection_cues(tmp_path) -> None:
    source_text = (
        "\uC81C4\uC7A5 \uC9C0\uC5ED\uAC1C\uD669 4.1 \uC0AC\uC5C5\uC9C0\uAD6C\uC758 \uC9C0\uB9AC\uC801 \uD2B9\uC131 "
        "\uAC00. \uD589\uC815\uAD6C\uC5ED \uAC00. \uC138\uBD80\uD56D\uBAA9 \uB098. \uB2E4\uC74C \uD56D\uBAA9"
    )
    candidate_text = """# \uC81C4\uC7A5 \uC9C0\uC5ED\uAC1C\uD669
## 4.1 \uC0AC\uC5C5\uC9C0\uAD6C\uC758 \uC9C0\uB9AC\uC801 \uD2B9\uC131

### \uAC00. \uD589\uC815\uAD6C\uC5ED

### \uB098. \uB2E4\uC74C \uD56D\uBAA9
"""
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert isclose(metrics.structure_retention, 0.8)


def test_pdf_without_structure_or_table_cues_uses_pdf_unlabeled_table_fallback(tmp_path) -> None:
    source_text = """alpha beta gamma
10 20 30
40 50 60

delta epsilon zeta
70 80 90
100 110 120
"""
    candidate_text = """| alpha | beta | gamma |
| --- | ---: | ---: |
| 10 | 20 | 30 |
| 40 | 50 | 60 |

| delta | epsilon | zeta |
| --- | ---: | ---: |
| 70 | 80 | 90 |
| 100 | 110 | 120 |
"""
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert metrics.structure_retention == calculate_structure_retention(source_text, candidate_text)
    assert metrics.table_preservation == 1.0


def test_pdf_unlabeled_table_fallback_does_not_give_full_credit_for_wrong_content(tmp_path) -> None:
    source_text = """alpha beta gamma
10 20 30
40 50 60

delta epsilon zeta
70 80 90
100 110 120
"""
    candidate_text = """| item | note | misc |
| --- | --- | --- |
| wrong | data | only |
| none | match | here |

| item2 | note2 | misc2 |
| --- | --- | --- |
| also | wrong | data |
| still | not | match |
"""
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert metrics.table_preservation < 1.0
    assert isclose(metrics.table_preservation, 0.5)


def test_pdf_table_preservation_uses_table_labels_and_matching_table_content(tmp_path) -> None:
    source_text = """\uD45C < 4.2-2> landuse area
landuse area
yeosu 512.3
\uD45C < 4.2-3> zoning status
zoning area
yeosu 302.7
"""
    candidate_text = """- <\uD45C 4.2-2> landuse area
| landuse | area |
| --- | ---: |
| yeosu | 512.3 |

- <\uD45C 4.2-3> zoning status
| zoning | area |
| --- | ---: |
| yeosu | 302.7 |
"""
    legacy_score = calculate_table_preservation(source_text, candidate_text)
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert legacy_score == 0.0
    assert metrics.table_preservation == 1.0


def test_pdf_table_preservation_accepts_html_table_near_label(tmp_path) -> None:
    source_text = """\uD45C < 4.2-2> landuse area
landuse area
yeosu 512.3
"""
    candidate_text = """- <\uD45C 4.2-2> landuse area
<table>
  <tr>
    <th>landuse</th>
    <th>area</th>
  </tr>
  <tr>
    <td>yeosu</td>
    <td>512.3</td>
  </tr>
</table>
"""
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert metrics.table_preservation == 1.0


def test_pdf_table_preservation_gives_partial_credit_when_caption_exists_without_table(tmp_path) -> None:
    source_text = """\uD45C < 4.2-2> landuse area
landuse area
yeosu 512.3
\uD45C < 4.2-3> zoning status
zoning area
yeosu 302.7
"""
    candidate_text = """- <\uD45C 4.2-2> landuse area
| landuse | area |
| --- | ---: |
| yeosu | 512.3 |

- <\uD45C 4.2-3> zoning status
follow-up content remains plain text only.
"""
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert isclose(metrics.table_preservation, 0.75)


def test_pdf_table_preservation_rejects_nearby_but_obviously_wrong_table(tmp_path) -> None:
    source_text = """\uD45C < 4.2-2> landuse area
landuse area
yeosu 512.3
"""
    candidate_text = """- <\uD45C 4.2-2> landuse area
| item | note |
| --- | --- |
| different | unrelated |
"""
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert isclose(metrics.table_preservation, 0.625)


def test_pdf_table_preservation_prefers_best_matching_table_among_multiple_nearby_tables(tmp_path) -> None:
    source_text = """\uD45C < 4.2-2> landuse area
landuse area
yeosu 512.3
"""
    candidate_text = """- <\uD45C 4.2-2> landuse area
| item | note |
| --- | --- |
| different | unrelated |

| landuse | area |
| --- | ---: |
| yeosu | 512.3 |
"""
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert metrics.table_preservation == 1.0


def test_pdf_table_preservation_does_not_reuse_one_table_for_multiple_labels(tmp_path) -> None:
    source_text = """\uD45C < 4.2-2> status area
status area
alpha 10.0
\uD45C < 4.2-3> status area
status area
alpha 10.0
"""
    candidate_text = """- <\uD45C 4.2-2> status area
- <\uD45C 4.2-3> status area
| status | area |
| --- | ---: |
| alpha | 10.0 |
"""
    source = _build_source(tmp_path, "application/pdf", source_text, ".pdf")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert isclose(metrics.table_preservation, 0.75)


def test_non_pdf_inputs_keep_legacy_structure_and_table_scoring(tmp_path) -> None:
    source_text = """\uD45C 4.2-2 landuse area
yeosu 512.3
"""
    candidate_text = """- <\uD45C 4.2-2> landuse area
| landuse | area |
| --- | ---: |
| yeosu | 512.3 |
"""
    source = _build_source(tmp_path, "text/plain", source_text, ".txt")
    candidate = ParseCandidate(parser_name="parser", content=candidate_text, format_name="md")

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert metrics.structure_retention == calculate_structure_retention(source_text, candidate_text)
    assert metrics.table_preservation == calculate_table_preservation(source_text, candidate_text)


def test_pdf_table_issue_taxonomy_uses_judge_issues_and_repeated_table_blocks(tmp_path) -> None:
    source = _build_source(tmp_path, "application/pdf", "표 4.2-2 source table", ".pdf")
    candidate = ParseCandidate(
        parser_name="layout-first-pdf",
        content=(
            "표 4.2-2 sample\n"
            "| amount | total |\n"
            "| --- | --- |\n"
            "| 12 . 3 | 45 |\n\n"
            "| amount | total |\n"
            "| --- | --- |\n"
            "| 12 . 3 | 45 |\n"
        ),
        format_name="md",
    )

    metrics = DeterministicEvaluator(
        WorkflowConfig(),
        judge=_IssueJudge(
            [
                "Table 4.2-2 missing header row.",
                "Table 4.2-2 numeric token break in the amount column.",
                "Table 4.2-2 duplicated table text around the caption.",
            ]
        ),
    ).evaluate(source, candidate)

    assert metrics.table_issues == [
        TABLE_ISSUE_MISSING_HEADER,
        TABLE_ISSUE_NUMERIC_TOKEN_BREAK,
        TABLE_ISSUE_TEXT_DUPLICATION,
    ]


def test_pdf_table_issue_taxonomy_recognizes_korean_judge_table_failures(tmp_path) -> None:
    source = _build_source(tmp_path, "application/pdf", "표 4.2-2 source table", ".pdf")
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="표 4.2-2\nbroken table",
        format_name="md",
    )

    metrics = DeterministicEvaluator(
        WorkflowConfig(),
        judge=_IssueJudge(
            [
                "표 4.2-2가 표 형태로 렌더링되지 않고 행/열 구조 보존이 미흡함.",
                "표 4.2-2에서 단위 오기와 숫자 깨짐이 있음.",
            ]
        ),
    ).evaluate(source, candidate)

    assert metrics.table_issues == [
        TABLE_ISSUE_MISSING_HEADER,
        TABLE_ISSUE_NUMERIC_TOKEN_BREAK,
    ]


def test_pdf_table_issue_taxonomy_uses_table_region_continuation_metadata(tmp_path) -> None:
    source = _build_source(tmp_path, "application/pdf", "표 4.2-2 source table", ".pdf")
    candidate = ParseCandidate(
        parser_name="layout-first-pdf",
        content="표 4.2-2 sample",
        format_name="md",
        metadata={
            "table_regions": [
                {
                    "table_id": "p2-t1",
                    "page": 2,
                    "continued_from_page": 1,
                    "extraction_mode": "reference",
                }
            ]
        },
    )

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert metrics.table_issues == [TABLE_ISSUE_SPLIT_MULTIPAGE]


def test_pdf_table_issue_taxonomy_uses_support_parser_table_region_metadata(tmp_path) -> None:
    source = _build_source(tmp_path, "application/pdf", "??4.2-2 source table", ".pdf")
    support_metadata = {
        "table_regions": [
            {
                "table_id": "p2-t1",
                "page": 2,
                "continued_from_page": 1,
                "extraction_mode": "reference",
            }
        ]
    }
    candidate = ParseCandidate(
        parser_name="opendataloader-pdf",
        content="??4.2-2 sample",
        format_name="md",
        metadata={"support_parser_metadata": {"layout-first-pdf": support_metadata}},
    )

    metrics = DeterministicEvaluator(WorkflowConfig()).evaluate(source, candidate)

    assert metrics.table_issues == [TABLE_ISSUE_SPLIT_MULTIPAGE]
