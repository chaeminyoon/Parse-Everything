from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import re

from parsing_agent.interfaces import CandidateRepairer
from parsing_agent.models import DocumentSource, EvaluationMetrics, ParseCandidate, RepairAction
from parsing_agent.visual_repair import (
    VisualRepairTask,
    _page_table_selector_from_label,
    replace_page_table_block,
    replace_table_block,
)


def _excerpt(text: str, limit: int = 120) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


_LIST_RE = re.compile(r"^([-*+]|\d+\.)\s+")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?:<[^>]+>|[^)]+)\)")
_HEADING_RE = re.compile(r"^\s*#+\s*(.+?)\s*$")
_KEY_VALUE_RE = re.compile(r"^\s*(?P<label>[^:：|]{1,80}?)(?:[:：]|\s{2,})(?P<value>.+?)\s*$")


@dataclass(frozen=True, slots=True)
class RepairDirective:
    issue_type: str
    route_name: str
    action_name: str
    description: str
    transform: callable


@dataclass(frozen=True, slots=True)
class RepairTarget:
    target_kind: str
    issue_type: str
    route_name: str
    description: str


def _join_lines(lines: list[str], original_text: str) -> str:
    normalized = "\n".join(lines)
    if original_text.endswith("\n"):
        normalized += "\n"
    return normalized


def _collapse_blank_lines(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip():
            blank_run = 0
            result.append(line)
            continue
        if blank_run == 0:
            result.append("")
        blank_run += 1

    return _join_lines(result, text)


def _contains_markdown_images(text: str) -> bool:
    return bool(_IMAGE_RE.search(text))


def _strip_markdown_images(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and _IMAGE_RE.fullmatch(stripped):
            continue
        cleaned = _IMAGE_RE.sub("", line)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).rstrip()
        result.append(cleaned)
    return _join_lines(result, text)


def _remove_repeated_lines(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result: list[str] = []
    previous_non_empty: str | None = None
    for line in lines:
        normalized = line.strip().lower()
        if normalized and normalized == previous_non_empty:
            continue
        result.append(line)
        if normalized:
            previous_non_empty = normalized

    return _join_lines(result, text)


def _has_duplicate_headings(text: str) -> bool:
    seen: set[str] = set()
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match is None:
            continue
        normalized = match.group(1).strip().lower()
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


def _remove_duplicate_headings(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        match = _HEADING_RE.match(line)
        if match is None:
            result.append(line)
            continue
        normalized = match.group(1).strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(line)
    return _join_lines(result, text)


def _is_structural_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if _LIST_RE.match(stripped):
        return True
    return stripped.count("|") >= 2 and (stripped.startswith("|") or stripped.endswith("|"))


def _should_merge_lines(current: str, next_line: str) -> bool:
    left = current.rstrip()
    right = next_line.lstrip()
    if not left or not right:
        return False
    if _is_structural_line(left) or _is_structural_line(right):
        return False
    if left.endswith((".", "!", "?", ":", ";")):
        return False
    if left.endswith("-"):
        return True
    first_char = right[:1]
    return first_char.islower() or first_char.isdigit() or first_char in {'"', "'", "(", "["}


def _has_wrapped_line_sequences(text: str) -> bool:
    lines = text.splitlines()
    return any(_should_merge_lines(lines[index], lines[index + 1]) for index in range(len(lines) - 1))


def _merge_wrapped_lines(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    result: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        while index + 1 < len(lines) and _should_merge_lines(current, lines[index + 1]):
            next_line = lines[index + 1]
            if current.rstrip().endswith("-"):
                current = current.rstrip()[:-1] + next_line.lstrip()
            else:
                current = current.rstrip() + " " + next_line.lstrip()
            index += 1
        result.append(current)
        index += 1

    return _join_lines(result, text)


def _looks_like_corrupted_table_line(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    if _IMAGE_RE.search(stripped):
        return True
    return stripped.count("|") >= 2 and not (stripped.startswith("|") or stripped.endswith("|"))


def _has_table_layout_noise(text: str) -> bool:
    return any(_looks_like_corrupted_table_line(line) for line in text.splitlines())


def _normalize_table_layout(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    result: list[str] = []
    for line in lines:
        if "|" not in line:
            result.append(line)
            continue
        cleaned = _IMAGE_RE.sub("", line)
        if cleaned.count("|") >= 2:
            cells = [cell.strip() for cell in cleaned.split("|")]
            if cells and not cells[0]:
                cells = cells[1:]
            if cells and not cells[-1]:
                cells = cells[:-1]
            normalized_cells = [cell if cell else " " for cell in cells]
            cleaned = "| " + " | ".join(normalized_cells) + " |"
        result.append(cleaned.rstrip())
    return _join_lines(result, text)


def _looks_like_table_text_row(line: str) -> bool:
    if "|" in line:
        return False
    match = _KEY_VALUE_RE.match(line)
    if match is None:
        return False
    label = match.group("label").strip()
    value = match.group("value").strip()
    if not label or not value:
        return False
    if len(value.split()) > 12 and not re.search(r"\d", value):
        return False
    return True


def _has_table_text_blocks(text: str) -> bool:
    lines = text.splitlines()
    run_length = 0
    for line in lines:
        if _looks_like_table_text_row(line):
            run_length += 1
            if run_length >= 2:
                return True
            continue
        run_length = 0
    return False


def _reconstruct_table_text_blocks(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    result: list[str] = []
    index = 0
    while index < len(lines):
        if not _looks_like_table_text_row(lines[index]):
            result.append(lines[index])
            index += 1
            continue

        block: list[tuple[str, str]] = []
        while index < len(lines) and _looks_like_table_text_row(lines[index]):
            match = _KEY_VALUE_RE.match(lines[index])
            assert match is not None
            block.append((match.group("label").strip(), match.group("value").strip()))
            index += 1

        if len(block) < 2:
            label, value = block[0]
            result.append(f"{label}: {value}")
            continue

        result.append("| 항목 | 값 |")
        result.append("| --- | --- |")
        for label, value in block:
            result.append(f"| {label} | {value} |")

    return _join_lines(result, text)


def _is_repeated_boundary_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _is_structural_line(stripped):
        return False
    if len(stripped) > 80:
        return False
    return len(stripped.split()) <= 8


def _remove_repeated_boundary_lines(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 3:
        return text

    boundary_counts: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not _is_repeated_boundary_candidate(stripped):
            continue
        previous_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
        if previous_blank or next_blank:
            normalized = stripped.lower()
            boundary_counts[normalized] = boundary_counts.get(normalized, 0) + 1

    repeated_boundaries = {line for line, count in boundary_counts.items() if count >= 2}
    if not repeated_boundaries:
        return text

    result: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        previous_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
        if stripped.lower() in repeated_boundaries and (previous_blank or next_blank):
            continue
        result.append(line)

    normalized = _collapse_blank_lines(_join_lines(result, text)).strip("\n")
    if not normalized:
        return normalized
    if text.endswith("\n"):
        normalized += "\n"
    return normalized


def _is_pdf_candidate(source: DocumentSource, candidate: ParseCandidate) -> bool:
    source_path = candidate.source_path or source.path
    return source.media_type == "application/pdf" or source_path.suffix.lower() == ".pdf"


def _should_defer_table_rewrite(source: DocumentSource, candidate: ParseCandidate, metrics: EvaluationMetrics) -> bool:
    """PDF 표 이슈가 명확하면 heuristic 표 재작성보다 visual repair를 우선한다.

    PDF candidate에서 table issue가 이미 잡혔다면, 텍스트를 억지로
    다시 쓰는 것보다 원본 PDF 이미지를 다시 보는 경로가 더 적절하다고
    판단한다.
    """
    return _is_pdf_candidate(source, candidate) and bool(metrics.table_issues)


def _classify_repair_directives(
    source: DocumentSource,
    candidate: ParseCandidate,
    metrics: EvaluationMetrics,
) -> list[RepairDirective]:
    """metric과 콘텐츠 패턴을 heuristic repair 지시문으로 바꾼다.

    각 directive는 `issue_type`, `route_name`, 설명, 실제 변환 함수를 함께
    가진다. 기준은 의도적으로 단순하며, 대부분 `0.75` 미만 점수이거나
    직접적인 텍스트 패턴이 보일 때 발동한다.
    """
    content = candidate.content
    directives: list[RepairDirective] = []
    seen_issue_types: set[str] = set()
    defer_table_rewrite = _should_defer_table_rewrite(source, candidate, metrics)

    def add_directive(
        issue_type: str,
        route_name: str,
        action_name: str,
        description: str,
        transform,
    ) -> None:
        if issue_type in seen_issue_types:
            return
        seen_issue_types.add(issue_type)
        directives.append(
            RepairDirective(
                issue_type=issue_type,
                route_name=route_name,
                action_name=action_name,
                description=description,
                transform=transform,
            )
        )

    if _contains_markdown_images(content):
        add_directive(
            "image_link_noise",
            "remove_image_noise",
            "strip_markdown_images",
            "Remove markdown image links that add rendering noise or leak into parsed cells.",
            _strip_markdown_images,
        )
    if not defer_table_rewrite and (metrics.table_preservation < 0.75 or _has_table_layout_noise(content)):
        add_directive(
            "table_layout_noise",
            "normalize_table_layout",
            "normalize_table_layout",
            "Normalize table rows and strip inline image fragments from table cells.",
            _normalize_table_layout,
        )
    if not defer_table_rewrite and metrics.table_preservation < 0.75 and _has_table_text_blocks(content):
        add_directive(
            "table_text_block_recovery",
            "reconstruct_table_blocks",
            "reconstruct_table_text_blocks",
            "Reconstruct repeated key/value text rows into a conservative two-column markdown table.",
            _reconstruct_table_text_blocks,
        )
    if metrics.structure_retention < 0.75 or _has_duplicate_headings(content):
        add_directive(
            "structure_heading_noise",
            "deduplicate_headings",
            "remove_duplicate_headings",
            "Remove duplicated markdown headings that distort the document hierarchy.",
            _remove_duplicate_headings,
        )
    if metrics.empty_block_penalty > 0:
        add_directive(
            "blank_line_noise",
            "collapse_blank_runs",
            "collapse_blank_lines",
            "Collapse oversized blank-line runs to a single blank line.",
            _collapse_blank_lines,
        )
    if metrics.repetition_penalty > 0:
        add_directive(
            "boundary_repetition_noise",
            "deduplicate_boundaries",
            "remove_repeated_boundary_lines",
            "Remove repeated short boundary lines that look like headers or footers.",
            _remove_repeated_boundary_lines,
        )
        add_directive(
            "line_repetition_noise",
            "deduplicate_lines",
            "remove_repeated_lines",
            "Remove consecutive repeated non-empty lines.",
            _remove_repeated_lines,
        )
    if _has_wrapped_line_sequences(content):
        add_directive(
            "wrapped_line_noise",
            "merge_wrapped_lines",
            "merge_wrapped_lines",
            "Merge neighboring wrapped lines inside plain-text paragraphs.",
            _merge_wrapped_lines,
        )
    return directives


def _repair_target_kind(issue_type: str) -> str:
    if issue_type.startswith("table_"):
        return "table"
    if issue_type.startswith("structure_") or issue_type.startswith("wrapped_"):
        return "text"
    return "document"


def identify_repair_targets(
    source: DocumentSource,
    candidate: ParseCandidate,
    metrics: EvaluationMetrics,
) -> list[RepairTarget]:
    """inspect 단계의 최종 출력인 repair target 목록을 만든다.

    heuristic directive는 그대로 repair target이 되고, evaluation 단계에서
    잡힌 표 이슈는 항상 visual repair target으로 변환된다. route 노드는
    이 목록을 보고 heuristic 수리와 이미지 기반 수리를 나눈다.
    """
    targets = [
        RepairTarget(
            target_kind=_repair_target_kind(directive.issue_type),
            issue_type=directive.issue_type,
            route_name=directive.route_name,
            description=directive.description,
        )
        for directive in _classify_repair_directives(source, candidate, metrics)
    ]
    for issue_type in metrics.table_issues:
        targets.append(
            RepairTarget(
                target_kind="table",
                issue_type=issue_type,
                route_name="recover_tables_from_pdf_image",
                description=f"Recover broken table regions affected by {issue_type}.",
            )
        )
    return targets


class HeuristicRepairer(CandidateRepairer):
    def __init__(self, *, visual_table_recoverer=None) -> None:
        self._visual_table_recoverer = visual_table_recoverer

    def repair_heuristics(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
        targets: list[RepairTarget] | None = None,
    ) -> tuple[ParseCandidate, list[RepairAction]]:
        """route에서 허용한 heuristic transform만 적용한다.

        `targets`가 있으면 `(issue_type, route_name)` 기준으로 directive를
        필터링한다. 즉 모든 heuristic을 한 번에 적용하는 것이 아니라,
        route 노드가 고른 전략만 실행한다.
        """
        updated = candidate.content
        actions: list[RepairAction] = []
        directives = _classify_repair_directives(source, candidate, metrics)
        allowed_routes = None
        if targets is not None:
            allowed_routes = {(target.issue_type, target.route_name) for target in targets}
        for directive in directives:
            if allowed_routes is not None and (directive.issue_type, directive.route_name) not in allowed_routes:
                continue
            transformed = directive.transform(updated)
            if transformed == updated:
                continue
            actions.append(
                RepairAction(
                    action_name=directive.action_name,
                    description=directive.description,
                    before_excerpt=_excerpt(updated),
                    after_excerpt=_excerpt(transformed),
                    issue_type=directive.issue_type,
                    route_name=directive.route_name,
                )
            )
            updated = transformed

        if not actions:
            return candidate, []

        repaired_from = candidate.repaired_from or candidate.parser_name
        repaired_candidate = replace(
            candidate,
            content=updated,
            repaired_from=repaired_from,
            metadata={
                **candidate.metadata,
                "repair_actions": [action.action_name for action in actions],
                "repair_issue_types": [action.issue_type for action in actions if action.issue_type is not None],
                "repair_routes": [action.route_name for action in actions if action.route_name is not None],
            },
        )
        return repaired_candidate, actions

    def plan_chunk_repairs(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
        max_tasks: int,
    ) -> list[VisualRepairTask]:
        """어떤 표 영역을 visual repair 할지 task 목록만 계획한다.

        이 단계에서는 내용을 수정하지 않는다. visual recoverer가 없거나
        table issue가 없으면 빈 목록을 반환한다.
        """
        if self._visual_table_recoverer is None:
            return []
        if not metrics.table_issues:
            return []
        return self._visual_table_recoverer.plan_tasks(
            source,
            candidate.content,
            metrics,
            candidate_metadata=candidate.metadata,
            max_tasks=max_tasks,
        )[:max_tasks]

    def apply_chunk_repair(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        task: VisualRepairTask,
    ) -> tuple[ParseCandidate, RepairAction] | None:
        """계획된 visual table repair task 하나를 실제로 수행한다.

        visual recoverer가 빈 결과를 주거나 confidence가 `0.45` 미만이면
        버린다. 통과한 경우에만 candidate 안의 해당 표 블록을 교체한다.
        """
        if self._visual_table_recoverer is None:
            return None
        recovery = self._visual_table_recoverer.recover_task(source, candidate.content, task)
        if recovery is None or recovery.confidence < 0.45 or not recovery.markdown.strip():
            return None
        transformed = replace_table_block(
            candidate.content,
            task.table_label,
            recovery.markdown,
            candidate_metadata=candidate.metadata,
        )
        page_number = None
        if transformed == candidate.content and task.table_label.startswith("__page_table__:"):
            page_number, table_index = _page_table_selector_from_label(task.table_label)
            if page_number is not None:
                transformed = replace_page_table_block(
                    candidate.content,
                    page_number,
                    recovery.markdown,
                    table_index=table_index,
                )
        if transformed == candidate.content:
            return None
        note_suffix = ""
        if recovery.notes:
            note_suffix = f" Notes: {'; '.join(recovery.notes[:2])}"
        crop_suffix = ""
        if recovery.bbox is not None:
            crop_suffix = f" Crop: {recovery.crop_method} bbox={recovery.bbox}."
        action = RepairAction(
            action_name="recover_table_from_pdf_image",
            description=(
                f"Recover {recovery.table_label} from the source PDF page image and replace the broken parsed block."
                f"{crop_suffix}"
                f"{note_suffix}"
            ),
            before_excerpt=_excerpt(candidate.content),
            after_excerpt=_excerpt(transformed),
            issue_type="table_visual_recovery",
            route_name="recover_tables_from_pdf_image",
        )
        return (
            replace(
                candidate,
                content=transformed,
                repaired_from=candidate.repaired_from or candidate.parser_name,
                metadata={
                    **candidate.metadata,
                    "repair_chunk_table_label": task.table_label,
                    "repair_chunk_markdown": recovery.markdown,
                    "repair_chunk_issue_types": list(task.issue_types),
                    "repair_chunk_output_format": task.preferred_output_format,
                },
            ),
            action,
        )

    def repair(
        self,
        source: DocumentSource,
        candidate: ParseCandidate,
        metrics: EvaluationMetrics,
    ) -> tuple[ParseCandidate, list[RepairAction]]:
        """heuristic 수리와 visual repair를 함께 쓰는 범용 repair 진입점이다.

        현재 workflow는 주로 route 기반 repair를 사용하지만, 이 메서드는
        adapter 수준의 기본 경로로 남아 있다. 먼저 heuristic 수리를 하고,
        필요하면 visual table recoverer를 추가로 적용한다.
        """
        repaired_candidate, actions = self.repair_heuristics(source, candidate, metrics)
        current_candidate = repaired_candidate
        all_actions = list(actions)

        if self._visual_table_recoverer is not None:
            try:
                visual_content, visual_actions = self._visual_table_recoverer.repair(
                    source,
                    current_candidate.content,
                    metrics,
                    candidate_metadata=current_candidate.metadata,
                )
            except TypeError:
                visual_content, visual_actions = self._visual_table_recoverer.repair(
                    source,
                    current_candidate.content,
                    metrics,
                )
            if visual_actions:
                current_candidate = replace(
                    current_candidate,
                    content=visual_content,
                    repaired_from=current_candidate.repaired_from or current_candidate.parser_name,
                )
                all_actions.extend(visual_actions)

        if not all_actions:
            return candidate, []

        final_candidate = replace(
            current_candidate,
            metadata={
                **candidate.metadata,
                "repair_actions": [action.action_name for action in all_actions],
                "repair_issue_types": [action.issue_type for action in all_actions if action.issue_type is not None],
                "repair_routes": [action.route_name for action in all_actions if action.route_name is not None],
            },
        )
        return final_candidate, all_actions
