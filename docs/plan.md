# Parsing Agent System Plan

## Goal

Build an AI agent system that:
- runs one or more document parsers on source files,
- quantitatively evaluates parsed output against the original document,
- edits or normalizes parsed output based on evaluation findings,
- repeats the workflow when useful,
- emits the best available final parsing result with an evaluation report.

## Scope For This Iteration

This iteration will extend the working local framework with:
- PDF ingestion and source-text extraction,
- a real `opendataloader-pdf` parser adapter,
- an optional LLM judge that contributes a quantitative score,
- quality-gated candidate selection with deterministic thresholds,
- a simplified LangGraph-backed orchestrator with one parse/evaluate path plus a repair loop for weak results,
- an always-run final summary step that surfaces deterministic document attributes and lightweight stats,
- tests for PDF-capable workflow behavior.

## Assumptions

- Primary implementation language: Python.
- Parser and LLM providers must be swappable through adapters.
- This workspace currently has no existing codebase, so the system will be scaffolded from scratch.

## TODO / Unverified Choices

- TODO: Confirm whether `opendataloader-pdf` markdown output alone is sufficient, or whether its JSON/structured output should also be scored.
- TODO: Confirm the production LLM judge provider and model policy. Current implementation targets OpenAI-compatible chat endpoints.
- TODO: Confirm whether source evaluation should later include page images or OCR confidence in addition to extracted text.
- TODO: Confirm whether the repair layer should remain heuristic, or whether some edits should be delegated to an LLM rewrite step.

## Success Criteria

- A user can run a single CLI command against a text file or PDF input document.
- The system produces at least one parsed candidate.
- The system computes numeric quality metrics for each candidate.
- The system can optionally blend an LLM judge score into candidate ranking.
- The system selects and emits a best result plus an explanation report.
- The system can run a repair pass and show whether metrics improved after repair.
- Core domain logic is covered by automated tests.

## System Architecture

### 1. Ingestion

Responsibilities:
- accept input document paths,
- detect file types,
- load raw source metadata,
- prepare source artifacts for downstream evaluation.

Initial implementation:
- local file ingestion for PDF and text-like files,
- metadata model with path, type, size, run identifiers, extracted source text, and page count,
- `pypdf`-based source text extraction for PDF scoring.

### 2. Parser Layer

Responsibilities:
- expose a common parser interface,
- run one or more parsing providers,
- return normalized candidate objects.

Initial implementation:
- `BaseParserAdapter` interface,
- `MockParserAdapter` for tests,
- simple local parser for text-like inputs,
- `opendataloader-pdf` adapter for real PDF parsing.

### 3. Evaluation Layer

Responsibilities:
- compare original source and parsed candidate,
- generate quantitative metrics,
- support both deterministic checks and LLM-based judgment.

Initial metrics:
- text coverage ratio,
- normalized similarity score,
- line/section retention score,
- table marker preservation heuristic,
- penalty counts for empty blocks, repeated blocks, or malformed structure.

Initial LLM strategy:
- deterministic evaluator remains the baseline,
- optional OpenAI-compatible LLM judge adds a bounded score contribution,
- judge activation is configuration-driven so local deterministic runs still work without API keys.

### 4. Repair / Editing Layer

Responsibilities:
- transform weak parse outputs into improved candidates,
- apply repair heuristics based on evaluation findings,
- keep a trace of edits performed.

Initial repair strategies:
- whitespace normalization,
- broken line merge,
- repeated line removal,
- repeated header/footer cleanup,
- simple heading normalization,
- table-text cleanup heuristics for markdown-like output,
- inline image stripping from table cells,
- conservative reconstruction of repeated key/value text rows into markdown tables when table structure was flattened during parsing,
- targeted visual table recovery for PDF inputs by detecting a table bounding box with the configured lightweight table detector, rendering only that crop, and asking a vision-capable model to reconstruct the judge-identified broken table.

Interim layout strategy:
- Keep the existing parse -> verify -> evaluate -> repair loop intact.
- In `visual_repair`, prefer precise table bounding boxes from the configured detector (`pymupdf` by default) before falling back to label-window or full-page crops.
- Record the crop method and bounding box in repair action descriptions so LangSmith/report review can compare crop precision, latency, and retry behavior.

Target layout-first architecture:
- Add a future pre-parse layout analysis stage that renders each page, detects text/table/image/chart regions, routes each region to the right extractor, and merges outputs by page coordinates.
- Compare this future layout-first path against the current whole-document parser plus targeted visual repair path using latency, VLM call count, retry count, and table fidelity metrics.

Implemented layout-first v1:
- `layout-first-pdf` is now a parser candidate and runs before `opendataloader-pdf` by default.
- It uses PyMuPDF page layout primitives: text blocks from `page.get_text("blocks")`, table boxes from `page.find_tables()`, and coordinate-ordered merging.
- Text blocks overlapping detected table boxes are skipped to avoid duplicate table text.
- Detected tables are converted into markdown tables locally; image/chart blocks are preserved as placeholders for future VLM captioning.
- This v1 does not yet call a VLM for every table or image block. It establishes the routing and merge path needed for that next step while keeping existing parser candidates available for comparison.

### 5. Orchestrator

Responsibilities:
- run parse -> verify -> evaluate -> repair -> evaluate,
- compare candidates across providers and passes,
- select the best final result,
- summarize the selected result for the user,
- produce audit-friendly artifacts.

Selection strategy:
- prefer candidates that satisfy configured quality gates, then sort by weighted total score,
- if no candidate passes the gate, keep the highest-scoring fallback and record the gate failure,
- keep per-step reports,
- keep looping repair -> evaluate until a candidate passes the quality gate or the configured repair budget is exhausted,
- execute orchestration through a LangGraph state graph while preserving the existing runner API.
- after final result selection, always run a distinct summarize step that records file name, media type, page count when known, a deterministic content overview, and simple output stats.
- keep fallback parser configuration as a compatibility/reporting field only; this simplified iteration does not validate or execute a fallback-parser runtime branch.

### 6. Output Layer

Responsibilities:
- write final parsed output,
- write machine-readable report,
- write human-readable summary.

Initial outputs:
- final text or markdown file,
- JSON report,
- run summary printed to CLI, including document attributes and lightweight stats from the selected output.

## Data Model

Core entities:
- `DocumentSource`
- `ParseCandidate`
- `EvaluationMetrics`
- `RepairAction`
- `WorkflowResult`

## Implementation Tasks

### Task 1. Project Scaffold And Core Models

Deliverables:
- Python project structure,
- dependency config,
- core dataclasses/models,
- config loader,
- test harness.

Acceptance criteria:
- tests run locally,
- shared models can be imported across modules,
- CLI skeleton executes.

### Task 2. Parser Adapters

Deliverables:
- parser adapter base interface,
- mock parser,
- simple local parser path,
- parser registry.

Acceptance criteria:
- orchestrator can request parser runs through a common interface,
- tests cover adapter selection and candidate generation.

### Task 3. Evaluation Engine

Deliverables:
- deterministic metric calculators,
- score aggregation,
- evaluator report model,
- optional LLM judge interface stub.

Acceptance criteria:
- evaluator returns numeric metrics and weighted score,
- tests show better parses score higher than worse parses.

### Task 4. Repair Engine

Deliverables:
- repair pipeline,
- heuristic transforms,
- repair trace model.

Acceptance criteria:
- repair pass can transform a weak candidate,
- tests show repaired output improves at least one metric in a representative case.

### Task 5. Workflow Orchestrator And CLI

Deliverables:
- end-to-end workflow runner,
- parse/evaluate/repair loop,
- result selection,
- output writer,
- CLI command.

Acceptance criteria:
- one command executes the whole workflow on a local sample,
- output artifacts are produced,
- tests cover no-improvement and improvement paths.

## Execution Order

1. Build scaffold and models.
2. Build parser layer.
3. Build evaluator.
4. Build repair pipeline.
5. Integrate orchestrator and CLI.
6. Run verification and refine docs.

## Notes On Subagent Execution

Independent implementation slices for subagents:
- Scaffold and shared models
- Parser layer
- Evaluation layer
- Repair plus orchestration layer

Review sequence after each substantial slice:
- spec compliance review,
- code quality review.
