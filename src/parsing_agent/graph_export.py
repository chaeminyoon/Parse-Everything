from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.runnables.graph_mermaid import draw_mermaid_png

from parsing_agent.config import WorkflowConfig
from parsing_agent.workflow import WorkflowRunner


def _build_langgraph():
    runner = WorkflowRunner(WorkflowConfig(judge_weight=0, langsmith_tracing=False))
    return runner.get_graph()

def build_workflow_graph_mermaid() -> str:
    return _build_langgraph().draw_mermaid()


def export_workflow_graph(output_path: Path, output_format: str | None = None) -> Path:
    resolved_format = (output_format or output_path.suffix.lstrip(".") or "mermaid").lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = _build_langgraph()
    mermaid = graph.draw_mermaid()

    if resolved_format in {"mmd", "mermaid", "md"}:
        output_path.write_text(mermaid, encoding="utf-8")
        return output_path
    if resolved_format == "png":
        draw_mermaid_png(mermaid, output_file_path=str(output_path))
        return output_path
    raise ValueError(f"Unsupported graph output format: {resolved_format}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the parsing workflow LangGraph diagram.")
    parser.add_argument(
        "output_path",
        nargs="?",
        default="outputs/workflow-graph.mmd",
        help="Output path for the graph. Supported suffixes: .mmd, .md, .png.",
    )
    parser.add_argument(
        "--format",
        choices=["mermaid", "mmd", "md", "png"],
        help="Optional output format override.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written_path = export_workflow_graph(Path(args.output_path), output_format=args.format)
    print(f"Graph: {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
