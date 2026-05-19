from pathlib import Path

from parsing_agent.graph_export import build_workflow_graph_mermaid, export_workflow_graph


def test_build_workflow_graph_mermaid_uses_langgraph_nodes() -> None:
    mermaid = build_workflow_graph_mermaid()

    assert "parse" in mermaid
    assert "evaluate" in mermaid
    assert "inspect" in mermaid
    assert "route" in mermaid
    assert "repair" in mermaid
    assert "finalize" in mermaid
    assert "__start__ --> parse" in mermaid
    assert "parse --> evaluate" in mermaid
    assert "evaluate -.-> inspect" in mermaid
    assert "evaluate -.-> finalize" in mermaid
    assert "inspect -.-> route" in mermaid
    assert "route -.-> repair" in mermaid
    assert "repair --> evaluate" in mermaid


def test_export_workflow_graph_writes_mermaid_file(tmp_path: Path) -> None:
    output_path = tmp_path / "workflow-graph.mmd"

    written_path = export_workflow_graph(output_path, output_format="mermaid")

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8").startswith("---")
