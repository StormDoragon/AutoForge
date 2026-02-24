from __future__ import annotations

from langgraph.graph import END, StateGraph

from autoforge.agents.nodes import (
    load_data_node,
    optimize_node,
    registry_export_node,
    train_eval_mlflow_node,
)
from autoforge.types import AutoForgeState


def build_graph():
    graph = StateGraph(AutoForgeState)
    graph.add_node("load_data", load_data_node)
    graph.add_node("optimize", optimize_node)
    graph.add_node("train_eval", train_eval_mlflow_node)
    graph.add_node("registry_export", registry_export_node)

    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "optimize")
    graph.add_edge("optimize", "train_eval")
    graph.add_edge("train_eval", "registry_export")
    graph.add_edge("registry_export", END)

    return graph.compile()


def run_pipeline(initial_state: AutoForgeState) -> AutoForgeState:
    app = build_graph()
    return app.invoke(initial_state)
