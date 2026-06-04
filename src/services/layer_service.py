from __future__ import annotations

from typing import Any

from src.domain import GraphCore, LayerConfig, RunContext
from src.graph_build import build_graph
from src.layers import (
    AttackSimulationLayer,
    CascadeLayer,
    CoreMetricsLayer,
    EdgeMetricsLayer,
    FlowLayer,
    LayerRegistry,
    LayerRunner,
    MLExportLayer,
    NodeMetricsLayer,
    RicciLayer,
    UrbanLayer,
    VulnerabilityLayer,
)
from src.state_models import GraphEntry


class LayerService:
    @staticmethod
    def default_registry() -> LayerRegistry:
        registry = LayerRegistry()
        registry.register(CoreMetricsLayer())
        registry.register(NodeMetricsLayer())
        registry.register(EdgeMetricsLayer())
        registry.register(AttackSimulationLayer())
        registry.register(FlowLayer())
        registry.register(CascadeLayer())
        registry.register(RicciLayer())
        registry.register(UrbanLayer())
        registry.register(MLExportLayer())
        registry.register(VulnerabilityLayer())
        return registry

    @staticmethod
    def build_core(
        entry: GraphEntry,
        *,
        min_conf: float,
        min_weight: float,
        analysis_mode: str,
    ) -> GraphCore:
        graph = build_graph(
            entry.edges,
            src_col=entry.src_col,
            dst_col=entry.dst_col,
            min_conf=float(min_conf),
            min_weight=float(min_weight),
            analysis_mode=str(analysis_mode),
        )
        return GraphCore(
            graph_id=entry.id,
            name=entry.name,
            nx_graph=graph,
            edges=entry.edges,
            source=entry.source,
            src_col=entry.src_col,
            dst_col=entry.dst_col,
            metadata={
                "analysis_mode": str(analysis_mode),
                "min_conf": float(min_conf),
                "min_weight": float(min_weight),
            },
            created_at=float(entry.created_at),
        )

    @staticmethod
    def default_config(
        *,
        core_metrics: bool = True,
        node_metrics: bool = True,
        edge_metrics: bool = True,
        attack_simulation: bool = False,
        flow: bool = False,
        cascade: bool = False,
        ricci: bool = False,
        urban: bool = False,
        ml_export: bool = False,
        vulnerability: bool = False,
        compute_heavy: bool = False,
        betweenness_samples: int = 100,
        edge_betweenness_max_edges_light: int = 500,
        attack_kind: str = "degree",
        attack_family: str = "node",
        attack_remove_frac: float = 0.2,
        attack_steps: int = 10,
        flow_mode: str = "rw",
        flow_steps: int = 25,
        cascade_threshold: float = 1.0,
        cascade_max_steps: int = 5,
        ricci_sample_edges: int = 80,
        urban_max_nodes: int = 250,
        vulnerability_max_exact_nodes: int = 1000,
        vulnerability_max_exact_edges: int = 2000,
        vulnerability_top_frac: float = 0.2,
    ) -> dict[str, LayerConfig]:
        return {
            "core_metrics": LayerConfig(
                enabled=bool(core_metrics),
                params={"compute_curvature": False},
                heavy=False,
            ),
            "node_metrics": LayerConfig(
                enabled=bool(node_metrics),
                params={"betweenness_samples": int(betweenness_samples)},
                heavy=bool(compute_heavy),
            ),
            "edge_metrics": LayerConfig(
                enabled=bool(edge_metrics),
                params={"edge_betweenness_max_edges_light": int(edge_betweenness_max_edges_light)},
                heavy=bool(compute_heavy),
            ),
            "attack_simulation": LayerConfig(
                enabled=bool(attack_simulation),
                params={
                    "attack_family": str(attack_family),
                    "attack_kind": str(attack_kind),
                    "remove_frac": float(attack_remove_frac),
                    "steps": int(attack_steps),
                    "eff_sources_k": 16,
                    "compute_heavy_every": 2,
                },
                heavy=bool(compute_heavy),
            ),
            "flow": LayerConfig(
                enabled=bool(flow),
                params={"flow_mode": str(flow_mode), "steps": int(flow_steps), "damping": 1.0},
                heavy=False,
            ),
            "cascade": LayerConfig(
                enabled=bool(cascade),
                params={
                    "threshold": float(cascade_threshold),
                    "max_steps": int(cascade_max_steps),
                    "flow_mode": str(flow_mode),
                    "flow_steps": min(int(flow_steps), 25),
                    "damping": 1.0,
                },
                heavy=False,
            ),
            "ricci": LayerConfig(
                enabled=bool(ricci),
                params={"sample_edges": int(ricci_sample_edges)},
                heavy=True,
            ),
            "urban": LayerConfig(
                enabled=bool(urban),
                params={"max_nodes": int(urban_max_nodes), "include_damage_dataset": True},
                heavy=False,
            ),
            "ml_export": LayerConfig(
                enabled=bool(ml_export),
                params={"max_nodes": int(urban_max_nodes), "critical_top_frac": float(vulnerability_top_frac)},
                heavy=False,
            ),
            "vulnerability": LayerConfig(
                enabled=bool(vulnerability),
                params={
                    "max_exact_nodes": int(vulnerability_max_exact_nodes),
                    "max_exact_edges": int(vulnerability_max_exact_edges),
                    "critical_top_frac": float(vulnerability_top_frac),
                    "include_edges": True,
                },
                heavy=bool(compute_heavy),
            ),
        }

    @staticmethod
    def run_layers(
        entry: GraphEntry,
        *,
        min_conf: float,
        min_weight: float,
        analysis_mode: str,
        seed: int,
        config: dict[str, LayerConfig] | None = None,
        compute_heavy: bool = False,
    ):
        core = LayerService.build_core(
            entry,
            min_conf=float(min_conf),
            min_weight=float(min_weight),
            analysis_mode=str(analysis_mode),
        )
        use_config = LayerService.default_config(compute_heavy=compute_heavy) if config is None else config
        context = RunContext(seed=int(seed), compute_heavy=bool(compute_heavy))
        runner = LayerRunner(LayerService.default_registry())
        return runner.run(core, config=use_config, context=context)

    @staticmethod
    def result_summary(augmented) -> list[dict[str, Any]]:
        rows = []
        for layer_id, result in augmented.layers.items():
            rows.append(
                {
                    "layer": layer_id,
                    "status": result.status,
                    "runtime_sec": round(float(result.runtime_sec), 4),
                    "warnings": "; ".join(result.warnings),
                    "node_columns": _column_count(result.node_attrs),
                    "edge_columns": _column_count(result.edge_attrs),
                    "metrics": len(result.graph_metrics),
                    "artifacts": len(result.artifacts),
                }
            )
        return rows


def _column_count(table) -> int:
    if table is None:
        return 0
    if getattr(table, "empty", True):
        return 0
    return int(len(table.columns))
