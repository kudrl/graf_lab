from __future__ import annotations

import pandas as pd

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext
from src.services.urban_resilience import (
    city_damage_dataset,
    city_graph_from_edges,
    city_status,
    compute_node_interactions,
    compute_node_potentials,
    has_city_schema,
)

from .base import BaseLayer


class UrbanLayer(BaseLayer):
    id = "urban"
    name = "Urban"
    description = "Urban resilience status and damage labels."
    default_config = LayerConfig(
        enabled=False,
        params={
            "max_nodes": 250,
            "include_damage_dataset": True,
            "include_potentials": True,
            "include_interactions": True,
        },
        heavy=False,
    )

    def compute(
        self,
        core: GraphCore,
        augmented: AugmentedGraph,
        config: LayerConfig,
        context: RunContext,
    ) -> LayerResult:
        if not has_city_schema(core.edges):
            return LayerResult(
                layer_id=self.id,
                status="skipped",
                warnings=["Urban skipped: graph does not have city schema"],
            )

        city_graph = city_graph_from_edges(core.edges)

        metrics = {f"urban_{key}": value for key, value in city_status(city_graph).items()}
        artifacts = {}
        node_frames: list[pd.DataFrame] = []
        pairwise_attrs = pd.DataFrame()
        if bool(config.params.get("include_damage_dataset", True)):
            dataset = city_damage_dataset(
                city_graph,
                max_nodes=int(config.params.get("max_nodes", 250)),
            )
            artifacts["city_damage_dataset_csv"] = dataset.to_csv(index=False).encode("utf-8")
            keep_cols = [
                col
                for col in [
                    "node",
                    "node_type",
                    "damage_score",
                    "critical",
                    "severity",
                    "hospital_people_without_access",
                    "shelter_people_without_access",
                    "power_people_without_access",
                ]
                if col in dataset.columns
            ]
            if keep_cols:
                node_frames.append(dataset[keep_cols].copy())

        if bool(config.params.get("include_potentials", True)):
            potentials = compute_node_potentials(city_graph)
            if not potentials.empty:
                artifacts["urban_node_potentials_csv"] = potentials.to_csv(index=False).encode("utf-8")
                if node_frames and "node_type" in potentials.columns:
                    potentials = potentials.drop(columns=["node_type"])
                node_frames.append(potentials.copy())

        if bool(config.params.get("include_interactions", True)):
            interactions = compute_node_interactions(city_graph)
            if not interactions.empty:
                pairwise_attrs = interactions.copy()
                artifacts["urban_node_interactions_csv"] = interactions.to_csv(index=False).encode("utf-8")

        return LayerResult(
            layer_id=self.id,
            status="success",
            node_attrs=_merge_node_frames(node_frames),
            pairwise_attrs=pairwise_attrs,
            graph_metrics=metrics,
            artifacts=artifacts,
            provenance={"source": "src.services.urban_resilience"},
        )


def _merge_node_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    result = pd.DataFrame()
    for frame in frames:
        if frame.empty:
            continue
        if result.empty:
            result = frame.copy()
        else:
            result = result.merge(frame.copy(), on="node", how="outer")
    return result
