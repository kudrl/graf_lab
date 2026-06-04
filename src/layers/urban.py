from __future__ import annotations

import pandas as pd

from src.domain import AugmentedGraph, GraphCore, LayerConfig, LayerResult, RunContext
from src.services.urban_resilience import city_damage_dataset, city_status, has_city_schema

from .base import BaseLayer


class UrbanLayer(BaseLayer):
    id = "urban"
    name = "Urban"
    description = "Urban resilience status and damage labels."
    default_config = LayerConfig(
        enabled=False,
        params={"max_nodes": 250, "include_damage_dataset": True},
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

        metrics = {f"urban_{key}": value for key, value in city_status(core.nx_graph).items()}
        artifacts = {}
        node_attrs = pd.DataFrame()
        if bool(config.params.get("include_damage_dataset", True)):
            dataset = city_damage_dataset(
                core.nx_graph,
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
            node_attrs = dataset[keep_cols].copy() if keep_cols else pd.DataFrame()

        return LayerResult(
            layer_id=self.id,
            status="success",
            node_attrs=node_attrs,
            graph_metrics=metrics,
            artifacts=artifacts,
            provenance={"source": "src.services.urban_resilience"},
        )
