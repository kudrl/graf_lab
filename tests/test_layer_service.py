from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pandas as pd

from src.services.layer_service import LayerService
from src.services.urban_resilience import city_graph_to_edges, create_city_preset
from src.state_models import build_graph_entry
from src.ui.tabs.layers import _build_config, _state_key


def make_entry():
    edges = pd.DataFrame(
        [
            {"src": "a", "dst": "b", "weight": 2.0, "confidence": 90.0},
            {"src": "b", "dst": "c", "weight": 1.0, "confidence": 80.0},
            {"src": "x", "dst": "y", "weight": 1.0, "confidence": 10.0},
        ]
    )
    return build_graph_entry(
        name="Layer service graph",
        source="test",
        edges=edges,
        src_col="src",
        dst_col="dst",
        entry_id="G_test",
        created_at=1.0,
    )


def test_layer_service_build_core_uses_filters_and_metadata() -> None:
    core = LayerService.build_core(
        make_entry(),
        min_conf=50.0,
        min_weight=0.0,
        analysis_mode="Global",
    )

    assert core.graph_id == "G_test"
    assert core.nx_graph.number_of_nodes() == 3
    assert core.nx_graph.number_of_edges() == 2
    assert core.metadata["min_conf"] == 50.0
    assert core.metadata["analysis_mode"] == "Global"


def test_layer_service_default_config_controls_enabled_layers() -> None:
    config = LayerService.default_config(
        core_metrics=True,
        node_metrics=False,
        edge_metrics=True,
        compute_heavy=True,
        betweenness_samples=7,
        edge_betweenness_max_edges_light=11,
    )

    assert config["core_metrics"].enabled is True
    assert config["node_metrics"].enabled is False
    assert config["edge_metrics"].heavy is True
    assert config["node_metrics"].params["betweenness_samples"] == 7
    assert config["edge_metrics"].params["edge_betweenness_max_edges_light"] == 11
    assert config["attack_simulation"].enabled is False
    assert config["flow"].enabled is False
    assert config["ricci"].enabled is False
    assert config["urban"].enabled is False
    assert config["ml_export"].enabled is False


def test_layer_service_registry_contains_all_mvp_wrappers() -> None:
    layer_ids = {layer.id for layer in LayerService.default_registry().list_available()}

    assert layer_ids == {
        "attack_simulation",
        "cascade",
        "core_metrics",
        "edge_metrics",
        "flow",
        "ml_export",
        "node_metrics",
        "ricci",
        "urban",
        "vulnerability",
    }


def test_layer_service_run_layers_returns_augmented_graph() -> None:
    augmented = LayerService.run_layers(
        make_entry(),
        min_conf=50.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=LayerService.default_config(
            core_metrics=True,
            node_metrics=True,
            edge_metrics=True,
            betweenness_samples=2,
            edge_betweenness_max_edges_light=10,
        ),
    )

    assert set(augmented.layers) == {"core_metrics", "node_metrics", "edge_metrics"}
    assert augmented.graph_metrics["N"] == 3
    assert set(augmented.node_attributes["node"]) == {"a", "b", "c"}
    assert len(augmented.edge_attributes) == 2


def test_layer_service_run_disabled_new_layers_are_not_executed() -> None:
    augmented = LayerService.run_layers(
        make_entry(),
        min_conf=50.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=LayerService.default_config(
            core_metrics=True,
            node_metrics=False,
            edge_metrics=False,
            attack_simulation=False,
            flow=False,
            cascade=False,
            ricci=False,
            urban=False,
            ml_export=False,
            vulnerability=False,
        ),
    )

    assert set(augmented.layers) == {"core_metrics"}


def test_attack_flow_and_ricci_layers_return_expected_shapes() -> None:
    augmented = LayerService.run_layers(
        make_entry(),
        min_conf=50.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=LayerService.default_config(
            core_metrics=False,
            node_metrics=False,
            edge_metrics=False,
            attack_simulation=True,
            flow=True,
            ricci=True,
            compute_heavy=True,
            attack_steps=3,
            flow_steps=3,
            ricci_sample_edges=2,
        ),
        compute_heavy=True,
    )

    assert augmented.layers["attack_simulation"].status == "success"
    assert "removed_nodes" in augmented.layers["attack_simulation"].artifacts
    assert not augmented.temporal_states.empty
    assert "flow_final" in augmented.node_attributes.columns
    assert "flow_overload_risk" in augmented.node_attributes.columns
    assert "flow_flux_final" in augmented.edge_attributes.columns
    assert "flow_edge_overload_risk" in augmented.edge_attributes.columns
    assert augmented.layers["ricci"].status == "success"
    assert "kappa_mean" in augmented.graph_metrics


def test_attack_simulation_layer_supports_edge_family() -> None:
    augmented = LayerService.run_layers(
        make_entry(),
        min_conf=50.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=LayerService.default_config(
            core_metrics=False,
            node_metrics=False,
            edge_metrics=False,
            attack_simulation=True,
            attack_family="edge",
            attack_kind="weak_edges_by_weight",
            attack_steps=3,
        ),
    )

    result = augmented.layers["attack_simulation"]
    assert result.status == "success"
    assert result.artifacts["attack_family"] == "edge"
    assert result.artifacts["attack_kind"] == "weak_edges_by_weight"
    assert "removed_edges_order" in result.artifacts
    assert "attack_family" in augmented.temporal_states.columns


def test_layer_service_cascade_returns_states_and_metrics() -> None:
    augmented = LayerService.run_layers(
        make_entry(),
        min_conf=50.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=LayerService.default_config(
            core_metrics=False,
            node_metrics=False,
            edge_metrics=False,
            flow=True,
            cascade=True,
            flow_steps=3,
            cascade_threshold=0.01,
            cascade_max_steps=3,
        ),
    )

    assert {"flow", "cascade"}.issubset(augmented.layers)
    assert "cascade_size" in augmented.graph_metrics
    assert "cascade_final_lcc_frac" in augmented.graph_metrics
    assert "cascade_step" in augmented.temporal_states.columns


def test_layer_service_vulnerability_adds_damage_score() -> None:
    augmented = LayerService.run_layers(
        make_entry(),
        min_conf=50.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=LayerService.default_config(
            core_metrics=False,
            node_metrics=True,
            edge_metrics=True,
            vulnerability=True,
            betweenness_samples=2,
            vulnerability_top_frac=0.5,
        ),
    )

    assert "vulnerability" in augmented.layers
    assert "damage_score" in augmented.node_attributes.columns
    assert "criticality_rank" in augmented.node_attributes.columns
    assert "edge_damage_score" in augmented.edge_attributes.columns


def test_urban_layer_skips_regular_graph_and_succeeds_on_city_graph() -> None:
    regular = LayerService.run_layers(
        make_entry(),
        min_conf=0.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=LayerService.default_config(
            core_metrics=False,
            node_metrics=False,
            edge_metrics=False,
            urban=True,
        ),
    )
    assert regular.layers["urban"].status == "skipped"

    city_graph = create_city_preset("Compact city", seed=1)
    city_edges = city_graph_to_edges(city_graph)
    city_entry = build_graph_entry(
        name="City",
        source="urban_resilience:preset",
        edges=city_edges,
        src_col="src",
        dst_col="dst",
        entry_id="G_city",
        created_at=1.0,
    )
    city = LayerService.run_layers(
        city_entry,
        min_conf=0.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=LayerService.default_config(
            core_metrics=False,
            node_metrics=False,
            edge_metrics=False,
            urban=True,
            urban_max_nodes=20,
        ),
    )

    assert city.layers["urban"].status == "success"
    assert "city_damage_dataset_csv" in city.layers["urban"].artifacts
    assert "damage_score" in city.node_attributes.columns


def test_ml_export_layer_generic_and_urban_artifacts() -> None:
    generic = LayerService.run_layers(
        make_entry(),
        min_conf=50.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=LayerService.default_config(
            core_metrics=False,
            node_metrics=True,
            edge_metrics=True,
            ml_export=True,
            vulnerability=False,
            betweenness_samples=2,
        ),
    )
    assert generic.layers["ml_export"].status == "skipped"
    assert "damage_score is required" in generic.layers["ml_export"].warnings[0]

    generic = LayerService.run_layers(
        make_entry(),
        min_conf=50.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=LayerService.default_config(
            core_metrics=False,
            node_metrics=True,
            edge_metrics=True,
            vulnerability=True,
            ml_export=True,
            betweenness_samples=2,
            vulnerability_top_frac=0.5,
        ),
    )
    artifacts = generic.layers["ml_export"].artifacts
    assert generic.layers["ml_export"].status == "success"
    assert {
        "ml_handoff_zip",
        "node_dataset_csv",
        "features_nodes_csv",
        "labels_nodes_csv",
        "edge_attributes_csv",
        "metadata_json",
        "readme_md",
    }.issubset(artifacts)
    with zipfile.ZipFile(BytesIO(artifacts["ml_handoff_zip"])) as archive:
        assert {
            "node_dataset.csv",
            "features_nodes.csv",
            "labels_nodes.csv",
            "edge_attributes.csv",
            "metadata.json",
            "README.md",
        }.issubset(archive.namelist())
        node_dataset = pd.read_csv(archive.open("node_dataset.csv"))
        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
    expected_columns = {
        "graph_id",
        "node",
        "graph_family",
        "graph_n_nodes",
        "graph_n_edges",
        "degree",
        "degree_norm",
        "strength",
        "strength_norm",
        "betweenness",
        "closeness",
        "clustering",
        "pagerank",
        "eigenvector",
        "core_number",
        "core_number_norm",
        "local_density",
        "energy_final",
        "energy_peak_pressure",
        "energy_cumulative_inflow",
        "energy_overload_risk",
        "damage_score",
        "critical",
    }
    assert expected_columns.issubset(node_dataset.columns)
    assert node_dataset["damage_score"].ge(0).all()
    assert set(node_dataset["critical"].unique()).issubset({0, 1})
    assert node_dataset[[
        "energy_final",
        "energy_peak_pressure",
        "energy_cumulative_inflow",
        "energy_overload_risk",
    ]].eq(0.0).all().all()
    assert metadata["target_repository"] == "graph-vulnerability-gnn"
    assert metadata["flow_features_source"] == "missing_filled_zero"
    assert metadata["label_definition"].startswith("damage_score = LCC_fraction_before")

    city_graph = create_city_preset("Compact city", seed=2)
    city_entry = build_graph_entry(
        name="City",
        source="urban_resilience:preset",
        edges=city_graph_to_edges(city_graph),
        src_col="src",
        dst_col="dst",
        entry_id="G_city",
        created_at=1.0,
    )
    city = LayerService.run_layers(
        city_entry,
        min_conf=0.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=2,
        config=LayerService.default_config(
            core_metrics=False,
            node_metrics=False,
            edge_metrics=False,
            ml_export=True,
            urban_max_nodes=20,
        ),
    )
    blob = city.layers["ml_export"].artifacts["urban_ml_handoff_zip"]
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        assert "city_damage_dataset.csv" in archive.namelist()
        assert "ml_manifest.json" in archive.namelist()


def test_layers_ui_state_key_changes_with_filters_seed_and_config() -> None:
    config = _build_config(
        core_enabled=True,
        node_enabled=True,
        edge_enabled=True,
        attack_enabled=False,
        flow_enabled=False,
        cascade_enabled=False,
        vulnerability_enabled=False,
        ricci_enabled=False,
        urban_enabled=False,
        ml_export_enabled=False,
        compute_heavy=False,
        betweenness_samples=10,
        edge_light_limit=100,
        attack_kind="degree",
        attack_family="node",
        attack_steps=3,
        flow_mode="rw",
        flow_steps=3,
        cascade_threshold=1.0,
        cascade_max_steps=3,
        vulnerability_top_frac=0.2,
    )
    base = _state_key(
        "G",
        min_conf=0.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=config,
    )
    assert base != _state_key(
        "G",
        min_conf=10.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=config,
    )
    assert base != _state_key(
        "G",
        min_conf=0.0,
        min_weight=0.0,
        analysis_mode="LCC",
        seed=1,
        config=config,
    )
    config["flow"].enabled = True
    assert base != _state_key(
        "G",
        min_conf=0.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=config,
    )
    config["cascade"].enabled = True
    assert base != _state_key(
        "G",
        min_conf=0.0,
        min_weight=0.0,
        analysis_mode="Global",
        seed=1,
        config=config,
    )
