from __future__ import annotations

import zipfile
from io import BytesIO

from src.services.urban_resilience import (
    FailurePlan,
    add_city_entity,
    apply_city_entity_edits,
    build_ml_handoff_bundle,
    city_damage_dataset,
    city_edges_frame,
    city_graph_from_edges,
    city_graph_to_edges,
    city_nodes_frame,
    create_city_preset,
    format_impact_report,
    generate_city_graph,
    has_city_schema,
    recommend_intervention,
    simulate_failure_impact,
)


def test_city_graph_roundtrip_preserves_types() -> None:
    graph = generate_city_graph(grid_size=3, homes=5, seed=7)
    edges = city_graph_to_edges(graph)
    restored = city_graph_from_edges(edges)

    assert has_city_schema(edges)
    assert restored.number_of_edges() == graph.number_of_edges()
    assert "home" in {data.get("type") for _, data in restored.nodes(data=True)}
    assert "bridge" in {data.get("edge_type") for _, _, data in restored.edges(data=True)}


def test_failure_impact_reports_access_loss() -> None:
    graph = generate_city_graph(grid_size=3, homes=4, seed=2)
    hospital = next(node for node, data in graph.nodes(data=True) if data.get("type") == "hospital")
    impact = simulate_failure_impact(graph, FailurePlan("hospital outage", (hospital,)))
    report = format_impact_report(impact)

    assert impact["after"]["hospital_people_without_access"] > 0
    assert "Ущерб от отказа" in report
    assert "доступа к больнице" in report


def test_damage_dataset_and_intervention_are_non_empty() -> None:
    graph = generate_city_graph(grid_size=3, homes=4, seed=3)
    bridge = next((u, v) for u, v, d in graph.edges(data=True) if d.get("edge_type") == "bridge")
    impact = simulate_failure_impact(graph, FailurePlan("bridge", removed_edges=(bridge,)))
    dataset = city_damage_dataset(graph)
    intervention = recommend_intervention(graph, impact)

    assert not dataset.empty
    assert "damage_score" in dataset.columns
    assert "critical" in dataset.columns
    assert "graph_id" in dataset.columns
    assert "strength_norm" in dataset.columns
    assert intervention["action"]


def test_city_preset_can_be_edited_and_extended() -> None:
    graph = create_city_preset("Компактный город", seed=10)
    nodes = city_nodes_frame(graph)
    edges = city_edges_frame(graph)

    home_idx = nodes.index[nodes["type"] == "home"][0]
    nodes.loc[home_idx, "population"] = 25
    edge_idx = edges.index[edges["edge_type"].isin(["road", "bridge"])][0]
    edges.loc[edge_idx, "edge_type"] = "bridge"
    edited = apply_city_entity_edits(nodes, edges)
    added = add_city_entity(
        edited,
        node_id="H_extra",
        node_type="home",
        connect_to=next(iter(edited.nodes())),
        population=4,
    )

    assert edited.nodes[nodes.loc[home_idx, "node"]]["population"] == 25
    assert any(data.get("edge_type") == "bridge" for _, _, data in edited.edges(data=True))
    assert "H_extra" in added
    assert added.degree("H_extra") == 1


def test_ml_handoff_bundle_contains_transfer_materials() -> None:
    graph = create_city_preset("Компактный город", seed=11)
    bundle = build_ml_handoff_bundle(graph, graph_name="Город: Компактный город")

    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        assert "city_damage_dataset.csv" in names
        assert "city_graph_edges.csv" in names
        assert "city_nodes.csv" in names
        assert "city_roads.csv" in names
        assert "ml_manifest.json" in names
        assert "README.md" in names
