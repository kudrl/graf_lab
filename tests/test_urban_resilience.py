from __future__ import annotations

import zipfile
from io import BytesIO

from src.services.urban_resilience import (
    FailurePlan,
    add_city_entity,
    apply_city_entity_edits,
    build_failure_plan,
    build_ml_handoff_bundle,
    city_damage_dataset,
    city_edges_frame,
    city_graph_from_edges,
    city_graph_to_edges,
    city_nodes_frame,
    compute_node_interactions,
    compute_node_potentials,
    create_city_preset,
    format_impact_report,
    generate_city_graph,
    generate_random_shape_city_graph,
    has_city_schema,
    recommend_intervention,
    simulate_failure_impact,
)
from src.ui.plots.scene3d import make_city_3d_figure


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


def test_random_shape_city_roundtrip_preserves_elevation() -> None:
    graph = generate_random_shape_city_graph(intersections=18, homes=8, seed=21)
    edges = city_graph_to_edges(graph)
    restored = city_graph_from_edges(edges)

    assert graph.graph["shape"] == "random"
    assert {"src_elevation", "dst_elevation"}.issubset(edges.columns)
    assert restored.number_of_nodes() == graph.number_of_nodes()
    assert any("elevation" in data for _, data in restored.nodes(data=True))


def test_flood_level_drives_impact_and_3d_layers() -> None:
    graph = generate_random_shape_city_graph(intersections=16, homes=8, seed=22)
    elevations = [float(data["elevation"]) for _, data in graph.nodes(data=True)]
    water_level = sorted(elevations)[len(elevations) // 2]
    plan = FailurePlan("unused")
    flood_plan = build_failure_plan(graph, "Flood by water level", water_level=water_level)
    impact = simulate_failure_impact(graph, flood_plan)

    assert plan.water_level is None
    assert flood_plan.water_level == water_level
    assert impact["flood"]["flooded_nodes"] == len(flood_plan.flooded_nodes)
    assert impact["flood"]["flooded_population"] >= 0

    fig = make_city_3d_figure(
        graph,
        z_attr="elevation",
        water_level=flood_plan.water_level,
        flooded_nodes=list(flood_plan.flooded_nodes),
        flooded_edges=list(flood_plan.flooded_edges),
    )
    trace_names = {str(trace.name) for trace in fig.data}
    assert "water level" in trace_names
    assert "flooded nodes" in trace_names


def test_node_potentials_are_normalized_and_non_empty() -> None:
    graph = create_city_preset("Компактный город", seed=14)

    frame = compute_node_potentials(graph)

    expected = {
        "node",
        "node_type",
        "access_potential",
        "connectivity_potential",
        "vulnerability_potential",
        "service_potential",
        "evacuation_potential",
    }
    assert expected.issubset(frame.columns)
    assert len(frame) == graph.number_of_nodes()
    for column in expected - {"node", "node_type"}:
        assert frame[column].between(0.0, 1.0).all()
    assert frame["access_potential"].max() > 0.0
    assert frame["evacuation_potential"].max() > 0.0


def test_node_interactions_are_unique_and_sorted() -> None:
    graph = create_city_preset("Пригород с рекой", seed=15)

    frame = compute_node_interactions(graph)

    assert not frame.empty
    assert {"source", "target", "interaction_type", "dependency_score", "redundancy"}.issubset(frame.columns)

    pairs = {frozenset((str(row.source), str(row.target))) for row in frame.itertuples()}
    assert len(pairs) == len(frame)
    assert frame["dependency_score"].is_monotonic_decreasing
    assert (frame["redundancy"] >= 0).all()
