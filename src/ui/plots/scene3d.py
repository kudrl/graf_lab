from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import plotly.colors
import plotly.graph_objects as go

from ...core.physics import compute_energy_flow, simulate_energy_flow
from ...core_math import ollivier_ricci_edge


def make_energy_flow_figure_3d(
    G: nx.Graph,
    pos3d: dict,
    *,
    steps: int = 25,
    node_frames: Optional[List[Dict]] = None,
    edge_frames: Optional[List[Dict[Tuple, float]]] = None,
    flow_mode: str = "phys",
    damping: float = 1.0,
    sources: Optional[List] = None,
    phys_injection: float = 0.15,
    phys_leak: float = 0.02,
    phys_cap_mode: str = "strength",
    edge_bins: int = 7,
    hotspot_q: float = 0.92,
    hotspot_size_mult: float = 4.0,
    base_node_opacity: float = 0.25,
    rw_impulse: bool = True,
    max_nodes_viz: int = 6000,
    node_subset_mode: str = "top_degree",
    seed: int = 0,
    node_size: float = 6.0,
    node_base_size: float | None = None,
    vis_contrast: float = 1.0,
    vis_clip: float = 0.0,
    edge_subset_mode: str = "all",
    max_edges_viz: int = 1500,
    anim_duration: int = 80,
) -> go.Figure:
    """Render an animated 3D energy flow figure.
    """
    if node_frames is None or edge_frames is None:
        node_frames, edge_frames = simulate_energy_flow(
            G,
            steps=steps,
            flow_mode=flow_mode,
            damping=damping,
            sources=sources,
            phys_injection=phys_injection,
            phys_leak=phys_leak,
            phys_cap_mode=phys_cap_mode,
            rw_impulse=rw_impulse,
        )

    nodes = list(G.nodes())
    if not nodes:
        return go.Figure()

    # Limit nodes for browser performance (huge marker arrays kill Plotly 3D).
    max_nodes_viz = int(max_nodes_viz)
    node_subset_mode = str(node_subset_mode or "top_degree").lower()
    if max_nodes_viz > 0 and len(nodes) > max_nodes_viz:
        if node_subset_mode in ("top_degree", "top_strength"):
            degs = [(n, G.degree(n)) for n in nodes]
            degs.sort(key=lambda t: t[1], reverse=True)
            nodes = [n for n, _ in degs[:max_nodes_viz]]
        else:
            rng = np.random.default_rng(int(seed))
            nodes = rng.choice(np.asarray(nodes, dtype=object), size=max_nodes_viz, replace=False).tolist()

    steps = min(int(steps), len(node_frames) - 1)

    # ВАЖНО: Определяем глобальный максимум энергии для корректной нормировки цвета
    Emax = 0.0
    for fr in node_frames[: steps + 1]:
        if fr:
            vals = [v for v in fr.values() if np.isfinite(v)]
            if vals:
                Emax = max(Emax, max(vals))
    if Emax <= 0:
        Emax = 1.0

    all_edge_vals = []
    for fr in edge_frames[: steps + 1]:
        if fr:
            all_edge_vals.extend(list(fr.values()))
    if not all_edge_vals:
        all_edge_vals = [0.0]
    all_edge_vals = np.asarray(all_edge_vals, dtype=float)
    all_edge_vals = all_edge_vals[np.isfinite(all_edge_vals)]
    if all_edge_vals.size == 0:
        all_edge_vals = np.asarray([0.0], dtype=float)

    bin_edges = np.quantile(all_edge_vals, np.linspace(0.0, 1.0, int(edge_bins) + 1))
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 2:
        bin_edges = np.array([0.0, float(np.max(all_edge_vals) + 1e-9)])

    colors = plotly.colors.sample_colorscale(
        "Plasma",
        np.linspace(0.2, 1.0, max(2, bin_edges.size - 1)),
    )

    # Plotly иногда спотыкается об numpy-типы при JSON-сериализации
    # (особенно внутри frames). Поэтому приводим всё к простым python
    # спискам заранее.
    # Pre-process coordinates
    coords = np.array([pos3d.get(n, (0.0, 0.0, 0.0)) for n in nodes], dtype=float)
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    # UI opts (passed through from tabs/energy.py).
    node_size = float(node_size)
    node_base_size = float(node_base_size if node_base_size is not None else node_size)
    vis_gamma = float(vis_contrast)
    vis_clip = float(vis_clip)
    edge_subset_mode = str(edge_subset_mode or "all").lower()
    max_edges_viz = int(max_edges_viz)

    def _node_traces(frame_idx: int) -> List[go.Scatter3d]:
        """Build node core + glow traces for a "fire" effect."""
        fr = node_frames[frame_idx]
        # Получаем массив энергий
        energies = np.array([float(fr.get(n, 0.0)) for n in nodes], dtype=float)
        energies = np.nan_to_num(energies, nan=0.0, posinf=0.0, neginf=0.0)

        # Нормализация 0..1 для цвета
        intensities = np.clip(energies / Emax, 0.0, 1.0)
        if vis_clip > 0:
            clip_max = max(1e-6, 1.0 - vis_clip)
            intensities = np.clip(intensities, 0.0, clip_max) / clip_max

        # Гамма-коррекция для визуализации (чтобы средние значения были виднее)
        if np.isfinite(vis_gamma) and vis_gamma > 0:
            intensities = np.power(intensities, 1.0 / vis_gamma)

        # Динамический размер: чем больше энергии, тем жирнее узел
        # size = base + base * intensity * multiplier
        sizes = node_base_size * (1.0 + intensities * 2.5)
        traces: List[go.Scatter3d] = []

        # Разделяем на активные и "мертвые" узлы, чтобы мертвые не исчезали полностью.
        # Порог активности: 1% от максимума (в нормировке 0..1).
        mask_active = intensities > 0.01

        # Слой 1: Неактивные узлы — полупрозрачная подложка
        if np.any(~mask_active):
            traces.append(
                go.Scatter3d(
                    x=xs[~mask_active],
                    y=ys[~mask_active],
                    z=zs[~mask_active],
                    mode="markers",
                    marker=dict(
                        size=node_base_size,
                        color="#555555",
                        opacity=0.2,
                    ),
                    hoverinfo="skip",
                    name="nodes_dead",
                )
            )

        # Слой 2: Активные узлы (Core)
        if np.any(mask_active):
            traces.append(
                go.Scatter3d(
                    x=xs[mask_active],
                    y=ys[mask_active],
                    z=zs[mask_active],
                    mode="markers",
                    marker=dict(
                        size=sizes[mask_active],
                        color=intensities[mask_active],
                        colorscale="Blackbody",
                        cmin=0.0,
                        cmax=1.0,
                        opacity=1.0,
                    ),
                    text=[
                        f"{n}: {e:.2f}"
                        for n, e in zip(
                            np.asarray(nodes, dtype=object)[mask_active],
                            energies[mask_active],
                            strict=True,
                        )
                    ],
                    hoverinfo="text",
                    name="nodes_core",
                )
            )

        # 2. Слой свечения (Glow / Halo): Только для активных узлов
        # Берем узлы, где энергия > 5% от макс, чтобы не рисовать гало для мусора
        mask_glow = intensities > 0.05
        if np.any(mask_glow):
            traces.append(
                go.Scatter3d(
                    x=xs[mask_glow],
                    y=ys[mask_glow],
                    z=zs[mask_glow],
                    mode="markers",
                    marker=dict(
                        # Гало в 2 раза больше ядра
                        size=sizes[mask_glow] * 2.2,
                        color=intensities[mask_glow],
                        colorscale="Blackbody",
                        cmin=0.0,
                        cmax=1.0,
                        # Полупрозрачное
                        opacity=0.3,
                    ),
                    hoverinfo="skip",
                    name="nodes_glow",
                )
            )

        return traces

    def _edges_traces(frame_idx: int) -> List[go.Scatter3d]:
        fr = edge_frames[frame_idx]

        # Subsample edges for performance/readability.
        items = list(fr.items())
        if max_edges_viz > 0 and len(items) > max_edges_viz:
            if edge_subset_mode == "top_weight":
                def _w(e):
                    (u, v), _ = e
                    if G.has_edge(u, v):
                        return float(G[u][v].get("weight", 0.0))
                    return 0.0
                items.sort(key=lambda e: abs(_w(e)), reverse=True)
            elif edge_subset_mode in ("top_flux", "top_value"):
                # top flux/value by |val|
                items.sort(key=lambda e: abs(float(e[1])), reverse=True)
            else:
                # "all" mode: if edges are too many, still take the largest-by-|val| subset.
                items.sort(key=lambda e: abs(float(e[1])), reverse=True)
            items = items[:max_edges_viz]

        buckets: List[List[Tuple[float, float, float, float, float, float]]] = [
            [] for _ in range(max(1, bin_edges.size - 1))
        ]
        for (u, v), val in items:
            if u not in pos3d or v not in pos3d:
                continue
            try:
                vv = float(val)
            except Exception:
                continue
            if not np.isfinite(vv):
                continue
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            b = int(np.searchsorted(bin_edges, vv, side="right") - 1)
            b = max(0, min(b, len(buckets) - 1))
            buckets[b].append((float(x0), float(y0), float(z0), float(x1), float(y1), float(z1)))

        traces = []
        for i, segs in enumerate(buckets):
            if not segs:
                continue
            ex = []
            ey = []
            ez = []
            for x0, y0, z0, x1, y1, z1 in segs:
                ex.extend([x0, x1, None])
                ey.extend([y0, y1, None])
                ez.extend([z0, z1, None])
            traces.append(
                go.Scatter3d(
                    x=ex,
                    y=ey,
                    z=ez,
                    mode="lines",
                    line=dict(color=colors[i], width=3),
                    hoverinfo="none",
                    name=f"bin_{i}",
                )
            )
        return traces

    data0 = [*_edges_traces(0), *_node_traces(0)]

    frames = []
    for t in range(steps + 1):
        fr_traces = [*_edges_traces(t), *_node_traces(t)]
        frames.append(go.Frame(data=fr_traces, name=str(t)))

    fig = go.Figure(data=data0, frames=frames)
    # Скорость анимации (мс/кадр) можно передать через kwargs (например, anim_duration=...).
    anim_duration = int(anim_duration)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=anim_duration, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=str(k))
                    for k in range(steps + 1)
                ],
                active=0,
            )
        ],
    )
    return fig


def make_3d_traces(
    G: nx.Graph,
    pos3d: Dict,
    *,
    show_scale: bool = False,
    edge_overlay: str = "weight",
    flow_mode: str = "rw",
    show_nodes: bool = True,
    show_labels: bool = False,
    node_size: int = 6,
    node_opacity: float = 0.85,
    edge_opacity: float = 0.55,
    edge_width_min: float = 1.0,
    edge_width_max: float = 6.0,
    edge_quantiles: int = 7,
    max_nodes_viz: int = 6000,
    max_edges_viz: int = 20000,
    edge_subset_mode: str = "top_abs",
    coord_round: int = 4,
) -> tuple[list[go.Scatter3d], go.Scatter3d | None]:
    """Build edge traces + a node trace for a 3D graph visualization.

    The function returns edge traces separately so callers can adjust node styling
    (size/labels) without rebuilding the edges. Set ``show_scale`` to include a
    colorbar for the selected ``edge_overlay`` metric.
    """
    nodes = list(G.nodes())
    if not nodes:
        return [], None

    # Limit nodes: Plotly 3D gets slow from both compute and browser-side JSON.
    # We keep the most connected nodes to preserve structure.
    if int(max_nodes_viz) > 0 and len(nodes) > int(max_nodes_viz):
        degs = [(n, G.degree(n)) for n in nodes]
        degs.sort(key=lambda t: t[1], reverse=True)
        keep = set([n for n, _ in degs[: int(max_nodes_viz)]])
        nodes = [n for n in nodes if n in keep]

    coords = np.array([pos3d.get(n, (0.0, 0.0, 0.0)) for n in nodes], dtype=np.float32)
    if int(coord_round) >= 0:
        coords = np.round(coords.astype(np.float32), int(coord_round))
    xs = coords[:, 0].astype(float).tolist()
    ys = coords[:, 1].astype(float).tolist()
    zs = coords[:, 2].astype(float).tolist()

    # Color nodes by (unweighted) degree.
    cvals = np.array([G.degree(n) for n in nodes], dtype=float)

    edge_traces: list[go.Scatter3d] = []

    edges = []
    vals = []
    edge_overlay = str(edge_overlay).lower()
    edge_flux: Dict[Tuple, float] | None = None
    if edge_overlay == "flux":
        # Precompute energy flow once to avoid per-edge work.
        _, edge_flux = compute_energy_flow(G, steps=20, flow_mode=str(flow_mode), damping=1.0)
    node_set = set(nodes)
    for u, v, d in G.edges(data=True):
        if u not in node_set or v not in node_set:
            continue
        if u not in pos3d or v not in pos3d:
            continue
        edges.append((u, v))
        if edge_overlay == "confidence":
            vals.append(float(d.get("confidence", 0.0)))
        elif edge_overlay == "ricci":
            # Ricci per-edge is expensive. Keep it usable by computing only when
            # the edge count is already limited (we will also subsample below).
            vals.append(float(ollivier_ricci_edge(G, u, v)))
        elif edge_overlay == "flux" and edge_flux is not None:
            vals.append(float(edge_flux.get((u, v), edge_flux.get((v, u), 0.0))))
        elif edge_overlay == "none":
            vals.append(0.0)
        else:
            vals.append(float(d.get("weight", 0.0)))

    if vals:
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            bins = np.array([0.0, 1.0])
        else:
            bins = np.quantile(v, np.linspace(0.0, 1.0, int(edge_quantiles) + 1))
            bins = np.unique(bins)
            if bins.size < 2:
                bins = np.array([float(v.min()), float(v.max() + 1e-9)])
    else:
        bins = np.array([0.0, 1.0])

    colors = plotly.colors.sample_colorscale("Plasma", np.linspace(0.2, 1.0, max(2, bins.size - 1)))

    buckets: List[List[int]] = [[] for _ in range(max(1, bins.size - 1))]
    for i, val in enumerate(vals):
        b = int(np.searchsorted(bins, float(val), side="right") - 1)
        b = max(0, min(b, len(buckets) - 1))
        buckets[b].append(i)

    # Limit edges after we computed overlay values.
    if int(max_edges_viz) > 0 and len(edges) > int(max_edges_viz):
        idx = np.arange(len(edges), dtype=int)
        vv = np.asarray(vals, dtype=float)
        vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
        mode = str(edge_subset_mode or "top_abs").lower()
        if mode in ("top_abs", "top_value"):
            pick = np.argsort(np.abs(vv))[::-1][: int(max_edges_viz)]
        elif mode in ("top_weight",):
            w = np.array([
                float(G[u][v].get("weight", 0.0)) if G.has_edge(u, v) else 0.0
                for (u, v) in edges
            ], dtype=float)
            pick = np.argsort(np.abs(w))[::-1][: int(max_edges_viz)]
        else:
            # deterministic-ish random subset
            rng = np.random.default_rng(123)
            pick = rng.choice(idx, size=int(max_edges_viz), replace=False)

        # Rebuild buckets with picked indices.
        buckets = [[] for _ in range(max(1, bins.size - 1))]
        for i in pick.tolist():
            val = float(vals[int(i)])
            b = int(np.searchsorted(bins, val, side="right") - 1)
            b = max(0, min(b, len(buckets) - 1))
            buckets[b].append(int(i))

    for bi, idxs in enumerate(buckets):
        if not idxs:
            continue
        ex = []
        ey = []
        ez = []
        for i in idxs:
            u, v = edges[i]
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])
            ez.extend([z0, z1, None])
        width = float(edge_width_min + (edge_width_max - edge_width_min) * (bi / max(1, len(buckets) - 1)))
        edge_traces.append(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color=colors[bi], width=width),
                opacity=float(edge_opacity),
                hoverinfo="none",
                name=f"edges_{bi}",
            )
        )

    if show_scale:
        vmin = float(bins.min())
        vmax = float(bins.max())
        edge_traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    size=0.1,
                    color=[vmin, vmax],
                    colorscale="Plasma",
                    cmin=vmin,
                    cmax=vmax,
                    showscale=True,
                    colorbar=dict(title=edge_overlay),
                ),
                hoverinfo="none",
                name="edge_scale",
                showlegend=False,
            )
        )

    node_trace: go.Scatter3d | None = None
    if show_nodes:
        node_trace = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=int(node_size),
                color=cvals,
                colorscale="Viridis",
                opacity=float(node_opacity),
            ),
            text=[str(n) for n in nodes] if show_labels else None,
            hoverinfo="text",
            name="nodes",
        )

    return edge_traces, node_trace


# ---------------------------------------------------------------------------
# City 3D visualization
# ---------------------------------------------------------------------------

_CITY_NODE_COLORS = {
    "intersection": "#8b949e",
    "home": "#4ade80",
    "hospital": "#ef4444",
    "power_plant": "#facc15",
    "warehouse": "#a16207",
    "shelter": "#3b82f6",
}

_CITY_NODE_SYMBOLS = {
    "intersection": "circle",
    "home": "circle",
    "hospital": "cross",
    "power_plant": "diamond",
    "warehouse": "square",
    "shelter": "circle",
}

_CITY_NODE_LABELS_RU = {
    "intersection": "перекрёсток",
    "home": "дом",
    "hospital": "больница",
    "power_plant": "электростанция",
    "warehouse": "склад",
    "shelter": "убежище",
}


def make_city_3d_figure(
    graph,
    *,
    z_attr: str = "population",
    potentials: Optional[Dict] = None,
    damaged_nodes: Optional[List] = None,
    damaged_edges: Optional[List] = None,
    water_level: float | None = None,
    flooded_nodes: Optional[List] = None,
    flooded_edges: Optional[List] = None,
    title: str = "Городская инфраструктура 3D",
    height: int = 620,
) -> go.Figure:
    """Build an interactive 3D Plotly figure for a typed city graph.

    Parameters
    ----------
    graph : nx.Graph
        City graph with node attributes (type, x, y, population, etc.).
    z_attr : str
        Node attribute to map to Z-axis height. Can be a potentials column
        name if *potentials* dict is provided.
    potentials : dict, optional
        Mapping node -> {potential_name: value} from compute_node_potentials.
    damaged_nodes : list, optional
        Nodes removed in stress test (highlighted in red on the ground plane).
    damaged_edges : list, optional
        Edges removed in stress test (highlighted in red).
    """
    fig = go.Figure()
    nodes = list(graph.nodes())
    if not nodes:
        return fig

    # Prepare Z values.
    z_values = {}
    for node in nodes:
        data = graph.nodes[node]
        if potentials and node in potentials and z_attr in potentials[node]:
            z_values[node] = float(potentials[node][z_attr])
        elif z_attr in data:
            z_values[node] = float(data.get(z_attr, 0.0))
        else:
            z_values[node] = 0.0

    z_candidates = list(z_values.values())
    if water_level is not None and z_attr == "elevation":
        z_candidates.append(float(water_level))
    z_max = max((abs(v) for v in z_candidates), default=1.0) or 1.0
    # Scale Z to a reasonable visual range.
    z_scale = 3.0 / z_max if z_max > 0 else 1.0

    # ---- Edge traces (road / bridge / damaged) ----
    edge_groups = {
        "road": {"xs": [], "ys": [], "zs": [], "color": "#64748b", "width": 1.5, "name": "дорога"},
        "bridge": {"xs": [], "ys": [], "zs": [], "color": "#f97316", "width": 4.0, "name": "мост"},
    }

    damaged_edge_set = set()
    if damaged_edges:
        damaged_edge_set = {frozenset((str(u), str(v))) for u, v in damaged_edges}
    flooded_edge_set = set()
    if flooded_edges:
        flooded_edge_set = {frozenset((str(u), str(v))) for u, v in flooded_edges}

    dmg_xs, dmg_ys, dmg_zs = [], [], []
    flood_xs, flood_ys, flood_zs = [], [], []

    for u, v, data in graph.edges(data=True):
        edge_type = str(data.get("edge_type", "road"))
        ux = float(graph.nodes[u].get("x", 0.0))
        uy = float(graph.nodes[u].get("y", 0.0))
        uz = z_values.get(u, 0.0) * z_scale
        vx = float(graph.nodes[v].get("x", 0.0))
        vy = float(graph.nodes[v].get("y", 0.0))
        vz = z_values.get(v, 0.0) * z_scale

        if frozenset((str(u), str(v))) in damaged_edge_set:
            dmg_xs.extend([ux, vx, None])
            dmg_ys.extend([uy, vy, None])
            dmg_zs.extend([0.0, 0.0, None])
            continue
        if frozenset((str(u), str(v))) in flooded_edge_set:
            flood_xs.extend([ux, vx, None])
            flood_ys.extend([uy, vy, None])
            flood_zs.extend([uz, vz, None])
            continue

        group = edge_groups.get(edge_type, edge_groups["road"])
        group["xs"].extend([ux, vx, None])
        group["ys"].extend([uy, vy, None])
        group["zs"].extend([uz, vz, None])

    for _key, group in edge_groups.items():
        if group["xs"]:
            fig.add_trace(go.Scatter3d(
                x=group["xs"], y=group["ys"], z=group["zs"],
                mode="lines",
                line=dict(color=group["color"], width=group["width"]),
                name=group["name"],
                hoverinfo="skip",
                showlegend=True,
            ))

    if dmg_xs:
        fig.add_trace(go.Scatter3d(
            x=dmg_xs, y=dmg_ys, z=dmg_zs,
            mode="lines",
            line=dict(color="#dc2626", width=5.0, dash="dash"),
            name="повреждённая связь",
            hoverinfo="skip",
            showlegend=True,
        ))

    if flood_xs:
        fig.add_trace(go.Scatter3d(
            x=flood_xs, y=flood_ys, z=flood_zs,
            mode="lines",
            line=dict(color="#38bdf8", width=5.0),
            opacity=0.82,
            name="flooded roads",
            hoverinfo="skip",
            showlegend=True,
        ))

    # ---- Node traces (grouped by type) ----
    damaged_set = set(map(str, damaged_nodes or []))

    for node_type, color in _CITY_NODE_COLORS.items():
        type_nodes = [
            (n, graph.nodes[n]) for n in nodes
            if graph.nodes[n].get("type") == node_type and str(n) not in damaged_set
        ]
        if not type_nodes:
            continue

        xs = [float(d.get("x", 0.0)) for _, d in type_nodes]
        ys = [float(d.get("y", 0.0)) for _, d in type_nodes]
        zs = [z_values.get(n, 0.0) * z_scale for n, _ in type_nodes]
        sizes = _city_node_sizes(type_nodes, node_type)
        texts = _city_hover_texts(type_nodes, z_attr, z_values, potentials)

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=color,
                symbol=_CITY_NODE_SYMBOLS.get(node_type, "circle"),
                opacity=0.92,
                line=dict(color="#1e293b", width=1),
            ),
            text=[str(n) for n, _ in type_nodes],
            textposition="top center",
            textfont=dict(size=9, color="#e2e8f0"),
            customdata=texts,
            hovertemplate="%{customdata}<extra></extra>",
            name=_CITY_NODE_LABELS_RU.get(node_type, node_type),
            showlegend=True,
        ))

    # Damaged nodes: show as red X on the ground plane.
    if damaged_set:
        dmg_nodes = [(n, graph.nodes[n]) for n in damaged_set if n in graph]
        if dmg_nodes:
            fig.add_trace(go.Scatter3d(
                x=[float(d.get("x", 0.0)) for _, d in dmg_nodes],
                y=[float(d.get("y", 0.0)) for _, d in dmg_nodes],
                z=[0.0] * len(dmg_nodes),
                mode="markers+text",
                marker=dict(size=14, color="#dc2626", symbol="x", opacity=0.9,
                            line=dict(color="#991b1b", width=2)),
                text=[str(n) for n, _ in dmg_nodes],
                textposition="top center",
                textfont=dict(size=10, color="#fca5a5"),
                name="удалённые объекты",
                showlegend=True,
                hovertemplate="<b>%{text}</b><br>УДАЛЁН<extra></extra>",
            ))

    flooded_set = set(map(str, flooded_nodes or []))
    if flooded_set:
        flood_nodes = [(n, graph.nodes[n]) for n in flooded_set if n in graph]
        if flood_nodes:
            fig.add_trace(go.Scatter3d(
                x=[float(d.get("x", 0.0)) for _, d in flood_nodes],
                y=[float(d.get("y", 0.0)) for _, d in flood_nodes],
                z=[z_values.get(n, 0.0) * z_scale + 0.04 for n, _ in flood_nodes],
                mode="markers",
                marker=dict(
                    size=18,
                    color="#0ea5e9",
                    symbol="circle",
                    opacity=0.55,
                    line=dict(color="#bae6fd", width=2),
                ),
                text=[str(n) for n, _ in flood_nodes],
                name="flooded nodes",
                hovertemplate="<b>%{text}</b><br>flooded<extra></extra>",
                showlegend=True,
            ))

    if water_level is not None:
        xs_all = [float(graph.nodes[n].get("x", 0.0)) for n in nodes]
        ys_all = [float(graph.nodes[n].get("y", 0.0)) for n in nodes]
        pad = 0.6
        x0, x1 = min(xs_all) - pad, max(xs_all) + pad
        y0, y1 = min(ys_all) - pad, max(ys_all) + pad
        wz = float(water_level) * z_scale if z_attr == "elevation" else 0.0
        fig.add_trace(go.Mesh3d(
            x=[x0, x1, x1, x0],
            y=[y0, y0, y1, y1],
            z=[wz, wz, wz, wz],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#0284c7",
            opacity=0.28,
            name="water level",
            hovertemplate=f"water_level={float(water_level):.2f}<extra></extra>",
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        template="plotly_dark",
        height=int(height),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.05, x=0.5, xanchor="center",
            font=dict(size=11, color="#94a3b8"),
            bgcolor="rgba(15,23,42,0.6)",
        ),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showticklabels=False, title=""),
            zaxis=dict(
                showbackground=False,
                showticklabels=True,
                title=dict(text=z_attr, font=dict(size=11, color="#94a3b8")),
                tickfont=dict(size=9, color="#64748b"),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
    )
    return fig


def _city_node_sizes(
    type_nodes: List,
    node_type: str,
) -> List[float]:
    """Compute marker sizes for city nodes based on capacity/population."""
    if node_type == "intersection":
        return [6.0] * len(type_nodes)
    raw = []
    for _, d in type_nodes:
        val = max(
            float(d.get("population", 0) or 0),
            float(d.get("service_capacity", 0) or 0),
            float(d.get("power_capacity", 0) or 0),
            float(d.get("food_capacity", 0) or 0),
        )
        raw.append(val)
    if not raw:
        return []
    mx = max(raw) or 1.0
    return [8.0 + 14.0 * (v / mx) for v in raw]


def _city_hover_texts(
    type_nodes: List,
    z_attr: str,
    z_values: Dict,
    potentials: Optional[Dict],
) -> List[str]:
    """Build rich hover text for each node."""
    texts = []
    for node, d in type_nodes:
        label = _CITY_NODE_LABELS_RU.get(str(d.get("type", "node")), str(d.get("type", "node")))
        parts = [
            f"<b>{node}</b>",
            f"тип: {label}",
        ]
        pop = int(d.get("population", 0) or 0)
        if pop > 0:
            parts.append(f"жители: {pop}")
        cap = int(d.get("service_capacity", 0) or 0)
        if cap > 0:
            parts.append(f"ёмкость: {cap}")
        pwr = int(d.get("power_capacity", 0) or 0)
        if pwr > 0:
            parts.append(f"мощность: {pwr}")
        food = int(d.get("food_capacity", 0) or 0)
        if food > 0:
            parts.append(f"продовольствие: {food}")
        z_val = z_values.get(node, 0.0)
        parts.append(f"{z_attr}: {z_val:.3f}")
        if potentials and node in potentials:
            for k, v in potentials[node].items():
                if k != z_attr and k not in ("node", "node_type"):
                    parts.append(f"{k}: {float(v):.3f}")
        texts.append("<br>".join(parts))
    return texts
