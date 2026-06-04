from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.services.urban_resilience import (
    CITY_PRESETS,
    add_city_entity,
    apply_city_entity_edits,
    build_failure_plan,
    build_ml_handoff_bundle,
    city_damage_dataset,
    city_edges_frame,
    city_graph_from_edges,
    city_graph_to_edges,
    city_nodes_frame,
    city_status,
    compute_node_interactions,
    compute_node_potentials,
    create_city_preset,
    format_impact_report,
    generate_random_shape_city_graph,
    get_calculation_explanation,
    has_city_schema,
    recommend_intervention,
    simulate_failure_impact,
)
from src.state_models import GraphEntry
from src.ui.plots.scene3d import make_city_3d_figure

SCENARIOS = [
    "Случайная авария",
    "Удалить выбранный объект",
    "Серия отказов",
    "Атака на самые связные объекты",
    "Атака на мосты и узкие места",
    "Отключить категорию объектов",
    "Затопить нижний район",
]

NODE_TYPE_LABELS = {
    "intersection": "перекрёсток",
    "home": "дом",
    "hospital": "больница",
    "power_plant": "электростанция",
    "warehouse": "склад",
    "shelter": "убежище",
}

NODE_TYPE_COLORS = {
    "intersection": "#8b949e",
    "home": "#2ca02c",
    "hospital": "#d62728",
    "power_plant": "#ffbf00",
    "warehouse": "#8c564b",
    "shelter": "#1f77b4",
}

CATEGORY_OPTIONS = {
    "электростанции": "power_plant",
    "больницы": "hospital",
    "склады": "warehouse",
    "убежища": "shelter",
    "дома": "home",
}

ENTITY_OPTIONS = {
    "дом": "home",
    "больница": "hospital",
    "электростанция": "power_plant",
    "склад": "warehouse",
    "убежище": "shelter",
}


def _open_structure_tab() -> None:
    st.session_state["main_tab"] = "🕸️ 3D"


def render(active_entry: GraphEntry, seed_val: int, add_graph_callback) -> None:
    st.header("Городская песочница устойчивости [EXPERIMENTAL]")
    st.caption(
        "Экспериментальная typed-graph надстройка поверх основного Graph Lab. "
        "Базовый режим проекта остаётся обычной графовой лабораторией; город нужен для прикладных сценариев доступа и отказов. "
        "Соберите маленький город, запустите отказ и посмотрите, кто теряет доступ "
        "к больнице, убежищу, электричеству и складам. Для ML можно скачать готовый пакет."
    )
    is_city_graph = has_city_schema(active_entry.edges)
    graph = None
    if is_city_graph:
        graph = city_graph_from_edges(
            active_entry.edges,
            src_col=active_entry.src_col,
            dst_col=active_entry.dst_col,
        )


    if graph is not None:
        _render_action_center(graph, active_entry, seed_val)

    build_tab, stress_tab, impact_tab, protect_tab, pot_tab, interact_tab = st.tabs(
        ["Собрать", "Стресс-тест", "Последствия", "Защита", "Потенциалы", "Взаимодействия"]
    )

    with build_tab:
        _render_build(seed_val, add_graph_callback, graph, active_entry if is_city_graph else None)

    if graph is None:
        with stress_tab:
            st.info("Загрузите городской пресет во вкладке «Собрать», затем запустите стресс-сценарий.")
        with impact_tab:
            st.info("Отчёт о последствиях появится после загрузки typed-графа города.")
        with protect_tab:
            st.info("Рекомендации по защите появятся после загрузки typed-графа города.")
        with pot_tab:
            st.info("Потенциалы узлов появятся после загрузки typed-графа города.")
        with interact_tab:
            st.info("Взаимодействия узлов появятся после загрузки typed-графа города.")
        return

    with stress_tab:
        _render_stress(graph, seed_val)

    with impact_tab:
        _render_impact(graph, active_entry)

    with protect_tab:
        _render_protect(graph)

    with pot_tab:
        _render_potentials(graph)

    with interact_tab:
        _render_interactions(graph)



def _render_action_center(graph, active_entry: GraphEntry, seed_val: int) -> None:
    status = city_status(graph)
    st.subheader("Что можно сделать сейчас")
    st.markdown(
        "Быстрые сценарии ниже сразу ломают городскую сеть и показывают последствия. "
        "Карта помогает увидеть, где находятся дома, сервисы, дороги и мосты."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Население", int(status["population_total"]))
    m2.metric("Домов", int(status["homes"]))
    m3.metric("Жилых кластеров", int(status["isolated_home_clusters"]))
    m4.metric("Без больницы", int(status["hospital_people_without_access"]))

    with st.expander("📐 Как считаются метрики города", expanded=False):
        st.markdown(get_calculation_explanation("city_status"))

    # --- Map: 3D / 2D toggle ---
    map_mode = st.radio(
        "Режим карты", ["3D", "2D"], horizontal=True, key="urban_map_mode",
        label_visibility="collapsed",
    )
    impact = st.session_state.get("urban_last_impact")
    damaged_nodes = list(impact["plan"].removed_nodes) if impact else None
    damaged_edges = list(impact["plan"].removed_edges) if impact else None
    water_level = impact["plan"].water_level if impact and impact["plan"].water_level is not None else None
    flooded_nodes = list(impact["plan"].flooded_nodes) if impact else None
    flooded_edges = list(impact["plan"].flooded_edges) if impact else None

    if map_mode == "3D":
        z_options = ["elevation", "population", "service_capacity", "power_capacity", "food_capacity"]
        z_attr = st.selectbox(
            "Высота (Z-ось)", z_options, index=0, key="urban_z_attr",
        )
        fig_map = make_city_3d_figure(
            graph,
            z_attr=str(z_attr),
            damaged_nodes=damaged_nodes,
            damaged_edges=damaged_edges,
            water_level=water_level,
            flooded_nodes=flooded_nodes,
            flooded_edges=flooded_edges,
            title=f"Город: {active_entry.name}",
            height=620,
        )
        st.plotly_chart(fig_map, use_container_width=True, key="urban_city_map_3d")
    else:
        st.plotly_chart(_city_map_figure(graph), use_container_width=True, key="urban_city_map")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Проверить мост", use_container_width=True):
        _store_quick_impact(graph, "Атака на мосты и узкие места", seed_val, count=1)
    if c2.button("Отключить электричество", use_container_width=True):
        _store_quick_impact(
            graph,
            "Отключить категорию объектов",
            seed_val,
            category="power_plant",
        )
    if c3.button("Затопить район", use_container_width=True):
        _store_quick_impact(graph, "Затопить нижний район", seed_val)
    if c4.button("Атаковать хабы", use_container_width=True):
        _store_quick_impact(graph, "Атака на самые связные объекты", seed_val, count=3)

    st.button("Открыть этот город в 3D", use_container_width=True, on_click=_open_structure_tab)

    if impact:
        left, right = st.columns([3, 2])
        with left:
            st.text(format_impact_report(impact))
            st.plotly_chart(
                _impact_figure(impact),
                use_container_width=True,
                key="urban_action_impact_chart",
            )
            with st.expander("📐 Как считается уровень ущерба", expanded=False):
                st.markdown(get_calculation_explanation("severity"))
        with right:
            intervention = recommend_intervention(graph, impact)
            st.markdown(f"**Лучшее действие:** {intervention['action']}")
            st.write(
                pd.DataFrame(
                    [
                        {
                            "показатель": "людей без больницы",
                            "до": intervention["before"]["hospital_people_without_access"],
                            "после": intervention["after"]["hospital_people_without_access"],
                        },
                        {
                            "показатель": "людей без убежища",
                            "до": intervention["before"]["shelter_people_without_access"],
                            "после": intervention["after"]["shelter_people_without_access"],
                        },
                        {
                            "показатель": "людей без электричества",
                            "до": intervention["before"]["power_people_without_access"],
                            "после": intervention["after"]["power_people_without_access"],
                        },
                        {
                            "показатель": "оценка устойчивости",
                            "до": round(float(intervention["robustness_before"]), 3),
                            "после": round(float(intervention["robustness_after"]), 3),
                        },
                    ]
                )
            )
            with st.expander("📐 Как работает рекомендация", expanded=False):
                st.markdown(get_calculation_explanation("intervention"))
                st.markdown(get_calculation_explanation("robustness"))

    _render_ml_handoff(graph, active_entry, key_prefix="action")


def _render_ml_handoff(graph, active_entry: GraphEntry, *, key_prefix: str) -> None:
    st.subheader("Передача данных в ML")
    st.markdown(
        "Этот блок готовит материалы для `graph-vulnerability-gnn`: таблицу признаков узлов, "
        "целевой `damage_score`, метку `critical`, исходный typed-граф и manifest со схемой."
    )
    dataset = city_damage_dataset(graph, max_nodes=250)
    bundle = build_ml_handoff_bundle(graph, graph_name=active_entry.name, max_nodes=250)
    edges = city_graph_to_edges(graph)
    nodes = city_nodes_frame(graph)
    roads = city_edges_frame(graph)

    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(
        "Скачать ML-пакет",
        data=bundle,
        file_name=f"{active_entry.name}_ml_handoff.zip",
        mime="application/zip",
        use_container_width=True,
        key=f"{key_prefix}_ml_bundle",
    )
    c2.download_button(
        "Датасет CSV",
        data=dataset.to_csv(index=False).encode("utf-8"),
        file_name=f"{active_entry.name}_city_damage_dataset.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_ml_dataset",
    )
    c3.download_button(
        "Рёбра графа CSV",
        data=edges.to_csv(index=False).encode("utf-8"),
        file_name=f"{active_entry.name}_city_graph_edges.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_ml_edges",
    )
    c4.download_button(
        "Сущности CSV",
        data=nodes.to_csv(index=False).encode("utf-8"),
        file_name=f"{active_entry.name}_city_nodes.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_ml_nodes",
    )

    st.plotly_chart(
        _ml_damage_figure(dataset),
        use_container_width=True,
        key=f"{key_prefix}_ml_damage_chart",
    )

    with st.expander("Посмотреть таблицы для ML", expanded=False):
        t1, t2, t3 = st.tabs(["датасет", "рёбра графа", "дороги"])
        with t1:
            st.dataframe(dataset.head(30), use_container_width=True)
        with t2:
            st.dataframe(edges.head(30), use_container_width=True)
        with t3:
            st.dataframe(roads.head(30), use_container_width=True)


def _store_quick_impact(
    graph,
    scenario: str,
    seed_val: int,
    *,
    count: int = 1,
    category: str = "power_plant",
    water_level: float | None = None,
) -> None:
    plan = build_failure_plan(
        graph,
        scenario,
        count=int(count),
        category=category,
        seed=int(seed_val),
        water_level=water_level,
    )
    st.session_state["urban_last_impact"] = simulate_failure_impact(graph, plan)


def _render_build(
    seed_val: int,
    add_graph_callback,
    graph,
    active_entry: GraphEntry | None,
) -> None:
    st.subheader("Собрать город")
    st.markdown(
        "Выберите готовый городской шаблон или отредактируйте текущие сущности ниже. "
        "После сохранения будет создан новый активный граф, старый останется в списке."
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        preset = st.selectbox("Городской шаблон", list(CITY_PRESETS.keys()), key="urban_preset_build")
    with c2:
        seed = st.number_input("Seed города", value=int(seed_val), step=1, key="urban_seed_build")

    if st.button("Загрузить городской шаблон", type="primary", use_container_width=True):
        preset_graph = create_city_preset(str(preset), seed=int(seed))
        add_graph_callback(
            f"Urban {preset} seed={int(seed)}",
            city_graph_to_edges(preset_graph),
            "urban_resilience",
            "src",
            "dst",
        )
        st.session_state.pop("urban_last_impact", None)
        st.rerun()

    with st.expander("Random shape city", expanded=True):
        r1, r2, r3 = st.columns(3)
        intersections = r1.slider("Intersections", 10, 80, 32, key="urban_random_intersections")
        homes = r2.slider("Homes", 4, 80, 24, key="urban_random_homes")
        random_seed = r3.number_input("Random seed", value=int(seed), step=1, key="urban_random_seed")
        s1, s2, s3, s4 = st.columns(4)
        hospitals = s1.slider("Hospitals", 1, 6, 2, key="urban_random_hospitals")
        power_plants = s2.slider("Power plants", 1, 5, 1, key="urban_random_power")
        warehouses = s3.slider("Warehouses", 1, 6, 2, key="urban_random_warehouses")
        shelters = s4.slider("Shelters", 1, 6, 2, key="urban_random_shelters")
        if st.button("Generate random shape city", use_container_width=True, key="urban_random_shape_btn"):
            random_graph = generate_random_shape_city_graph(
                intersections=int(intersections),
                homes=int(homes),
                hospitals=int(hospitals),
                power_plants=int(power_plants),
                warehouses=int(warehouses),
                shelters=int(shelters),
                seed=int(random_seed),
            )
            add_graph_callback(
                f"Urban random shape seed={int(random_seed)}",
                city_graph_to_edges(random_graph),
                "urban_resilience:random_shape",
                "src",
                "dst",
            )
            st.session_state.pop("urban_last_impact", None)
            st.rerun()

    if graph is None or active_entry is None:
        return

    st.markdown("---")
    st.subheader("Редактировать сущности")
    _render_entity_editor(graph, active_entry, add_graph_callback)


def _render_entity_editor(graph, active_entry: GraphEntry, add_graph_callback) -> None:
    nodes_df = city_nodes_frame(graph)
    edges_df = city_edges_frame(graph)

    edited_nodes = st.data_editor(
        nodes_df,
        key=f"urban_nodes_editor_{active_entry.id}",
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["node"],
        column_config={
            "type": st.column_config.SelectboxColumn(
                "тип",
                options=[
                    "intersection",
                    "home",
                    "hospital",
                    "power_plant",
                    "warehouse",
                    "shelter",
                ],
            ),
            "medical_need": st.column_config.SelectboxColumn(
                "медицинская потребность",
                options=["", "low", "medium", "high"],
            ),
        },
    )

    edited_edges = st.data_editor(
        edges_df,
        key=f"urban_edges_editor_{active_entry.id}",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "edge_type": st.column_config.SelectboxColumn(
                "тип связи",
                options=["road", "bridge"],
            ),
        },
    )

    if st.button("Сохранить правки как новый граф", type="primary", use_container_width=True):
        try:
            edited_graph = apply_city_entity_edits(edited_nodes, edited_edges)
        except ValueError as exc:
            st.error(str(exc))
            return
        add_graph_callback(
            f"{active_entry.name} edited",
            city_graph_to_edges(edited_graph),
            "urban_resilience:edited",
            "src",
            "dst",
        )
        st.session_state.pop("urban_last_impact", None)
        st.rerun()

    with st.expander("Добавить объект", expanded=False):
        entity_label = st.selectbox(
            "Тип",
            list(ENTITY_OPTIONS.keys()),
            key=f"urban_new_type_{active_entry.id}",
        )
        entity_type = ENTITY_OPTIONS[entity_label]
        node_prefix = {
            "home": "H",
            "hospital": "MED",
            "power_plant": "PWR",
            "warehouse": "WH",
            "shelter": "SH",
        }[entity_type]
        default_id = _next_node_id(graph, node_prefix)
        node_id = st.text_input("ID", value=default_id, key=f"urban_new_id_{active_entry.id}")
        connect_to = st.selectbox(
            "Подключить к",
            sorted(map(str, graph.nodes())),
            key=f"urban_new_connect_{active_entry.id}",
        )
        c1, c2, c3, c4 = st.columns(4)
        population = c1.number_input("Жители", 0, 1000, 6 if entity_type == "home" else 0)
        service_capacity = c2.number_input(
            "Ёмкость сервиса",
            0,
            1000,
            60 if entity_type in ("hospital", "shelter") else 0,
        )
        power_capacity = c3.number_input(
            "Мощность",
            0,
            1000,
            120 if entity_type == "power_plant" else 0,
        )
        food_capacity = c4.number_input(
            "Запас еды",
            0,
            1000,
            120 if entity_type == "warehouse" else 0,
        )
        medical_need = st.selectbox(
            "Медицинская потребность",
            ["", "low", "medium", "high"],
            index=2 if entity_type == "home" else 0,
            key=f"urban_new_medical_{active_entry.id}",
        )
        travel_time = st.number_input("Время пути по новой дороге", 0.1, 100.0, 2.0, step=0.5)

        if st.button("Добавить подключённый объект", use_container_width=True):
            try:
                edited_graph = add_city_entity(
                    graph,
                    node_id=node_id,
                    node_type=entity_type,
                    connect_to=connect_to,
                    population=int(population),
                    service_capacity=int(service_capacity),
                    power_capacity=int(power_capacity),
                    food_capacity=int(food_capacity),
                    medical_need=medical_need,
                    travel_time=float(travel_time),
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            add_graph_callback(
                f"{active_entry.name} + {node_id}",
                city_graph_to_edges(edited_graph),
                "urban_resilience:edited",
                "src",
                "dst",
            )
            st.session_state.pop("urban_last_impact", None)
            st.rerun()


def _render_stress(graph, seed_val: int) -> None:
    st.subheader("Стресс-тест")
    st.markdown(
        "Здесь можно вручную выбрать сценарий отказа. Быстрые кнопки сверху делают то же самое, "
        "но этот блок даёт больше контроля."
    )
    scenario = st.selectbox("Сценарий", SCENARIOS)
    count = st.slider("Сколько объектов затронуть", 1, 10, 1)
    selected = None
    category = "power_plant"

    if scenario == "Удалить выбранный объект":
        selected = st.selectbox("Объект", sorted(map(str, graph.nodes())))
    if scenario == "Отключить категорию объектов":
        category_label = st.selectbox(
            "Категория",
            list(CATEGORY_OPTIONS.keys()),
        )
        category = CATEGORY_OPTIONS[category_label]

    water_level = None
    if "Flood" in scenario or "Затоп" in scenario:
        elev_min, elev_max = _elevation_range(graph)
        default_level = float(elev_min + (elev_max - elev_min) * 0.35)
        water_level = st.slider(
            "Уровень воды",
            min_value=float(elev_min),
            max_value=float(elev_max),
            value=default_level,
            step=max(0.01, float(elev_max - elev_min) / 100.0),
            key="urban_flood_water_level",
        )

    seed = st.number_input("Seed сценария", value=int(seed_val), step=1, key="urban_stress_seed")

    if st.button("Запустить стресс-тест", type="primary", use_container_width=True):
        plan = build_failure_plan(
            graph,
            scenario,
            count=int(count),
            selected_object=selected,
            category=category,
            seed=int(seed),
            water_level=water_level,
        )
        impact = simulate_failure_impact(graph, plan)
        st.session_state["urban_last_impact"] = impact
        st.success("Стресс-тест выполнен.")

    impact = st.session_state.get("urban_last_impact")
    if impact:
        plan = impact["plan"]
        st.write(
            pd.DataFrame(
                {
                    "удалённые узлы": [", ".join(plan.removed_nodes)],
                    "удалённые дороги/мосты": [", ".join(f"{u}-{v}" for u, v in plan.removed_edges)],
                    "уровень ущерба": [impact["severity"]],
                }
            )
        )
        with st.expander("📐 Как считается ущерб", expanded=False):
            st.markdown(get_calculation_explanation("severity"))
            st.markdown(get_calculation_explanation("damage_score"))


def _next_node_id(graph, prefix: str) -> str:
    existing = {str(node) for node in graph.nodes()}
    idx = 1
    while f"{prefix}{idx}" in existing:
        idx += 1
    return f"{prefix}{idx}"


def _elevation_range(graph) -> tuple[float, float]:
    values = [
        float(data.get("elevation", data.get("y", 0.0)))
        for _, data in graph.nodes(data=True)
    ]
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _render_impact(graph, active_entry: GraphEntry) -> None:
    st.subheader("Последствия человеческим языком")
    impact = st.session_state.get("urban_last_impact")
    if not impact:
        st.info("Сначала запустите стресс-тест.")
        return

    st.text(format_impact_report(impact))
    st.plotly_chart(
        _impact_figure(impact),
        use_container_width=True,
        key="urban_impact_tab_chart",
    )

    after = impact["after"]
    cols = st.columns(4)
    cols[0].metric("Людей без больницы", int(after["hospital_people_without_access"]))
    cols[1].metric("Людей без убежища", int(after["shelter_people_without_access"]))
    cols[2].metric("Людей без электричества", int(after["power_people_without_access"]))
    cols[3].metric("Жилых кластеров", int(after["isolated_home_clusters"]))

    with st.expander("📐 Как считаются потери доступа", expanded=False):
        st.markdown(get_calculation_explanation("severity"))
        st.markdown(get_calculation_explanation("city_status"))

    with st.expander("Экспорт материалов для ML", expanded=False):
        _render_ml_handoff(graph, active_entry, key_prefix="impact")


def _render_protect(graph) -> None:
    st.subheader("Что укрепить первым")
    impact = st.session_state.get("urban_last_impact")
    if not impact:
        st.info("Сначала запустите стресс-тест.")
        return

    intervention = recommend_intervention(graph, impact)
    st.markdown(f"**Лучшее действие:** {intervention['action']}")
    before = intervention["before"]
    after = intervention["after"]
    st.write(
        pd.DataFrame(
            [
                {
                    "показатель": "людей без больницы",
                    "до": before["hospital_people_without_access"],
                    "после": after["hospital_people_without_access"],
                },
                {
                    "показатель": "людей без убежища",
                    "до": before["shelter_people_without_access"],
                    "после": after["shelter_people_without_access"],
                },
                {
                    "показатель": "людей без электричества",
                    "до": before["power_people_without_access"],
                    "после": after["power_people_without_access"],
                },
                {
                    "показатель": "оценка устойчивости",
                    "до": round(float(intervention["robustness_before"]), 3),
                    "после": round(float(intervention["robustness_after"]), 3),
                },
            ]
        )
    )
    with st.expander("📐 Как работает рекомендация", expanded=False):
        st.markdown(get_calculation_explanation("intervention"))
        st.markdown(get_calculation_explanation("robustness"))


def _city_map_figure(graph) -> go.Figure:
    fig = go.Figure()
    for edge_type, color, width, name in [
        ("road", "#94a3b8", 2, "дорога"),
        ("bridge", "#f97316", 5, "мост"),
    ]:
        xs = []
        ys = []
        for u, v, data in graph.edges(data=True):
            if data.get("edge_type", "road") != edge_type:
                continue
            xs.extend([graph.nodes[u].get("x", 0.0), graph.nodes[v].get("x", 0.0), None])
            ys.extend([graph.nodes[u].get("y", 0.0), graph.nodes[v].get("y", 0.0), None])
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"color": color, "width": width},
                    name=name,
                    hoverinfo="skip",
                )
            )

    for node_type, label in NODE_TYPE_LABELS.items():
        nodes = [(node, data) for node, data in graph.nodes(data=True) if data.get("type") == node_type]
        if not nodes:
            continue
        fig.add_trace(
            go.Scatter(
                x=[data.get("x", 0.0) for _, data in nodes],
                y=[data.get("y", 0.0) for _, data in nodes],
                mode="markers+text",
                text=[str(node) for node, _ in nodes],
                textposition="top center",
                marker={
                    "size": 13 if node_type != "intersection" else 8,
                    "color": NODE_TYPE_COLORS.get(node_type, "#64748b"),
                    "line": {"color": "#111827", "width": 1},
                },
                name=label,
                customdata=[
                    [
                        label,
                        int(data.get("population", 0) or 0),
                        int(data.get("service_capacity", 0) or 0),
                    ]
                    for _, data in nodes
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "тип: %{customdata[0]}<br>"
                    "жители: %{customdata[1]}<br>"
                    "ёмкость: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Карта городской инфраструктуры",
        template="plotly_white",
        height=520,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        legend={"orientation": "h", "y": -0.06},
    )
    return fig


def _impact_figure(impact: dict[str, object]) -> go.Figure:
    before = impact["before"]
    after = impact["after"]
    frame = pd.DataFrame(
        [
            {
                "ресурс": "больница",
                "до отказа": before["hospital_people_without_access"],
                "после отказа": after["hospital_people_without_access"],
            },
            {
                "ресурс": "убежище",
                "до отказа": before["shelter_people_without_access"],
                "после отказа": after["shelter_people_without_access"],
            },
            {
                "ресурс": "электричество",
                "до отказа": before["power_people_without_access"],
                "после отказа": after["power_people_without_access"],
            },
        ]
    )
    long_frame = frame.melt(id_vars="ресурс", var_name="состояние", value_name="людей без доступа")
    fig = px.bar(
        long_frame,
        x="ресурс",
        y="людей без доступа",
        color="состояние",
        barmode="group",
        title="Потеря доступа к критическим ресурсам",
        color_discrete_map={"до отказа": "#94a3b8", "после отказа": "#ef4444"},
    )
    fig.update_layout(template="plotly_white", height=360, margin={"l": 10, "r": 10, "t": 55, "b": 10})
    return fig


def _ml_damage_figure(dataset: pd.DataFrame) -> go.Figure:
    top = dataset.sort_values("damage_score", ascending=False).head(12).copy()
    top["тип"] = top["node_type"].map(NODE_TYPE_LABELS).fillna(top["node_type"])
    fig = px.bar(
        top.sort_values("damage_score"),
        x="damage_score",
        y="node",
        color="тип",
        orientation="h",
        title="Топ объектов для ML по damage_score",
        labels={"damage_score": "ущерб при удалении", "node": "объект"},
        color_discrete_map={label: NODE_TYPE_COLORS.get(key, "#64748b") for key, label in NODE_TYPE_LABELS.items()},
    )
    fig.update_layout(template="plotly_white", height=420, margin={"l": 10, "r": 10, "t": 55, "b": 10})
    return fig


# ---------------------------------------------------------------------------
# Potentials tab
# ---------------------------------------------------------------------------

_POTENTIAL_COLS = [
    "access_potential",
    "connectivity_potential",
    "vulnerability_potential",
    "service_potential",
    "evacuation_potential",
]

_POTENTIAL_LABELS_RU = {
    "access_potential": "Доступность",
    "connectivity_potential": "Связность",
    "vulnerability_potential": "Уязвимость",
    "service_potential": "Сервисность",
    "evacuation_potential": "Эвакуация",
}


def _render_potentials(graph) -> None:
    st.subheader("📊 Потенциалы узлов")
    st.markdown(
        "5 городских потенциалов — heuristic-метрики, описывающие роль каждого узла "
        "в инфраструктурной сети. Все нормированы от 0 до 1."
    )

    if st.button("Рассчитать потенциалы", type="primary", key="calc_potentials"):
        with st.spinner("Вычисление потенциалов..."):
            pot_df = compute_node_potentials(graph)
        st.session_state["urban_potentials_df"] = pot_df

    pot_df = st.session_state.get("urban_potentials_df")
    if pot_df is None or pot_df.empty:
        st.info("Нажмите «Рассчитать потенциалы» для анализа.")
        return

    # Explanation expanders.
    with st.expander("📐 Формулы потенциалов", expanded=False):
        for key in _POTENTIAL_COLS:
            st.markdown(get_calculation_explanation(key))
            st.markdown("---")

    # Table.
    st.dataframe(
        pot_df.style.format(
            {col: "{:.3f}" for col in _POTENTIAL_COLS},
            na_rep="—",
        ),
        use_container_width=True,
        height=320,
    )

    # --- Radar chart for selected node ---
    st.markdown("#### 🕸️ Радар узла")
    node_options = pot_df["node"].tolist()
    selected_node = st.selectbox("Выберите узел", node_options, key="pot_radar_node")
    row = pot_df[pot_df["node"] == selected_node]
    if not row.empty:
        r_values = [float(row.iloc[0][col]) for col in _POTENTIAL_COLS]
        categories = [_POTENTIAL_LABELS_RU.get(c, c) for c in _POTENTIAL_COLS]
        # Close the radar polygon.
        r_values_closed = r_values + [r_values[0]]
        categories_closed = categories + [categories[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.25)",
            line=dict(color="#6366f1", width=2),
            name=str(selected_node),
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
                bgcolor="rgba(15,23,42,0.8)",
            ),
            template="plotly_dark",
            height=400,
            margin=dict(l=40, r=40, t=30, b=30),
            showlegend=False,
            title=dict(
                text=f"Потенциалы: {selected_node} ({row.iloc[0].get('node_type', '')})",
                font=dict(size=14),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="pot_radar_chart")

    # --- 3D heatmap by selected potential ---
    st.markdown("#### 🗺️ 3D-карта потенциала")
    pot_choice = st.selectbox(
        "Потенциал для Z-высоты",
        _POTENTIAL_COLS,
        format_func=lambda c: _POTENTIAL_LABELS_RU.get(c, c),
        key="pot_3d_z",
    )
    potentials_dict = {}
    for _, r in pot_df.iterrows():
        potentials_dict[str(r["node"])] = {col: float(r[col]) for col in _POTENTIAL_COLS}
    fig_pot_3d = make_city_3d_figure(
        graph,
        z_attr=str(pot_choice),
        potentials=potentials_dict,
        title=f"3D: {_POTENTIAL_LABELS_RU.get(pot_choice, pot_choice)}",
        height=560,
    )
    st.plotly_chart(fig_pot_3d, use_container_width=True, key="pot_3d_map")


# ---------------------------------------------------------------------------
# Interactions tab
# ---------------------------------------------------------------------------


def _render_interactions(graph) -> None:
    st.subheader("🔗 Взаимодействия узлов")
    st.markdown(
        "Парный анализ между важными объектами города: дистанции, зависимости, "
        "избыточность путей, общее обслуживаемое население."
    )

    if st.button("Рассчитать взаимодействия", type="primary", key="calc_interactions"):
        with st.spinner("Вычисление парных характеристик..."):
            inter_df = compute_node_interactions(graph)
        st.session_state["urban_interactions_df"] = inter_df

    inter_df = st.session_state.get("urban_interactions_df")
    if inter_df is None or inter_df.empty:
        st.info("Нажмите «Рассчитать взаимодействия» для анализа.")
        return

    with st.expander("📐 Как считаются взаимодействия", expanded=False):
        st.markdown(get_calculation_explanation("interactions"))

    # Filter by interaction type.
    types = sorted(inter_df["interaction_type"].unique().tolist())
    selected_types = st.multiselect(
        "Тип взаимодействия", types, default=types, key="inter_type_filter",
    )
    filtered = inter_df[inter_df["interaction_type"].isin(selected_types)]

    st.dataframe(
        filtered.style.format(
            {"distance": "{:.2f}", "dependency_score": "{:.4f}"},
            na_rep="—",
        ),
        use_container_width=True,
        height=400,
    )

    # --- Heatmap of dependency scores ---
    if len(filtered) > 1:
        st.markdown("#### 🔥 Тепловая карта зависимостей")
        all_nodes = sorted(set(filtered["source"].tolist() + filtered["target"].tolist()))
        matrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
        for _, r in filtered.iterrows():
            matrix.loc[str(r["source"]), str(r["target"])] = float(r["dependency_score"])
            matrix.loc[str(r["target"]), str(r["source"])] = float(r["dependency_score"])
        fig_heat = go.Figure(data=go.Heatmap(
            z=matrix.values.tolist(),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="Magma",
            reversescale=True,
            hovertemplate="%{y} ↔ %{x}: %{z:.4f}<extra></extra>",
        ))
        fig_heat.update_layout(
            template="plotly_dark",
            height=max(350, 30 * len(all_nodes)),
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="Зависимости между объектами", font=dict(size=14)),
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_heat, use_container_width=True, key="inter_heatmap")

    # Top interactions summary.
    st.markdown("#### 📋 Топ-10 критических взаимодействий")
    top10 = filtered.head(10)
    for _idx, r in top10.iterrows():
        st.markdown(
            f"**{r['source']}** ↔ **{r['target']}** — "
            f"_{r['interaction_type']}_ | "
            f"dist={r['distance']:.1f}, dep={r['dependency_score']:.4f}, "
            f"redundancy={r['redundancy']}, shared_pop={r['shared_population']}"
        )
