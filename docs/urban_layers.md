# Слои Urban sandbox

`[EXPERIMENTAL] Urban sandbox` - это не второе приложение рядом с Graph Lab, а typed-graph надстройка поверх обычной графовой модели. Городской граф всё ещё можно прогонять через общие слои Graph Lab, но Urban добавляет типы городских объектов, сценарии отказов, access-loss метрики и отдельный ML export для соседнего проекта `graph-vulnerability-gnn`.

Основные файлы реализации:

- `src/services/urban_resilience.py` - схема города, пресеты, симуляция отказов, impact-метрики, рекомендации и Urban ML bundle.
- `src/ui/tabs/urban.py` - основной Streamlit UI для Urban sandbox.
- `src/layers/urban.py` - `UrbanLayer` для общего layer pipeline.
- `src/layers/ml_export.py` - переключает city schema на Urban ML handoff.
- `src/services/layer_service.py` - регистрирует `UrbanLayer` вместе с обычными графовыми слоями.

## 1. Что здесь значит "слой"

В Urban есть два разных смысла слова "слой":

- **пользовательские слои Urban UI**: Build, Stress Test, Impact, Protect, ML Handoff;
- **вычислительный слой** `UrbanLayer` во вкладке `Layers`.

Это связанные части, но не дубли. Urban tab остаётся главным пользовательским интерфейсом для городских сценариев. `UrbanLayer` - компактная обёртка для общего pipeline: она берёт текущий city graph и превращает его в graph metrics, node labels и artifacts внутри `AugmentedGraph`.

## 2. Слой city schema

Urban graph - это обычный неориентированный `networkx.Graph` с дополнительными typed-атрибутами. Таблица считается городской, если в edge table есть колонки:

```text
src_type
dst_type
edge_type
```

Текущие типы узлов:

```text
intersection
home
hospital
power_plant
warehouse
shelter
```

Основные node attributes:

```text
type
label
x
y
population
service_capacity
power_capacity
food_capacity
medical_need
```

Основные edge attributes:

```text
edge_type
label
weight
confidence
travel_time
capacity
fragility
```

В Urban `weight` сейчас берётся из `travel_time`, то есть ведёт себя как стоимость/время пути. Это отличается от обычной Graph Lab трактовки, где больший `weight` часто означает более сильную связь. Именно поэтому Urban помечен как experimental: он использует общую графовую инфраструктуру, но семантика рёбер у него городская.

## 3. Слои Urban UI

Urban tab состоит из пяти практических уровней работы.

### Build

Build создаёт или редактирует typed city graph. Сейчас поддерживаются:

- городские пресеты;
- редактирование city entities;
- редактирование дорог и мостов;
- добавление нового объекта с подключением к существующему графу.

После редактирования город снова сериализуется в обычную edge table через `city_graph_to_edges()`, поэтому его можно открыть в общей 3D-вкладке и прогнать через обычные graph layers.

### Stress Test

Stress Test создаёт `FailurePlan` и применяет его к текущему городскому графу. Реализованные семейства сценариев:

- отказ мостов или узких мест;
- отключение электростанций;
- атака на самые связные объекты;
- атака на мосты и bottleneck-edges;
- отключение категории объектов;
- затопление нижнего района.

Сценарий удаляет выбранные узлы и/или рёбра, после чего пересчитывается состояние города.

### Impact

Impact сравнивает состояние города до и после отказа. Главные access-loss метрики:

```text
hospital_people_without_access
shelter_people_without_access
power_people_without_access
food_homes_without_access
hospital_avg_distance
isolated_home_clusters
```

Severity считается по доле населения, потерявшего доступ к критичному сервису:

```text
severity_value = unavailable_people / population_total
```

Текущие интервалы:

```text
низкий        severity_value < 0.10
средний       0.10 <= severity_value < 0.25
высокий       0.25 <= severity_value < 0.50
критический   severity_value >= 0.50
```

Это прикладная эвристика для UI, а не валидированная модель городского риска.

### Protect

Protect после отказа ищет простое восстановительное вмешательство. Текущая модель перебирает кандидатов на резервную дорогу между домом и сервисным объектом, добавляет ребро в копию повреждённого графа и сравнивает city status до/после.

Это heuristic recommendation layer, не оптимизатор с гарантией глобального optimum.

### ML Handoff

Urban UI умеет выгружать ZIP для соседнего ML-проекта. Архив содержит:

```text
city_damage_dataset.csv
city_graph_edges.csv
city_nodes.csv
city_roads.csv
ml_manifest.json
README.md
```

Это отдельный Urban bundle. Он не совпадает с generic `ml_handoff.zip`, который `MLExportLayer` строит для обычных графов.

## 4. UrbanLayer в общем Layers pipeline

`UrbanLayer` зарегистрирован в `LayerService.default_registry()`, но выключен по умолчанию.

Default config:

```text
enabled = false
max_nodes = 250
include_damage_dataset = true
heavy = false
```

Правило запуска:

- если у графа нет city schema, слой возвращает `skipped`;
- если city schema есть, слой считает city status metrics и при необходимости node damage dataset.

`UrbanLayer` пишет graph metrics с префиксом `urban_`, например:

```text
urban_population_total
urban_hospital_people_without_access
urban_shelter_people_without_access
urban_power_people_without_access
urban_isolated_home_clusters
```

Если `include_damage_dataset=true`, слой добавляет в `AugmentedGraph.node_attributes` выбранные node columns:

```text
node
node_type
damage_score
critical
severity
hospital_people_without_access
shelter_people_without_access
power_people_without_access
```

И отдаёт artifact:

```text
city_damage_dataset_csv
```

## 5. Urban damage dataset

`city_damage_dataset()` строит одну строку на узел, максимум до `max_nodes`. Для каждого candidate node он:

1. удаляет узел из копии city graph;
2. считает структурную потерю LCC;
3. запускает `simulate_failure_impact()`;
4. добавляет graph/node features и city access-loss labels.

Текущие колонки:

```text
graph_id
node
graph_family
graph_n_nodes
graph_n_edges
node_type
degree
degree_norm
strength
strength_norm
betweenness
closeness
clustering
pagerank
eigenvector
core_number
core_number_norm
local_density
energy_final
energy_peak_pressure
energy_cumulative_inflow
energy_overload_risk
damage_score
critical
severity
hospital_people_without_access
shelter_people_without_access
power_people_without_access
```

Energy columns в Urban dataset сейчас являются нулевыми placeholder-колонками:

```text
energy_final = 0.0
energy_peak_pressure = 0.0
energy_cumulative_inflow = 0.0
energy_overload_risk = 0.0
```

Это сохраняет близость к generic ML contract, но честно показывает, что Urban bundle пока не подтягивает результаты `FlowLayer`.

## 6. Важный нюанс про damage_score

Generic `VulnerabilityLayer` сейчас использует ML-compatible contract:

```text
damage_score = LCC_fraction_before - LCC_fraction_after
denominator = original graph node count
```

Urban-specific `city_damage_dataset()` пока использует старую LCC-relative формулу:

```text
damage_score = 1 - LCC_after / LCC_before
```

Для экспериментального Urban overlay это допустимо, но это контрактное отличие. Если Urban rows смешиваются с generic Graph Lab exports в `graph-vulnerability-gnn`, формулу нужно сначала унифицировать или явно держать отдельную convention через:

```text
graph_family = urban_resilience
```

## 7. Связь с обычными graph layers

Urban graph можно прогонять через обычные слои:

- `CoreMetricsLayer` считает глобальные graph metrics.
- `NodeMetricsLayer` считает centrality/features для узлов.
- `EdgeMetricsLayer` считает edge features, bridges и approximate edge betweenness.
- `VulnerabilityLayer` считает generic structural node/edge removal labels.
- `FlowLayer` может запустить общую flow-модель, но Urban ML bundle пока не потребляет эти значения.
- `MLExportLayer` распознаёт city schema и отдаёт `urban_ml_handoff_zip` вместо generic `ml_handoff_zip`.

Задуманное разделение:

- Urban tab - city scenarios, access-loss interpretation, intervention hints;
- Layers tab - pipeline output, merged attributes, artifacts, exports;
- `graph-vulnerability-gnn` - обучение моделей на выгруженных датасетах.

## 8. Текущие ограничения

Urban пока experimental. Ограничения:

- нет временной эвакуации и recovery simulation;
- нет capacity-constrained routing сверх простых проверок доступности;
- нет вероятностной модели множественных отказов;
- нет калиброванной транспортной физики;
- нет обучения ML-модели внутри Graph Lab;
- Urban `damage_score` ещё не унифицирован с generic `VulnerabilityLayer`;
- Urban energy features в handoff bundle пока placeholder, если отдельно не маппить туда `FlowLayer`.

Честный статус:

```text
Urban sandbox = typed city graph overlay + access-loss simulation + ML handoff,
not a validated urban planning simulator.
```
