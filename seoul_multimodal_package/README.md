# Seoul multimodal package: OD flows by mode + reconstructed transport networks

Seoul city only: 427 administrative dongs (codes as of Aug 2026). Prepared 28 Oct 2026. All metadata is in English.

## Naming
- **Dong names:** English labels from Wikidata (Ministry of the Interior style, e.g. "Banpo 1-dong") for 398 dongs. The other 29 use Revised Romanization; `name_source` says which.
- **Gu names:** official English names (e.g. "Jongno-gu").
- **Rail station names:** Wikidata English labels (302 of 307 stations), with the line in brackets, e.g. "Hapjeong Station (Line 6)". The rest use Revised Romanization.
- **Bus stop names and road names:** Revised Romanization, generated with korean-romanizer plus consonant-assimilation rules.
  - Road suffixes are separated, e.g. "Sejong-daero", "Achasan-ro 70-gil".
  - Bus stop names are run-together transliterations and will not match signage translations, e.g. "Dijiteolmidieositiyeok" = Digital Media City Station.
- **Linking back to the source datasets:** codes and IDs (dong_code, stop_id) are unchanged.

## 1. Flows — `flows_od_by_mode.csv.gz`
- **Source:** KT living-mobility data for the capital region, the OD × mode product OA-22657 (Seoul Open Data Plaza).
- **Week:** Mon 14 – Sun 20 Sep 2026, with both trip ends in Seoul.
- **Aggregation:** over all time slots, all user groups (Korean residents, short- and long-term foreigners) and all nationalities. **Divided only by mode.**

| Column | Meaning |
|---|---|
| origin, destination | dong codes (8 digits), matching `nodes_dongs.csv` |
| mode | KT mode code 1–9 (see `mode_codes.csv`) |
| trips_week | total trips over the 7 days (KT-expanded counts) |
| trips_weekday_mean | mean trips per weekday (Mon–Fri) |
| trips_weekend_mean | mean trips per weekend day |
| mean_dist_m | trip-weighted mean straight-line distance reported by KT |
| mean_time_min | trip-weighted mean KT travel time: last record at origin → first at destination; includes access, waiting and detection lag |
| intra_dong | TRUE when origin = destination |

Weekly totals: walk 52.0M, car 33.7M, subway 26.7M, local bus 20.2M, other 18.9M, wide-area bus 0.9M (152.4M in all).

**Caveats**
- KT splits trips at stays of 30 min or more, and mode is inferred.
- "Other" is a residual class.
- Short-term foreigners behave like in-vehicle devices.

## 2. Nodes
- `nodes_dongs.csv`: dong code, name, gu code, gu name, representative point (lon/lat), area and name source. Polygons are in the GeoPackage layer `dongs`.
- `nodes_rail_stops.csv`: rail stops (platform-level IDs) inside Seoul, with lat/lon, dong_code, and the station name with line. `station_group` merges the platforms of one station.
- `nodes_bus_stops.csv`: bus stops inside Seoul, with lat/lon and dong_code.

Stop IDs are KTDB GTFS IDs.

## 3. Edges (one layer per file)
- **`edges_rail.csv`:** consecutive-stop segments from the KTDB national GTFS (weekday timetable, March 2025).
  - Columns: line, trains per day (both directions), straight-line length.
  - `express_chord` = TRUE marks segments of express services that skip a station lying within 250 m of the chord. Drop these for a topological network.
  - gtfs_route_type: 1 = urban rail, 2 = Korail intercity / KTX.
- **`edges_bus.csv.gz`:** consecutive-stop segments for all bus routes running in Seoul.
  - Columns: bus trips per day, number of routes, straight-line length.
  - Segments longer than 8 km were removed (express jumps).
- **Roads:** GeoPackage layer `roads`, with attributes also in `edges_roads_attributes.csv.gz`.
  - OpenStreetMap motorway → secondary_link ways intersecting Seoul (BBBike extract, 24 Oct 2026).
  - `topis_matched` = the road is measured by TOPIS.
  - `v_h00 … v_h23` = TOPIS weekday (14–18 Sep 2026) road-level speeds in km/h: length-weighted harmonic means over the road's links and both directions.
- **`topis_road_speeds_weekday_hourly.csv`:** the same speeds per road (460 roads).

- **Pedestrian layer:** `edges_walk.csv.gz` and `nodes_walk.csv.gz`, with GeoPackage layers `walk_edges` and `walk_nodes`.
  - All OpenStreetMap ways open to walking, which includes minor roads: footways, pedestrian streets, steps, paths, residential, service, living-street and unclassified streets, and primary–tertiary streets (assumed to have sidewalks), plus cycleways and tracks.
  - Excluded: motorways, trunk roads (urban expressways), their links, and anything tagged foot=no.
  - Ways are split at junctions, giving a routable graph: 274,679 edges (14,234 km) between 197,348 nodes.
  - Edge columns: OSM way id, highway type, sidewalk tag, whether the street is named, and length in metres.
  - Node columns: degree and the dong the node lies in.
  - `giant_component` marks the connected network, which holds 99.4% of edge length; 575 small fragments remain, mostly isolated paths in parks and complexes.
  - By length: residential 4,921 km, service 2,460 km, footway 2,425 km, path 983 km, primary 887 km, cycleway 784 km, secondary 695 km, tertiary 558 km.
  - Street names are not exported for this layer (`has_name` only).

Geometry for all layers (EPSG:4326) is in `seoul_multimodal_network.gpkg`. Its layers are dongs, rail_stops, bus_stops, rail_edges, bus_edges, roads, walk_edges and walk_nodes.

## 4. Interlayer couplings — `interlayer_rail_bus_within300m.csv`
Every rail stop paired with every bus stop within 300 m (straight line), with the distance. That gives 4,915 pairs, and all 414 rail stops have at least one bus stop.

Each stop's `dong_code` couples the network layers to the flow layer.

`interlayer_to_walk_nodes.csv` gives the nearest pedestrian-network node for every rail stop, bus stop and dong representative point, with the snapping distance. Median distances: rail 16 m, bus 23 m, dong 29 m.

## 5. Routed OD travel times — `od_routed_travel_times.csv.gz`
R5 (r5py 1.1.7) between dong representative points, all 427 × 427 pairs, in minutes.

| Column | Meaning |
|---|---|
| transit_p50, subway_p50, bus_p50 | median over departures every minute 08:00–09:00, so waiting is included; walking at 4.8 km/h |
| walk | walking at 4.8 km/h |
| car_freeflow | OSM default speeds |
| car_topis_h03, car_topis_h08, car_topis_h17 | OSM speeds replaced by TOPIS measured speeds for that hour; local streets at 0.8 × the TOPIS "other roads" speed (an assumption) |

## Sources and terms
- **KT flows:** Seoul Metropolitan Government & KT, OA-22657. Korea Open Government License (KOGL) Type 1 (attribution).
- **Rail and bus networks:** derived from the KTDB national public-transport GTFS (2025 base), obtained from the Korea Transport Database (ktdb.go.kr) by application (request 53985, FBK). KTDB describes it as a pilot dataset built for accessibility analysis. **Check the KTDB terms before redistributing these layers outside the research team.**
- **Roads:** © OpenStreetMap contributors, ODbL 1.0. Derived databases must keep attribution and the ODbL share-alike terms.
- **Speeds:** Seoul TOPIS (Seoul Transport Operation and Information Service), vehicle speed data, September 2026, from the public reference room.
- **English names:** Wikidata (CC0).

## Scripts
Scripts are in the session's `disaggregation` outputs: build_layers.py, build_walk_layer.py, export_package.py, export_walk.py, anglicize_package.py, romanize_ko.py, r5_matrices.py, r5_car_topis.py and topis_osm_speeds.py.
