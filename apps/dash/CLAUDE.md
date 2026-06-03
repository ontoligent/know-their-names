# Person Network Explorer — Dash App

Interactive ego-network explorer for the Know Their Names capstone project (DS 6105, Summer 2026).

## Running the app

```bash
# From the repo root, activate the shared venv first:
source activate.sh

# Then from this directory:
cd apps/dash
python app.py
# → open http://localhost:8050
```

## What it does

- Search for any person by name or key fragment (min 2 characters)
- Select from matching results to set the center node
- A Cytoscape force-directed graph renders their ego network
- Slider controls degrees of separation (1–4)
- Hover over a node for details; click a node to navigate to that person

## Data

Reads two Parquet files from `../../data/` (produced by `notebooks/01-import.ipynb`):

| File | Index | Key columns |
|------|-------|-------------|
| `PERSON.parquet` | `person_key` | `n_mentions`, `n_relations` |
| `RELATION.parquet` | `(subject_key, predicate, object_key)` | `n_assertions` |

**Person key format:** `{birth_year}-{legal_status}-{norm_race}-{gender}-{full_name}`  
e.g. `1852-f-b-m-david_tompson` → born 1852, free, Black, male, David Tompson

**Legal status codes:** `e` = enslaved, `f` = free, `x` = unknown  
**Race codes:** `b` = Black, `w` = White, `m` = Mixed, `x` = unknown

## Architecture

Single-file Dash app (`app.py`). Key sections:

1. **Data loading** — reads Parquet, deduplicates to one directed edge per subject→object pair (keeping highest `n_assertions`), builds a `networkx.DiGraph`
2. **`parse_key(key)`** — splits a person_key string into its five components
3. **`ego_elements(center_key, degrees)`** — calls `nx.ego_graph(..., undirected=True)`, returns Cytoscape element dicts with per-node color/size data
4. **Stylesheet** — node color from `data(color)` (gold = center, yellow = Black, blue = White, green = Mixed, gray = unknown); edge color from `data(color)` keyed by predicate
5. **Layout** — `cose-bilkent` (via `cyto.load_extra_layouts()`), which produces much better spread than the default `cose`
6. **Three callbacks:**
   - `update_person` — handles both text search and node-click navigation via `dash.ctx.triggered_id`
   - `update_graph` — produces Cytoscape elements + info panel from the selected person key and degree slider
   - `hover_info` — updates the hover bar below the graph on `mouseoverNodeData`

## Dependencies

Installed in the shared venv at `../.venv/`:

```
dash >= 4.1.0
dash-cytoscape >= 1.0.2
networkx
pandas
pyarrow   # for Parquet reading
```

## Visual design

- Dark canvas background `#1c1c2e`
- Node colors encode racial classification in the historical records
- Edge colors encode relationship predicate (17 types, see `PREDICATE_COLORS` in `app.py`)
- Node labels placed below nodes at 7px; edge labels at 6px
- Center node is always gold/orange; size encodes degree distance
