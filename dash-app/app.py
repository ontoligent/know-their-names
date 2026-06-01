#!/usr/bin/env python3
"""
Person Network Explorer — Dash/Cytoscape app.
Run:  python app.py
Open: http://localhost:8050
"""
import pandas as pd
import networkx as nx
import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto

cyto.load_extra_layouts()  # enables cose-bilkent, cola, dagre, etc.

# ── Load data ─────────────────────────────────────────────────────────

PERSON = pd.read_parquet("../data/PERSON.parquet")
RELATION = pd.read_parquet("../data/RELATION.parquet").reset_index()

RELATION_g = (
    RELATION.sort_values("n_assertions", ascending=False)
    .drop_duplicates(subset=["subject_key", "object_key"])
)

G = nx.from_pandas_edgelist(
    RELATION_g, source="subject_key", target="object_key",
    edge_attr=["predicate", "n_assertions"],
    create_using=nx.DiGraph(),
)
all_keys = sorted(G.nodes())

# ── Color maps ────────────────────────────────────────────────────────

PREDICATE_COLORS = {
    "wasEnslavedBy":    "#e74c3c",
    "isParentOf":       "#2980b9",
    "isChildOf":        "#5dade2",
    "isSiblingOf":      "#27ae60",
    "isSpouseOf":       "#e91e63",
    "IsFatherOf":       "#1a5276",
    "IsMotherOf":       "#76448a",
    "isGrandParentOf":  "#148f77",
    "isGrandChildOf":   "#1abc9c",
    "isGrandfatherOf":  "#148f77",
    "isGrandmotherOf":  "#76d7c4",
    "isNiblingOf":      "#e67e22",
    "isPiblingOf":      "#ca6f1e",
    "isCousinOf":       "#f39c12",
    "isChildInLawOf":   "#8e44ad",
    "isParentInLawOf":  "#9b59b6",
    "isSiblingInLawOf": "#6c3483",
}
RACE_COLORS = {"b": "#f1c40f", "w": "#85c1e9", "m": "#82e0aa", "x": "#d7dbdd"}
LS_LABELS   = {"e": "enslaved", "f": "free", "x": "unknown"}


def parse_key(key):
    parts = key.split("-", 4)
    if len(parts) < 5:
        return dict(birth_year="?", legal_status="x", norm_race="x", gender="x", name=key)
    raw  = parts[4].replace("_", " ").strip()
    name = " ".join(w.capitalize() for w in raw.split()) if raw not in ("x", "") else "(unnamed)"
    return dict(
        birth_year    = parts[0] if parts[0] != "xxxx" else "?",
        legal_status  = parts[1],
        norm_race     = parts[2],
        gender        = parts[3],
        name          = name,
    )


def ego_elements(center_key, degrees):
    """Return Cytoscape element dicts for the ego network."""
    if center_key not in G:
        return []
    ego      = nx.ego_graph(G, center_key, radius=degrees, undirected=True)
    dist_map = nx.single_source_shortest_path_length(ego.to_undirected(), center_key)

    nodes, edges = [], []
    for node in ego.nodes():
        info      = parse_key(node)
        dist      = dist_map.get(node, degrees)
        is_center = node == center_key
        p         = PERSON.loc[node] if node in PERSON.index else None
        nodes.append({"data": {
            "id":           node,
            "label":        info["name"],
            "color":        "#f39c12" if is_center else RACE_COLORS.get(info["norm_race"], "#d7dbdd"),
            "border_color": "#e67e22" if is_center else "#555",
            "border_width": 3 if is_center else 1,
            "size":         35 if is_center else max(12, 28 - dist * 5),
            "birth_year":   info["birth_year"],
            "legal_status": LS_LABELS.get(info["legal_status"], "?"),
            "norm_race":    info["norm_race"].upper(),
            "gender":       info["gender"].upper(),
            "n_mentions":   int(p.n_mentions)  if p is not None else 0,
            "n_relations":  int(p.n_relations) if p is not None else 0,
        }})

    for u, v, data in ego.edges(data=True):
        pred = data.get("predicate", "")
        edges.append({"data": {
            "source": u, "target": v,
            "label": pred,
            "color": PREDICATE_COLORS.get(pred, "#888"),
        }})

    return nodes + edges


# ── Cytoscape stylesheet ──────────────────────────────────────────────

STYLESHEET = [
    {"selector": "node", "style": {
        "label":              "data(label)",
        "background-color":   "data(color)",
        "border-color":       "data(border_color)",
        "border-width":       "data(border_width)",
        "width":              "data(size)",
        "height":             "data(size)",
        "color":              "#111",
        "text-valign":        "bottom",
        "text-halign":        "center",
        "text-margin-y":      4,
        "font-size":          "7px",
        "font-weight":        "600",
        "text-wrap":          "wrap",
        "text-max-width":     "55px",
        "text-outline-width": 1,
        "text-outline-color": "#fff",
        "cursor":             "pointer",
    }},
    {"selector": "node:selected", "style": {
        "border-width": 4,
        "border-color": "#f39c12",
    }},
    {"selector": "edge", "style": {
        "label":               "data(label)",
        "line-color":          "data(color)",
        "target-arrow-color":  "data(color)",
        "target-arrow-shape":  "triangle",
        "curve-style":         "bezier",
        "font-size":           "6px",
        "color":               "#ccc",
        "text-rotation":       "autorotate",
        "text-margin-y":       -6,
        "text-outline-width":  1,
        "text-outline-color":  "#1c1c2e",
        "arrow-scale":         0.7,
        "width":               1.5,
        "opacity":             0.85,
    }},
]

# ── App layout ────────────────────────────────────────────────────────

PANEL = {"fontFamily": "system-ui, sans-serif", "padding": "0 20px"}
LABEL = {"fontWeight": "600", "marginBottom": "4px", "display": "block", "fontSize": "13px"}

app = dash.Dash(__name__, title="Person Network Explorer")

# Inject: (1) mousemove tracker so the tooltip can be placed at the cursor,
# (2) mouseleave listener on the graph div to hide the tooltip when mouse exits.
app.index_string = """<!DOCTYPE html>
<html>
  <head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}</head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
    <script>
      document.addEventListener('mousemove', function(e) {
        window._mx = e.clientX;
        window._my = e.clientY;
      });
      // Hide node tooltip when the mouse leaves the cytoscape canvas.
      var _ttInterval = setInterval(function() {
        var g = document.getElementById('graph');
        if (g) {
          clearInterval(_ttInterval);
          g.addEventListener('mouseleave', function() {
            var tt = document.getElementById('node-tooltip');
            if (tt) tt.style.display = 'none';
          });
        }
      }, 200);
    </script>
  </body>
</html>"""

app.layout = html.Div([
    html.H2("Person Network Explorer",
            style={"margin": "16px 20px 2px", "color": "#2c3e50"}),
    html.P("DS 6105 | Summer 2026 — Know Their Names  |  "
           "Click any node in the graph to navigate to that person.",
           style={"margin": "0 20px 14px", "color": "#7f8c8d", "fontSize": "13px"}),

    html.Div([
        html.Div([
            html.Label("Search", style=LABEL),
            dcc.Input(id="search", type="text", debounce=True,
                      placeholder="Type name or key fragment (min 2 chars)…",
                      style={"width": "100%", "padding": "6px 8px", "fontSize": "13px",
                             "border": "1px solid #ccc", "borderRadius": "4px",
                             "boxSizing": "border-box"}),
        ], style={"flex": "2", "marginRight": "24px"}),
        html.Div([
            html.Label("Degrees of separation", style=LABEL),
            dcc.Slider(id="degrees", min=1, max=4, step=1, value=2,
                       marks={i: str(i) for i in range(1, 5)}),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "alignItems": "flex-end", **PANEL, "marginBottom": "12px"}),

    html.Div([
        html.Label("Person", style=LABEL),
        dcc.Dropdown(id="person", options=[], placeholder="Select a person…",
                     style={"fontSize": "13px"}),
    ], style={**PANEL, "marginBottom": "12px"}),

    html.Div(id="info", style={
        "margin": "0 20px 12px", "padding": "8px 14px",
        "background": "#f0f3f4", "borderRadius": "6px",
        "fontSize": "13px", "minHeight": "34px",
    }),

    cyto.Cytoscape(
        id="graph",
        elements=[],
        layout={
            "name":                        "cose-bilkent",
            "animate":                     "end",
            "animationDuration":           600,
            "randomize":                   True,
            "nodeDimensionsIncludeLabels": True,
            "idealEdgeLength":             120,
            "nodeRepulsion":               6000,
            "edgeElasticity":              0.45,
            "gravity":                     0.15,
            "componentSpacing":            80,
            "numIter":                     2500,
            "tile":                        True,
            "tilingPaddingVertical":       20,
            "tilingPaddingHorizontal":     20,
        },
        style={"width": "100%", "height": "580px", "background": "#1c1c2e"},
        stylesheet=STYLESHEET,
        responsive=True,
        minZoom=0.2,
        maxZoom=3.0,
    ),

    html.Div(id="hover", style={
        "margin": "6px 20px 20px", "padding": "6px 14px",
        "background": "#f8f9fa", "borderRadius": "5px",
        "fontSize": "12px", "color": "#555", "minHeight": "26px",
    }),

    # Floating node tooltip — shown/hidden entirely via clientside JS.
    html.Div(id="node-tooltip", style={
        "position":     "fixed",
        "display":      "none",
        "background":   "rgba(28,28,46,0.96)",
        "color":        "#ecf0f1",
        "padding":      "10px 14px",
        "borderRadius": "5px",
        "border":       "1px solid #556",
        "fontSize":     "12px",
        "lineHeight":   "1.7",
        "pointerEvents":"none",
        "zIndex":       1000,
        "maxWidth":     "300px",
        "boxShadow":    "0 3px 12px rgba(0,0,0,0.55)",
    }),

    # Dummy store — required as Output target for the clientside tooltip callback.
    dcc.Store(id="_tt"),
], style={"maxWidth": "1100px", "margin": "0 auto"})


# ── Callbacks ─────────────────────────────────────────────────────────

@app.callback(
    Output("person", "options"),
    Output("person", "value"),
    Input("search", "value"),
    Input("graph", "tapNodeData"),
    State("person", "value"),
)
def update_person(search, tap, current):
    trigger = dash.ctx.triggered_id

    if trigger == "graph" and tap:
        node_id = tap["id"]
        return [{"label": node_id, "value": node_id}], node_id

    if not search or len(search.strip()) < 2:
        return [], dash.no_update

    q       = search.strip().lower()
    matches = [k for k in all_keys if q in k][:100]
    opts    = [{"label": k, "value": k} for k in matches]
    value   = current if current in matches else (matches[0] if matches else None)
    return opts, value


@app.callback(
    Output("graph", "elements"),
    Output("info", "children"),
    Input("person", "value"),
    Input("degrees", "value"),
)
def update_graph(key, degrees):
    if not key:
        return [], "Select a person above to explore their network."

    info = parse_key(key)
    p    = PERSON.loc[key] if key in PERSON.index else None

    summary = html.Span([
        html.B(info["name"], style={"fontSize": "15px"}), "  ",
        "Birth: ",    html.B(info["birth_year"]),                               "  |  ",
        "Status: ",   html.B(LS_LABELS.get(info["legal_status"], "?")),         "  |  ",
        "Race: ",     html.B(info["norm_race"].upper()),                         "  |  ",
        "Gender: ",   html.B(info["gender"].upper()),                            "  |  ",
        "Mentions: ", html.B(str(int(p.n_mentions))  if p is not None else "–"),"  |  ",
        "Relations: ",html.B(str(int(p.n_relations)) if p is not None else "–"),
    ])

    if key not in G:
        return [], html.Span([summary,
            html.Span("  — no recorded relations.", style={"color": "#e67e22"})])

    return ego_elements(key, degrees), summary


# Edge hover → bottom bar.
@app.callback(
    Output("hover", "children"),
    Input("graph", "mouseoverEdgeData"),
)
def hover_edge(data):
    if not data:
        return "Hover over a node for a detail popup; hover over an edge to see the relationship here."
    return html.Span([
        html.B(data.get("label", "relation")), "  —  ",
        data.get("source", ""), "  →  ", data.get("target", ""),
    ])


# Node hover → floating tooltip, positioned at cursor via clientside JS.
# dash-cytoscape exposes no mouseout event, so hiding is handled by the
# mouseleave listener on the graph div injected via app.index_string.
app.clientside_callback(
    """
    function(nodeData) {
        var tt = document.getElementById('node-tooltip');
        if (!nodeData) { tt.style.display = 'none'; return null; }

        var x = window._mx || 100, y = window._my || 100;
        // Keep tooltip inside the right edge of the viewport
        tt.style.display = 'block';
        tt.style.left = (x + 18) + 'px';
        tt.style.top  = (y + 18) + 'px';

        tt.innerHTML =
            '<b style="font-size:13px">' + (nodeData.label || '') + '</b><br>' +
            '<span style="color:#aaa;font-size:10px">' + (nodeData.id || '') + '</span><br>' +
            'Birth: <b>' + (nodeData.birth_year   || '?') + '</b>' +
            '&nbsp;&nbsp;Status: <b>' + (nodeData.legal_status || '?') + '</b><br>' +
            'Race: <b>' + (nodeData.norm_race || '?') + '</b>' +
            '&nbsp;&nbsp;Gender: <b>' + (nodeData.gender || '?') + '</b><br>' +
            'Mentions: <b>' + (nodeData.n_mentions  || 0) + '</b>' +
            '&nbsp;&nbsp;Relations: <b>' + (nodeData.n_relations || 0) + '</b>';

        return null;
    }
    """,
    Output("_tt", "data"),
    Input("graph", "mouseoverNodeData"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    print("Person Network Explorer → http://localhost:8050")
    app.run(debug=False, port=8050)
