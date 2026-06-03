#!/usr/bin/env python3
"""
Person Network Explorer — Dash/Cytoscape app.
Run:  python app.py
Open: http://localhost:8050
"""
from pathlib import Path
import math
import pandas as pd
import networkx as nx
import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto

DATA_DIR = Path(__file__).parent / ".." / ".." / "data"

cyto.load_extra_layouts()  # enables cose-bilkent, cola, dagre, etc.

# ── Load data ─────────────────────────────────────────────────────────

PERSON = pd.read_parquet(DATA_DIR / "PERSON.parquet")
RELATION = pd.read_parquet(DATA_DIR / "RELATION.parquet").reset_index()

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
RACE_COLORS   = {"b": "#f1c40f", "w": "#85c1e9", "m": "#82e0aa", "x": "#d7dbdd"}
GENDER_SHAPES = {"m": "triangle", "f": "ellipse", "x": "rectangle"}


def parse_key(key):
    parts = key.split("-", 3)
    if len(parts) < 4:
        return dict(birth_year="?", norm_race="x", gender="x", name=key)
    raw  = parts[3].replace("_", " ").strip()
    name = " ".join(w.capitalize() for w in raw.split()) if raw not in ("x", "") else "(unnamed)"
    return dict(
        birth_year = parts[0] if parts[0] != "xxxx" else "?",
        norm_race  = parts[1],
        gender     = parts[2],
        name       = name,
    )


CYTO_LAYOUTS = {
    "vertical": {"name": "preset", "fit": True, "padding": 40},
    "circular": {"name": "preset", "fit": True, "padding": 40},
    "dagre":    {"name": "dagre", "rankDir": "TB", "rankSep": 80,
                 "nodeSep": 60, "fit": True, "padding": 40},
    "force":    {"name": "cose-bilkent", "animate": "end",
                 "animationDuration": 600, "randomize": True,
                 "nodeRepulsion": 6000, "idealEdgeLength": 120,
                 "fit": True, "padding": 40},
}

def ego_elements(center_key, degrees, layout_type="vertical"):
    """Return Cytoscape element dicts for the ego network."""
    if center_key not in G:
        return []
    ego      = nx.ego_graph(G, center_key, radius=degrees, undirected=True)
    dist_map = nx.single_source_shortest_path_length(ego.to_undirected(), center_key)

    def birth_year_int(n):
        y = parse_key(n)["birth_year"]
        return int(y) if y != "?" else float("inf")

    # Pre-compute positions for preset layouts.
    pos = {}
    if layout_type == "vertical":
        spring_pos   = nx.spring_layout(ego.to_undirected(), seed=42, k=1.5)
        other_nodes  = [n for n in ego.nodes() if n != center_key]
        years        = sorted({birth_year_int(n) for n in other_nodes if birth_year_int(n) != float("inf")})
        year_rank    = {yr: i for i, yr in enumerate(years)}
        unknown_rank = len(years)
        Y_STEP       = 80
        pos[center_key] = (0, 0)
        for n in other_nodes:
            yr    = birth_year_int(n)
            y_pos = year_rank[yr] * Y_STEP if yr != float("inf") else unknown_rank * Y_STEP
            pos[n] = (spring_pos[n][0] * 200, y_pos)

    elif layout_type == "circular":
        ring_nodes = sorted([n for n in ego.nodes() if n != center_key], key=birth_year_int)
        n_ring     = len(ring_nodes)
        radius     = max(150, 7 * n_ring)
        pos[center_key] = (0, 0)
        for i, n in enumerate(ring_nodes):
            angle  = 2 * math.pi * i / n_ring - math.pi / 2
            pos[n] = (radius * math.cos(angle), radius * math.sin(angle))

    nodes, edges = [], []
    for node in ego.nodes():
        info      = parse_key(node)
        dist      = dist_map.get(node, degrees)
        is_center = node == center_key
        p         = PERSON.loc[node] if node in PERSON.index else None

        elem = {"data": {
            "id":           node,
            "label":        str(int(p.n_mentions) if p is not None else 0),
            "name":         info["name"],
            "color":        "#f39c12" if is_center else RACE_COLORS.get(info["norm_race"], "#d7dbdd"),
            "shape":        GENDER_SHAPES.get(info["gender"], "rectangle"),
            "border_color": "#e67e22" if is_center else "#555",
            "border_width": 3 if is_center else 1,
            "size":         35 if is_center else max(18, 28 - dist * 5),
            "birth_year":   info["birth_year"],
            "norm_race":    info["norm_race"].upper(),
            "gender":       info["gender"].upper(),
            "n_mentions":   int(p.n_mentions)  if p is not None else 0,
            "n_relations":  int(p.n_relations) if p is not None else 0,
        }}
        if node in pos:
            elem["position"] = {"x": pos[node][0], "y": pos[node][1]}
        nodes.append(elem)

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
        "shape":              "data(shape)",
        "background-color":   "data(color)",
        "border-color":       "data(border_color)",
        "border-width":       "data(border_width)",
        "width":              "data(size)",
        "height":             "data(size)",
        "color":              "#fff",
        "text-valign":        "center",
        "text-halign":        "center",
        "text-margin-y":      0,
        "font-size":          "9px",
        "font-weight":        "700",
        "text-outline-width": 0,
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
            var hv = document.getElementById('hover');
            if (hv) hv.style.display = 'none';
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
            dcc.Input(id="search", type="text",
                      placeholder="Type name or key fragment (min 2 chars)…",
                      style={"width": "100%", "padding": "6px 8px", "fontSize": "13px",
                             "border": "1px solid #ccc", "borderRadius": "4px",
                             "boxSizing": "border-box", "marginBottom": "4px"}),
            dcc.RadioItems(id="person", options=[], value=None,
                           inputStyle={"display": "none"},
                           labelStyle={"display": "block", "padding": "5px 10px",
                                       "cursor": "pointer", "borderBottom": "1px solid #eee",
                                       "fontSize": "13px"},
                           style={"display": "none"}),
        ], style={"flex": "2", "marginRight": "24px"}),
        html.Div([
            html.Label("Degrees of separation", style=LABEL),
            dcc.Slider(id="degrees", min=1, max=4, step=1, value=2,
                       marks={i: str(i) for i in range(1, 5)}),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "alignItems": "flex-start", **PANEL, "marginBottom": "12px"}),

    html.Div(id="info", style={
        "margin": "0 20px 12px", "padding": "8px 14px",
        "background": "#f0f3f4", "borderRadius": "6px",
        "fontSize": "13px", "minHeight": "34px",
    }),

    html.Div([
        html.Label("Layout", style={**LABEL, "display": "inline", "marginRight": "12px"}),
        dcc.RadioItems(
            id="layout-type",
            options=[
                {"label": "Vertical (birth year)", "value": "vertical"},
                {"label": "Circular (birth year)", "value": "circular"},
                {"label": "Generational (dagre)",  "value": "dagre"},
                {"label": "Force-directed",         "value": "force"},
            ],
            value="vertical",
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "18px", "fontSize": "13px"},
        ),
    ], style={**PANEL, "marginBottom": "8px"}),

    html.Div([

        cyto.Cytoscape(
            id="graph",
            elements=[],
            layout={"name": "preset", "fit": True, "padding": 40},
            style={"flex": "1", "height": "580px", "background": "#1c1c2e"},
            stylesheet=STYLESHEET,
            responsive=True,
            minZoom=0.2,
            maxZoom=3.0,
        ),

        # ── Legend ────────────────────────────────────────────────────
        html.Div([
            html.Div("Legend", style={"fontWeight": "700", "fontSize": "13px",
                                      "marginBottom": "10px"}),

            html.Div("Node color — Race", style={"fontWeight": "600", "fontSize": "11px",
                                                  "color": "#666", "marginBottom": "5px"}),
            *[html.Div([
                html.Span(style={"display": "inline-block", "width": "14px", "height": "14px",
                                 "borderRadius": "50%", "background": color,
                                 "marginRight": "7px", "verticalAlign": "middle",
                                 "border": "1px solid #aaa"}),
                html.Span(label, style={"fontSize": "12px", "verticalAlign": "middle"}),
            ], style={"marginBottom": "4px"}) for color, label in [
                ("#f39c12", "Center node"),
                ("#f1c40f", "Black"),
                ("#85c1e9", "White"),
                ("#82e0aa", "Mixed"),
                ("#d7dbdd", "Unknown"),
            ]],

            html.Div("Node shape — Gender", style={"fontWeight": "600", "fontSize": "11px",
                                                    "color": "#666", "margin": "10px 0 5px"}),
            *[html.Div([
                html.Span(symbol, style={"fontSize": "15px", "marginRight": "7px",
                                         "color": "#444", "verticalAlign": "middle"}),
                html.Span(label, style={"fontSize": "12px", "verticalAlign": "middle"}),
            ], style={"marginBottom": "4px"}) for symbol, label in [
                ("▲", "Male"),
                ("●", "Female"),
                ("■", "Unknown"),
            ]],

            html.Div("Edge color — Relation", style={"fontWeight": "600", "fontSize": "11px",
                                                      "color": "#666", "margin": "10px 0 5px"}),
            *[html.Div([
                html.Span(style={"display": "inline-block", "width": "20px", "height": "3px",
                                 "background": color, "marginRight": "7px",
                                 "verticalAlign": "middle"}),
                html.Span(label, style={"fontSize": "11px", "verticalAlign": "middle"}),
            ], style={"marginBottom": "4px"}) for color, label in [
                ("#e74c3c", "Was Enslaved By"),
                ("#2980b9", "Is Parent Of"),
                ("#5dade2", "Is Child Of"),
                ("#1a5276", "Is Father Of"),
                ("#76448a", "Is Mother Of"),
                ("#27ae60", "Is Sibling Of"),
                ("#e91e63", "Is Spouse Of"),
                ("#148f77", "Is Grandparent Of"),
                ("#1abc9c", "Is Grandchild Of"),
                ("#76d7c4", "Is Grandmother Of"),
                ("#e67e22", "Is Nibling Of"),
                ("#ca6f1e", "Is Pibling Of"),
                ("#f39c12", "Is Cousin Of"),
                ("#8e44ad", "Is Child-in-Law Of"),
                ("#9b59b6", "Is Parent-in-Law Of"),
                ("#6c3483", "Is Sibling-in-Law Of"),
            ]],

        ], style={
            "width": "160px", "minWidth": "160px", "marginLeft": "10px",
            "padding": "12px 14px", "background": "#f8f9fa",
            "borderRadius": "6px", "overflowY": "auto", "maxHeight": "580px",
            "fontFamily": "sans-serif",
        }),

    ], style={"display": "flex", "margin": "0 20px"}),

    html.Div(id="hover", style={"display": "none"}),

    # Floating node tooltip — shown/hidden entirely via clientside JS.
    html.Div(id="node-tooltip", style={
        "position":     "fixed",
        "display":      "none",
        "background":   "#ffffff",
        "color":        "#000",
        "padding":      "10px 14px",
        "borderRadius": "5px",
        "border":       "1px solid #ccc",
        "fontSize":     "12px",
        "fontFamily":   "sans-serif",
        "lineHeight":   "1.7",
        "pointerEvents":"none",
        "zIndex":       1000,
        "maxWidth":     "300px",
        "boxShadow":    "0 3px 12px rgba(0,0,0,0.15)",
    }),

    dcc.Store(id="_tt"),
    dcc.Store(id="_hv"),
], style={"maxWidth": "1100px", "margin": "0 auto"})


# ── Callbacks ─────────────────────────────────────────────────────────

RESULTS_STYLE = {
    "maxHeight": "200px", "overflowY": "auto",
    "border": "1px solid #ccc", "borderRadius": "4px",
    "background": "#fff",
}

@app.callback(
    Output("person", "options"),
    Output("person", "value"),
    Output("person", "style"),
    Input("search", "value"),
    Input("graph", "tapNodeData"),
)
def update_person(search, tap):
    trigger = dash.ctx.triggered_id

    if trigger == "graph" and tap:
        return dash.no_update, tap["id"], dash.no_update

    if not search or len(search.strip()) < 2:
        return [], None, {"display": "none"}

    q       = search.strip().lower()
    matches = [k for k in all_keys if q in k][:100]
    return [{"label": k, "value": k} for k in matches], None, RESULTS_STYLE


@app.callback(
    Output("graph", "elements"),
    Output("graph", "layout"),
    Output("info", "children"),
    Input("person", "value"),
    Input("degrees", "value"),
    Input("layout-type", "value"),
)
def update_graph(key, degrees, layout_type):
    cyto_layout = CYTO_LAYOUTS.get(layout_type, CYTO_LAYOUTS["vertical"])

    if not key:
        return [], cyto_layout, "Select a person above to explore their network."

    info = parse_key(key)
    p    = PERSON.loc[key] if key in PERSON.index else None

    summary = html.Span([
        html.B(info["name"], style={"fontSize": "15px"}), "  ",
        "Birth: ",    html.B(info["birth_year"]),                                "  |  ",
        "Race: ",     html.B(info["norm_race"].upper()),                          "  |  ",
        "Gender: ",   html.B(info["gender"].upper()),                             "  |  ",
        "Mentions: ", html.B(str(int(p.n_mentions))  if p is not None else "–"), "  |  ",
        "Relations: ",html.B(str(int(p.n_relations)) if p is not None else "–"),
    ])

    if key not in G:
        return [], cyto_layout, html.Span([summary,
            html.Span("  — no recorded relations.", style={"color": "#e67e22"})])

    return ego_elements(key, degrees, layout_type), cyto_layout, summary


# Edge hover → bottom bar via clientside JS (keeps style out of React state).
app.clientside_callback(
    """
    function(edgeData) {
        var hv = document.getElementById('hover');
        if (!hv) return null;
        if (!edgeData) { hv.style.display = 'none'; return null; }
        hv.style.display = 'block';
        hv.innerHTML = '<b>' + (edgeData.label || 'relation') + '</b>' +
                       '&nbsp;&nbsp;—&nbsp;&nbsp;' +
                       (edgeData.source || '') +
                       '&nbsp;&nbsp;→&nbsp;&nbsp;' +
                       (edgeData.target || '');
        return null;
    }
    """,
    Output("_hv", "data"),
    Input("graph", "mouseoverEdgeData"),
    prevent_initial_call=True,
)


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
            '<b style="font-size:13px;color:#000">' + (nodeData.name || '') + '</b><br>' +
            '<span style="color:#666;font-size:10px">' + (nodeData.id || '') + '</span><br>' +
            'Birth: <b>' + (nodeData.birth_year || '?') + '</b><br>' +
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
