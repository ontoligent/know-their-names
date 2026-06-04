from pathlib import Path
from collections import defaultdict, deque
import math
import pandas as pd
import networkx as nx

DATA_DIR = Path(__file__).parent / ".." / ".." / "data"

PERSON   = pd.read_parquet(DATA_DIR / "PERSON.parquet")
RELATION = pd.read_parquet(DATA_DIR / "RELATION.parquet").reset_index()

G = nx.from_pandas_edgelist(
    RELATION.sort_values("n_assertions", ascending=False)
            .drop_duplicates(subset=["subject_key", "object_key"]),
    source="subject_key",
    target="object_key",
    edge_attr=["predicate", "n_assertions"],
    create_using=nx.DiGraph(),
)
ALL_KEYS = sorted(G.nodes())

RACE_COLORS   = {"b": "#f1c40f", "w": "#85c1e9", "m": "#82e0aa", "x": "#d7dbdd"}
GENDER_SHAPES = {"m": "triangle", "f": "ellipse", "x": "box"}

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

# Generation delta for directed edges (source → target).
# Positive = target is a later generation (child); negative = earlier (parent).
_GEN_DELTA = {
    "isParentOf":      +1, "IsFatherOf":     +1, "IsMotherOf":    +1,
    "isGrandParentOf": +2, "isGrandfatherOf":+2, "isGrandmotherOf":+2,
    "isChildOf":       -1,
    "isGrandChildOf":  -2,
    # siblings, spouses, cousins, in-laws → same generation (0)
}


def parse_key(key: str) -> dict:
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


def _birth_year_int(key: str) -> int:
    y = parse_key(key)["birth_year"]
    try:
        return int(y)
    except (ValueError, TypeError):
        return 9999


def _genealogical_pos(ego, center_key: str) -> dict:
    """Assign positions: Y = generation (parents above, children below),
    X = spread within generation sorted by birth year."""
    # Build neighbour→delta lookup from directed edges (both directions)
    adj = defaultdict(list)
    for u, v, d in ego.edges(data=True):
        delta = _GEN_DELTA.get(d.get("predicate", ""), 0)
        adj[u].append((v, +delta))
        adj[v].append((u, -delta))

    # BFS from center to assign generation numbers
    gen = {center_key: 0}
    queue = deque([center_key])
    while queue:
        node = queue.popleft()
        for nb, delta in adj[node]:
            if nb not in gen:
                gen[nb] = gen[node] + delta
                queue.append(nb)

    # Group nodes by generation; sort each group by birth year for x ordering
    gen_groups = defaultdict(list)
    for node, g in gen.items():
        gen_groups[g].append(node)
    for g in gen_groups:
        gen_groups[g].sort(key=_birth_year_int)

    Y_STEP, X_STEP = 150, 130
    pos = {}
    for g, nodes_in_gen in gen_groups.items():
        n = len(nodes_in_gen)
        for i, node in enumerate(nodes_in_gen):
            pos[node] = {
                "x": round((i - (n - 1) / 2.0) * X_STEP),
                "y": g * Y_STEP,
            }
    return pos


def _radial_pos(ego, center_key: str) -> dict:
    """Ring layout sorted by birth year."""
    ring   = sorted([n for n in ego.nodes() if n != center_key], key=_birth_year_int)
    n      = len(ring)
    radius = max(150, 7 * n)
    pos    = {center_key: {"x": 0, "y": 0}}
    for i, node in enumerate(ring):
        angle      = 2 * math.pi * i / n - math.pi / 2
        pos[node]  = {"x": round(radius * math.cos(angle)), "y": round(radius * math.sin(angle))}
    return pos


def ego_vis(center_key: str, degrees: int = 2, layout: str = "force") -> tuple[list, list, bool]:
    """Return (nodes, edges, use_physics) in vis-network format."""
    if center_key not in G:
        return [], [], True

    ego      = nx.ego_graph(G, center_key, radius=degrees, undirected=True)
    dist_map = nx.single_source_shortest_path_length(ego.to_undirected(), center_key)

    # Pre-compute preset positions for non-force layouts
    pos = {}
    if layout == "genealogical":
        pos = _genealogical_pos(ego, center_key)
    elif layout == "radial":
        pos = _radial_pos(ego, center_key)

    use_physics = (layout == "force")

    nodes = []
    for node in ego.nodes():
        info      = parse_key(node)
        dist      = dist_map.get(node, degrees)
        is_center = node == center_key
        p         = PERSON.loc[node] if node in PERSON.index else None
        n_ment    = int(p.n_mentions)  if p is not None else 0
        n_rel     = int(p.n_relations) if p is not None else 0

        bg           = RACE_COLORS.get(info["norm_race"], "#d7dbdd")
        border       = "#ffffff" if is_center else "#333"
        border_width = 4        if is_center else 1
        base_size    = 30       if is_center else max(12, 24 - dist * 4)
        shape        = GENDER_SHAPES.get(info["gender"], "square")

        if shape == "triangle":
            size      = round(base_size * 0.75)
            font_size = 14
            vadjust   = -round(size * 2 + 5)
        elif shape == "box":
            size      = round(base_size * 0.50)
            font_size = 10
            vadjust   = 0
        else:  # ellipse
            size      = base_size
            font_size = 10
            vadjust   = 0

        font = {"color": "#1c1c2e", "size": font_size, "bold": True}
        if vadjust:
            font["vadjust"] = vadjust

        tooltip = (
            f"<b>{info['name']}</b><br>"
            f"<span style='color:#aaa;font-size:10px'>{node}</span><br>"
            f"Born: {info['birth_year']} &nbsp;·&nbsp; "
            f"Race: {info['norm_race'].upper()} &nbsp;·&nbsp; "
            f"Gender: {info['gender'].upper()}<br>"
            f"Mentions: {n_ment} &nbsp;·&nbsp; Relations: {n_rel}"
        )

        node_dict = {
            "id":          node,
            "label":       str(n_ment),
            "title":       tooltip,
            "color": {
                "background": bg,
                "border":     border,
                "highlight":  {"background": bg, "border": "#fff"},
            },
            "shape":       shape,
            "borderWidth": border_width,
            "size":        size,
            "font":        font,
        }
        if node in pos:
            node_dict["x"] = pos[node]["x"]
            node_dict["y"] = pos[node]["y"]
        nodes.append(node_dict)

    # Pre-compute which directed pairs have a reverse edge so we can curve them
    bidirectional = frozenset(
        (u, v) for u, v in ego.edges() if ego.has_edge(v, u)
    )

    edges = []
    for u, v, data in ego.edges(data=True):
        pred  = data.get("predicate", "")
        color = PREDICATE_COLORS.get(pred, "#888")
        if (u, v) in bidirectional:
            # Both edges use curvedCW. The reverse edge travels in the opposite
            # direction, so "right of direction of travel" lands on the opposite
            # side of the line — the two arcs naturally bow away from each other.
            smooth = {"enabled": True, "type": "curvedCW", "roundness": 0.2}
        else:
            smooth = {"enabled": False}
        edges.append({
            "from":   u,
            "to":     v,
            "title":  pred,          # hover tooltip only — no canvas label
            "color":  {"color": color, "opacity": 0.85},
            "arrows": "to",
            "width":  1.5,
            "smooth": smooth,
        })

    return nodes, edges, use_physics
