#!/usr/bin/env python3
"""
Person Network Explorer — Streamlit/Plotly app.
Run:  streamlit run app.py
"""
from pathlib import Path
import math
from collections import defaultdict

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# ── Data loading ──────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / ".." / ".." / "data"

@st.cache_resource
def load_data():
    person   = pd.read_parquet(DATA_DIR / "PERSON.parquet")
    relation = pd.read_parquet(DATA_DIR / "RELATION.parquet").reset_index()
    rel_g    = (relation.sort_values("n_assertions", ascending=False)
                        .drop_duplicates(subset=["subject_key", "object_key"]))
    G = nx.from_pandas_edgelist(
        rel_g, source="subject_key", target="object_key",
        edge_attr=["predicate", "n_assertions"],
        create_using=nx.DiGraph(),
    )
    return person, G

PERSON, G = load_data()
ALL_KEYS  = sorted(G.nodes())

# ── Color / shape maps ────────────────────────────────────────────────

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
RACE_COLORS    = {"b": "#f1c40f", "w": "#85c1e9", "m": "#82e0aa", "x": "#d7dbdd"}
PLOTLY_SYMBOLS = {"m": "triangle-up", "f": "circle",  "x": "square"}
RACE_LABELS    = {"b": "Black", "w": "White", "m": "Mixed", "x": "?"}
GENDER_LABELS  = {"m": "M", "f": "F", "x": "?"}

# ── parse_key ─────────────────────────────────────────────────────────

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

# Precompute search index: (key, searchable_text) — includes parsed display name
# so "unnamed" matches people whose key contains only "x" as the name part.
KEY_SEARCH_INDEX = [
    (k, k + " " + parse_key(k)["name"].lower())
    for k in ALL_KEYS
]

# ── Position computation ──────────────────────────────────────────────

def _birth_year_int(node):
    y = parse_key(node)["birth_year"]
    return int(y) if y != "?" else float("inf")


def compute_positions(ego, center_key, layout_type):
    pos = {}
    other = [n for n in ego.nodes() if n != center_key]

    if layout_type == "vertical":
        spring   = nx.spring_layout(ego.to_undirected(), seed=42, k=1.5)
        years    = sorted({_birth_year_int(n) for n in other if _birth_year_int(n) != float("inf")})
        yr_rank  = {yr: i for i, yr in enumerate(years)}
        unk_rank = len(years)
        pos[center_key] = (0, -(unk_rank * 80) / 2)  # center vertically
        for n in other:
            yr     = _birth_year_int(n)
            y_pos  = -(yr_rank[yr] * 80 if yr != float("inf") else unk_rank * 80)
            pos[n] = (spring[n][0] * 200, y_pos)

    elif layout_type == "circular":
        ring   = sorted(other, key=_birth_year_int)
        n_ring = len(ring)
        radius = max(150, 7 * n_ring)
        pos[center_key] = (0, 0)
        for i, n in enumerate(ring):
            angle  = 2 * math.pi * i / n_ring - math.pi / 2
            pos[n] = (radius * math.cos(angle), radius * math.sin(angle))

    elif layout_type == "generational":
        try:
            gens = list(nx.topological_generations(ego))
        except nx.NetworkXUnfeasible:
            gens = [list(ego.nodes())]
        n_gens = len(gens)
        for g_idx, gen in enumerate(gens):
            y_pos = -(g_idx / max(n_gens - 1, 1)) * 400
            for x_idx, n in enumerate(gen):
                x_pos = (x_idx - (len(gen) - 1) / 2) * 80
                pos[n] = (x_pos, y_pos)

    else:  # force-directed
        raw = nx.spring_layout(ego.to_undirected(), seed=42, k=1.5)
        for n, (x, y) in raw.items():
            pos[n] = (x * 300, y * 300)

    return pos


# ── Build Plotly figure ───────────────────────────────────────────────

def build_figure(center_key, degrees, layout_type):
    ego      = nx.ego_graph(G, center_key, radius=degrees, undirected=True)
    dist_map = nx.single_source_shortest_path_length(ego.to_undirected(), center_key)
    pos      = compute_positions(ego, center_key, layout_type)

    # ── Edge traces (grouped by predicate color) ──────────────────────
    edge_groups = defaultdict(lambda: {"x": [], "y": [], "labels": []})
    for u, v, data in ego.edges(data=True):
        pred  = data.get("predicate", "")
        color = PREDICATE_COLORS.get(pred, "#888")
        xu, yu = pos.get(u, (0, 0))
        xv, yv = pos.get(v, (0, 0))
        edge_groups[color]["x"]      += [xu, xv, None]
        edge_groups[color]["y"]      += [yu, yv, None]
        edge_groups[color]["labels"] += [pred, pred, None]

    fig = go.Figure()
    for color, g in edge_groups.items():
        fig.add_trace(go.Scatter(
            x=g["x"], y=g["y"],
            mode="lines",
            line=dict(color=color, width=1.5),
            opacity=0.85,
            hovertemplate="<b>%{customdata}</b><extra></extra>",
            customdata=g["labels"],
            showlegend=False,
        ))

    # ── Node trace ────────────────────────────────────────────────────
    xs, ys          = [], []
    sizes           = []
    colors          = []
    symbols         = []
    border_colors   = []
    border_widths   = []
    texts           = []
    hover_data      = []

    for node in ego.nodes():
        info      = parse_key(node)
        dist      = dist_map.get(node, degrees)
        is_center = node == center_key
        p         = PERSON.loc[node] if node in PERSON.index else None
        x, y      = pos.get(node, (0, 0))

        xs.append(x)
        ys.append(y)
        sizes.append(35 if is_center else max(18, 28 - dist * 5))
        colors.append("#f39c12" if is_center else RACE_COLORS.get(info["norm_race"], "#d7dbdd"))
        symbols.append("star" if is_center else PLOTLY_SYMBOLS.get(info["gender"], "square"))
        border_colors.append("#e67e22" if is_center else "#555")
        border_widths.append(3 if is_center else 1)
        texts.append(str(int(p.n_mentions) if p is not None else 0))
        hover_data.append([
            node,
            info["name"],
            info["birth_year"],
            info["norm_race"].upper(),
            info["gender"].upper(),
            int(p.n_mentions)  if p is not None else 0,
            int(p.n_relations) if p is not None else 0,
        ])

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=colors,
            symbol=symbols,
            line=dict(color=border_colors, width=border_widths),
        ),
        text=texts,
        textposition="middle center",
        textfont=dict(color="white", size=9, family="sans-serif"),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "<span style='color:#888'>%{customdata[0]}</span><br>"
            "Birth: <b>%{customdata[2]}</b><br>"
            "Race: <b>%{customdata[3]}</b>  Gender: <b>%{customdata[4]}</b><br>"
            "Mentions: <b>%{customdata[5]}</b>  Relations: <b>%{customdata[6]}</b>"
            "<extra></extra>"
        ),
        customdata=hover_data,
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor="#1c1c2e",
        plot_bgcolor="#1c1c2e",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        dragmode="pan",
        clickmode="event+select",
        hoverlabel=dict(bgcolor="white", font_color="black", font_family="sans-serif"),
    )
    return fig


# ── Legend HTML helper ────────────────────────────────────────────────

def legend_html():
    parts = ["<div style='font-family:sans-serif;font-size:12px;line-height:1.8'>"]

    parts.append("<b style='font-size:13px'>Node color — Race</b><br>")
    for color, label in [
        ("#f39c12", "Center node"),
        ("#f1c40f", "Black"),
        ("#85c1e9", "White"),
        ("#82e0aa", "Mixed"),
        ("#d7dbdd", "Unknown"),
    ]:
        parts.append(
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"border-radius:50%;background:{color};border:1px solid #aaa;"
            f"vertical-align:middle;margin-right:6px'></span>{label}<br>"
        )

    parts.append("<br><b style='font-size:13px'>Node shape — Gender</b><br>")
    for sym, label in [("▲", "Male"), ("●", "Female"), ("■", "Unknown")]:
        parts.append(f"<span style='margin-right:6px'>{sym}</span>{label}<br>")

    parts.append("<br><b style='font-size:13px'>Edge — Relationship</b><br>")
    for pred, color in PREDICATE_COLORS.items():
        readable = pred[0].lower() + "".join(
            f" {c.lower()}" if c.isupper() else c for c in pred[1:]
        ).strip()
        parts.append(
            f"<span style='display:inline-block;width:18px;height:3px;"
            f"background:{color};vertical-align:middle;margin-right:6px'></span>"
            f"{readable}<br>"
        )

    parts.append("</div>")
    return "".join(parts)


# ── Streamlit app ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Person Network Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "current_person" not in st.session_state:
    st.session_state.current_person = None

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Person Network Explorer")
    st.caption("DS 6105 | Summer 2026 — Know Their Names")

    st.subheader("Search")
    search = st.text_input("Name or key fragment", label_visibility="collapsed",
                           placeholder="Type to search (min 2 chars)…")

    if search and len(search.strip()) >= 2:
        q       = search.strip().lower()
        matches = [k for k, s in KEY_SEARCH_INDEX if q in s][:100]
        if matches:
            chosen = st.selectbox(
                "Results",
                matches,
                format_func=lambda k: (lambda i: (
                f"{i['name']}  "
                f"({i['birth_year']}, "
                f"{RACE_LABELS.get(i['norm_race'], '?')}, "
                f"{GENDER_LABELS.get(i['gender'], '?')})"
            ))(parse_key(k)),
                label_visibility="collapsed",
            )
            if chosen and chosen != st.session_state.current_person:
                st.session_state.current_person = chosen
        else:
            st.caption("No matches.")

    st.divider()
    degrees     = st.slider("Degrees of separation", 1, 4, 2)
    layout_type = st.radio(
        "Layout",
        ["vertical", "circular", "generational", "force"],
        format_func={
            "vertical":     "Vertical (birth year)",
            "circular":     "Circular (birth year)",
            "generational": "Generational (topological)",
            "force":        "Force-directed",
        }.get,
    )

    st.divider()
    st.markdown(legend_html(), unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────

person = st.session_state.current_person

if not person:
    st.info("Search for a person in the sidebar to explore their network.")
else:
    info = parse_key(person)
    p    = PERSON.loc[person] if person in PERSON.index else None

    cols = st.columns([3, 1, 1, 1, 1, 1])
    cols[0].markdown(f"### {info['name']}")
    cols[1].metric("Birth",     info["birth_year"])
    cols[2].metric("Race",      info["norm_race"].upper())
    cols[3].metric("Gender",    info["gender"].upper())
    cols[4].metric("Mentions",  int(p.n_mentions)  if p is not None else "–")
    cols[5].metric("Relations", int(p.n_relations) if p is not None else "–")

    if person not in G:
        st.warning("This person has no recorded relations in the network.")
    else:
        fig   = build_figure(person, degrees, layout_type)
        event = st.plotly_chart(fig, use_container_width=True,
                                on_select="rerun", selection_mode=["points"])

        # Node click → navigate (edge traces have predicate strings in customdata[0],
        # not node keys, so check the clicked key is actually in the graph)
        if event and event.selection and event.selection.points:
            clicked = event.selection.points[0]["customdata"][0]
            if clicked and clicked in G and clicked != person:
                st.session_state.current_person = clicked
                st.rerun()
