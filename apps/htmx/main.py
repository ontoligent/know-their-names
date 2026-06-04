#!/usr/bin/env python3
"""
Know Their Names — FastAPI + HTMX + vis-network network explorer.
Run:  uvicorn main:app --reload
Open: http://localhost:8000
"""
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from data import ALL_KEYS, PERSON, G, ego_vis, parse_key

app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _search_results(q: str, limit: int = 80) -> list[dict]:
    q = q.strip().lower()
    if len(q) < 2:
        return []
    matches = [k for k in ALL_KEYS if q in k][:limit]
    return [{"key": k, **parse_key(k)} for k in matches]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "person_key":  None,
        "info":        None,
        "person_row":  None,
        "nodes":       [],
        "edges":       [],
        "degrees":     2,
        "layout":      "force",
        "use_physics": True,
    })


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = ""):
    results = _search_results(q)
    return templates.TemplateResponse(request, "partials/search_results.html", {
        "results": results,
    })


@app.get("/person/{key:path}", response_class=HTMLResponse)
async def person(request: Request, key: str, degrees: int = 2, layout: str = "force"):
    key  = unquote(key)
    info = parse_key(key)
    p    = PERSON.loc[key] if key in PERSON.index else None
    nodes, edges, use_physics = ego_vis(key, degrees, layout)
    return templates.TemplateResponse(request, "index.html", {
        "person_key":  key,
        "info":        info,
        "person_row":  p,
        "nodes":       nodes,
        "edges":       edges,
        "degrees":     degrees,
        "layout":      layout,
        "use_physics": use_physics,
    })


@app.get("/graph", response_class=HTMLResponse)
async def graph(request: Request, key: str = "", degrees: int = 2, layout: str = "force"):
    key = unquote(key)
    nodes, edges, use_physics = ego_vis(key, degrees, layout) if key else ([], [], True)
    return templates.TemplateResponse(request, "partials/graph.html", {
        "nodes":       nodes,
        "edges":       edges,
        "key":         key,
        "use_physics": use_physics,
    })


@app.get("/info/{key:path}", response_class=HTMLResponse)
async def info(request: Request, key: str):
    key   = unquote(key)
    info_ = parse_key(key)
    p     = PERSON.loc[key] if key in PERSON.index else None
    return templates.TemplateResponse(request, "partials/person_info.html", {
        "person_key": key,
        "info":       info_,
        "person_row": p,
    })
