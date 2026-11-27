"""
pypvroof demo: tilt and azimuth of PV polygons from the IGN elevation models.

    pip install -e ".[viz]" gradio
    python app/app.py

Upload or paste polygons (GeoJSON, GeoPackage, GeoParquet, in France), choose
the elevation model, and get a table of results, a downloadable GeoJSON and a
visual check (orthophoto + elevation models) for each polygon.
"""

import json
import os
import tempfile

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pypvroof import characterize_roofs
from pypvroof.geometry import to_geodataframe
from pypvroof.viz import plot_polygon

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples", "exemples.geojson")
MAX_POLYGONS = int(os.environ.get("PYPVROOF_MAX_POLYGONS", 200))  # be gentle with the IGN services

SERVICES = {"LiDAR HD (IGN)": ["ign"], "MNS Correl (IGN)": ["ign-correl"],
            "Les deux": ["ign", "ign-correl"]}
SUFFIX = {"ign": "lidar", "ign-correl": "correl"}
SHOWN = ["tilt", "azimuth", "status", "inspect", "surface_proj", "surface", "offset_suspected"]


def load(file, text):
    if file is not None:
        return to_geodataframe(file if isinstance(file, str) else file.name)
    if text and text.strip():
        return to_geodataframe(json.loads(text))
    return to_geodataframe(EXAMPLES)


def run(file, text, service_label, split):
    gdf = load(file, text)
    if len(gdf) > MAX_POLYGONS:
        raise gr.Error(f"{len(gdf)} polygones : la démo est limitée à {MAX_POLYGONS}.")
    if "id" not in gdf.columns:
        gdf["id"] = [str(i) for i in range(len(gdf))]
    gdf["id"] = gdf["id"].astype(str)
    specs = SERVICES[service_label]

    merged = None
    for spec in specs:
        out = characterize_roofs(gdf, spec, split=split and len(specs) == 1, workers=8, id_column="id")
        if len(specs) == 1:
            merged = out
            break
        cols = out[["parent_id"] + SHOWN].rename(columns={c: f"{c}_{SUFFIX[spec]}" for c in SHOWN})
        merged = cols if merged is None else merged.merge(cols, on="parent_id")
    if len(specs) > 1:
        merged = gdf.merge(merged, left_on="id", right_on="parent_id")

    table = pd.DataFrame(merged.drop(columns=[merged.geometry.name, "modes"], errors="ignore"))
    keep = [c for c in table.columns if c in ("parent_id", "part", "description")
            or any(c.startswith(s) for s in SHOWN) or c == "meta_date_fin_acquisition"]
    table = table[keep].round(2)

    path = os.path.join(tempfile.mkdtemp(), "pypvroof_resultats.geojson")
    export = merged.drop(columns=["modes"], errors="ignore")
    export.to_file(path, driver="GeoJSON")

    ids = list(gdf["id"])
    fig = draw(ids[0], gdf.to_json(), service_label) if ids else None
    return table, path, gr.update(choices=ids, value=ids[0] if ids else None), gdf.to_json(), fig


def draw(polygon_id, gdf_json, service_label):
    if not polygon_id or not gdf_json:
        return None
    gdf = to_geodataframe(json.loads(gdf_json))
    row = gdf[gdf["id"].astype(str) == str(polygon_id)]
    if row.empty:
        return None
    plt.close("all")
    return plot_polygon(row, services=SERVICES[service_label])


with gr.Blocks(title="pypvroof") as demo:
    gr.Markdown(
        "# pypvroof — tilt et azimut des installations PV\n"
        "Inclinaison et orientation du pan de toit sous chaque polygone, mesurées sur les modèles "
        "numériques de surface de l'IGN (LiDAR HD ou MNS Correl, 50 cm). Polygones en France, "
        f"{MAX_POLYGONS} au maximum. Sans fichier, les exemples intégrés sont utilisés. "
        "Azimut : 0 = nord, sens horaire.")
    with gr.Row():
        with gr.Column(scale=1):
            file = gr.File(label="Polygones (GeoJSON, GeoPackage, GeoParquet)",
                           file_types=[".geojson", ".json", ".gpkg", ".parquet"])
            text = gr.Textbox(label="… ou coller un GeoJSON", lines=4)
            service = gr.Radio(list(SERVICES), value="Les deux", label="Modèle de surface")
            split = gr.Checkbox(label="Découper les polygones sur plusieurs pans (un seul modèle)")
            go = gr.Button("Estimer", variant="primary")
        with gr.Column(scale=2):
            table = gr.Dataframe(label="Résultats", wrap=True)
            download = gr.File(label="Résultats (GeoJSON)")
    with gr.Row():
        pick = gr.Dropdown(label="Polygone à visualiser", choices=[])
    plot = gr.Plot(label="Orthophoto et modèles de surface")
    state = gr.State()

    go.click(run, [file, text, service, split], [table, download, pick, state, plot])
    pick.change(draw, [pick, state, service], plot)

if __name__ == "__main__":
    demo.launch()
