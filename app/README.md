---
title: PyPVRoof
emoji: ☀️
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Estimate tilt, azimuth and installed capacities on-the-fly
---

# pypvroof demo

Tilt, azimuth and installed capacity of rooftop PV systems, measured on the
open surface models (DSM) of European mapping agencies: France, Netherlands,
Flanders, England, North Rhine-Westphalia, Mecklenburg-Vorpommern, Poland,
Norway, Switzerland.

- **Draw on the map:** draw a polygon on a roof; the surface model covering the
  location is queried and the characteristics are computed on the fly. Polygons
  covering several roof planes are split, and the planes drawn back on the map.
- **Upload polygons:** GeoJSON, GeoPackage or GeoParquet; results as a table
  and a GeoJSON to download.

Untick "Extract PV characteristics" to measure roof pans (no installed
capacity).

Run locally from the repository root:

```bash
pip install -e ".[viz]" gradio
python app/app.py
```

To deploy as a Hugging Face Space, copy this folder (with `requirements.txt`
and `examples/`) to the Space repository. The number of polygons per run is
capped (`PYPVROOF_MAX_POLYGONS`, default 200) to be gentle with the public
services.
