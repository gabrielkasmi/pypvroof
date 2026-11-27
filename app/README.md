---
title: pypvroof
emoji: ☀️
colorFrom: blue
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# pypvroof demo

Tilt and azimuth of rooftop PV polygons measured on the IGN elevation models
(LiDAR HD and MNS Correl, 50 cm), with a visual check of each polygon.

Run locally from the repository root:

```bash
pip install -e ".[viz]" gradio
python app/app.py
```

To deploy as a Hugging Face Space, copy this folder (with `requirements.txt`)
to the Space repository. Polygons must be in France (IGN data); the number of
polygons per run is capped (`PYPVROOF_MAX_POLYGONS`, default 200) to be
gentle with the Géoplateforme services.
