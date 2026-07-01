---
title: Open Vocabulary Temporal Change Retrieval
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.18.0"
app_file: app.py
pinned: false
---

# Open Vocabulary Temporal Change Retrieval

A **semantic change search engine** for satellite image time series: describe a land-cover change
in plain language — e.g. *"new buildings on former agricultural land"*, *"forest cleared to bare
soil"* — and retrieve the image **pairs and the timestep** where that change happened, without
training a class-specific detector. Frozen vision–language encoders (CLIP / GeoRSCLIP / RemoteCLIP)
embed each timestep; a bi-temporal change feature is matched against the query text.

## Using this demo

Type a free-text change query (or click an example) and press **Search**. The top match opens with
before/after swipe views and a query-conditioned change heatmap on the later image, alongside its
match score; the remaining matches fill a ranked, exportable grid. The **Settings** panel switches
dataset, encoder, colour mode, and scoring approach; the **About** panel explains each approach and
the honest accuracy limits.

> Research demo — retrieval is approximate. Frozen vision–language change retrieval hits a
> ≈0.20 cross-validated-mAP ceiling, with recovery scaling by how visually salient the change is.

## Full project

- **Code, methodology, and reproducible pipeline** — [GitHub repository](https://github.com/Panagiotis427/MSc_GBDA-OV_Temporal_Change_Retrieval)
- **Technical report (PDF)** — [`report/main.pdf`](https://github.com/Panagiotis427/MSc_GBDA-OV_Temporal_Change_Retrieval/blob/main/report/main.pdf)
