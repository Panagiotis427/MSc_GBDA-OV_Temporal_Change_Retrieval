"""
Semantic Change Search Engine — Gradio frontend (rewired for real change
retrieval).

A natural-language query is scored against every bi-temporal pair's *change*
representation (not single images), returning a ranked list of change events.
Each result shows the actual T1/T2 tiles of the matched pair, a
query-conditioned change heatmap (per-patch Δ-similarity T1→T2) on the After
tile, a relative match score, and a seasonal-vs-permanent note grounded in the dataset's
labels.

Selectors: Dataset, Encoder (CLIP / GeoRSCLIP / RemoteCLIP), Approach
(naive / zero_shot Δ-similarity / peft adapter).

CLI:
    python -m src.app --dataset dynamic_earthnet --root data/DynamicEarthNet \
        --encoder clip_vitl14 --approach zero_shot
"""
from __future__ import annotations

import argparse
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from src.datasets.registry import get_dataset
from src.embeddings import load_or_compute
from src.encoders import get_encoder
from src.retrieval import APPROACHES, ChangeRetriever
from src.heatmap import generate_change_heatmap, generate_heatmap
from src.rerank import RERANK_STRATEGIES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SNOW = "snow_and_ice"


def _app_datasets() -> list:
    """Datasets offered in the app UI: those with a registered query set, so
    label-grounded retrieval, relevance and the seasonal note all work. Any
    loader without a query set (no labels / no benchmarked data) is excluded,
    since it would otherwise error if picked.
    """
    from src.queries import get_queries  # importing the package registers the sets
    from src.datasets.registry import list_datasets
    return [d for d in list_datasets() if get_queries(d)]


# Friendly display labels for the dataset / encoder dropdowns. The value sent to
# the callbacks stays the registry key; only the shown label changes. Unknown
# keys fall back to the raw key so a newly-registered dataset/encoder still appears.
DATASET_LABELS = {
    "dynamic_earthnet": "Dynamic EarthNet",
    "levir_cc": "LEVIR-CC",
    "qfabric_teo": "QFabric — change type",
    "qfabric_status": "QFabric — construction status",
}
ENCODER_LABELS = {
    "clip_vitl14": "CLIP ViT-L/14",
    "georsclip": "GeoRSCLIP",
    "remoteclip": "RemoteCLIP",
}


def _labeled(choices, labels):
    """``[(display_label, value)]`` for a Dropdown; value stays the registry key."""
    return [(labels.get(c, c), c) for c in choices]


@dataclass
class RunConfig:
    dataset: str = "dynamic_earthnet"
    # Defaults match the REPORT headline config (GeoRSCLIP + NRG zero-shot on the
    # held-out test split — the best generalising setup, §7.3). Earlier defaults
    # (clip_vitl14 / rgb / split=train) diverged from every reported number and,
    # worse, split=train is the corpus the PEFT adapter was fit on (memorisation).
    encoder: str = "georsclip"
    # patch = localised per-patch Delta scoring, the best DEN configuration
    # (REPORT Appendix B.10, CV mAP 0.193 vs ~0.10 for global zero-shot). The
    # first query lazily encodes per-patch embeddings for the loaded corpus.
    approach: str = "patch"
    root: str = str(_PROJECT_ROOT / "data" / "DynamicEarthNet")
    pairing: str = "bimonthly"
    split: Optional[str] = "test"    # 110 held-out DEN pairs (not the PEFT-fit train corpus)
    cache_dir: str = str(_PROJECT_ROOT / "data" / "cache")
    feature_mode: str = "difference"
    top_k: int = 5
    # Spectral / encoder options (require Apply to reload embeddings)
    color_mode: str = "nrg"         # rgb | nrg | ndvi
    use_lora: bool = False          # load LoRA-adapted embeddings (must be pre-cached)
    # Extension toggles (take effect on next Search; no Apply needed)
    geo_filter: bool = False        # enable geographic region filtering
    rerank: bool = False            # enable post-retrieval re-ranking
    rerank_strategy: str = "diversity"  # diversity | coherence


@dataclass
class ChangeEvent:
    rank: int
    pair_id: str
    location: str
    t1_key: str
    t2_key: str
    score: float
    confidence: float
    caption: str
    seasonal_note: str
    t1_img: Optional[Image.Image]
    t2_img: Optional[Image.Image]
    heatmap: Optional[Image.Image]


class SemanticChangeSearch:
    _CSS = """
        .top-card {border: 1px solid var(--border-color-primary, #d0d0d0);
                   border-radius: 12px; padding: 14px 18px;
                   background: var(--background-fill-secondary, #fafafa);}
        .conf-pill {display:inline-block; padding:4px 12px; border-radius:999px;
                    background:#2e7d32; color:white; font-weight:600;}
        .conf-pill.mid {background:#f9a825; color:#222;}
        .conf-pill.low {background:#c62828;}
        #search-btn { min-height: 3.5rem !important; font-size: 1.1rem !important;
                      letter-spacing: 0.03em; }
        .stats-card {border:1px solid var(--border-color-primary,#d0d0d0);
                     border-left:4px solid #1565c0;
                     border-radius:10px; padding:10px 14px;
                     background:var(--background-fill-secondary,#f5f9ff);
                     font-size:0.95em; line-height:1.7;}
        .stats-card code {background:rgba(21,101,192,0.08); padding:1px 6px;
                          border-radius:4px; font-weight:600;}
        .stats-err {border-left-color:#c62828; background:#fff4f4;}
        """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._build(cfg)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_components(cls, dataset, encoder, store, cfg: RunConfig,
                        adapter=None) -> "SemanticChangeSearch":
        self = cls.__new__(cls)
        self.cfg = cfg
        self.dataset = dataset
        self.encoder = encoder
        self.store = store
        self.retriever = ChangeRetriever(store, encoder,
                                         feature_mode=cfg.feature_mode)
        self._adapter_feature_mode = cfg.feature_mode
        if adapter is not None:
            self.retriever.set_adapter(adapter, cfg.feature_mode)
        self._adapter = adapter
        self._patch_t1 = self._patch_t2 = None  # lazy patch-embedding cache
        self._init_extensions(cfg)
        return self

    def _build(self, cfg: RunConfig) -> None:
        from src.datasets.registry import build_dataset
        from src.embeddings import cache_path as _cache_path
        from src.embeddings import cache_tag_for
        self.dataset = build_dataset(cfg.dataset, root=cfg.root,
                                     pairing=cfg.pairing, split=cfg.split,
                                     color_mode=cfg.color_mode)
        self.encoder = get_encoder(cfg.encoder)

        # Canonical cache tag (single source of truth in src.embeddings) so the
        # pre-computed run_pipeline caches are reused. split=None ("all") -> "all".
        split_str = cfg.split or "all"
        cache_tag = cache_tag_for(split_str, cfg.color_mode, cfg.use_lora)

        if cfg.use_lora:
            lora_cache = _cache_path(cfg.cache_dir, self.dataset.name,
                                     self.encoder.name, tag=cache_tag)
            if not lora_cache.exists():
                raise RuntimeError(
                    f"LoRA embedding cache not found: {lora_cache.name}\n"
                    "Train LoRA first:  python -m scripts.run_pipeline --lora ...\n"
                    "Then re-open the app with 'Use LoRA embeddings' enabled."
                )

        self.store = load_or_compute(self.dataset, self.encoder,
                                     cache_dir=cfg.cache_dir,
                                     cache_tag=cache_tag)
        self.retriever = ChangeRetriever(self.store, self.encoder,
                                         feature_mode=cfg.feature_mode)
        self._adapter = self._maybe_load_adapter(cfg)
        if self._adapter is not None:
            self.retriever.set_adapter(self._adapter, self._adapter_feature_mode)
        self._patch_t1 = self._patch_t2 = None  # invalidate lazy patch cache on (re)build
        self._init_extensions(cfg)

    def _init_extensions(self, cfg: RunConfig) -> None:
        """Load geo-filter and reranker if aoi_metadata.json is present."""
        meta_path = Path(cfg.root) / "aoi_metadata.json"
        if meta_path.exists():
            from src.geo_filter import GeoFilter
            from src.rerank import Reranker
            self._geo_filter = GeoFilter(meta_path)
            self._reranker = Reranker(meta_path)
        else:
            self._geo_filter = None
            self._reranker = None

    def _maybe_load_adapter(self, cfg: RunConfig):
        # Adapter filename must match run_pipeline's save convention, which is
        # colour-tagged: models/<dataset>__<encoder>[_<color>]__adapter.pt.
        # Without the colour tag the app would silently apply the RGB-trained
        # adapter to NRG/NDVI embeddings (a dim-compatible but wrong head).
        color_tag = f"_{cfg.color_mode}" if cfg.color_mode != "rgb" else ""
        fname = f"{cfg.dataset}__{cfg.encoder}{color_tag}__adapter.pt"
        path = Path(cfg.cache_dir).parent / "models" / fname
        if not path.exists():
            path = _PROJECT_ROOT / "models" / fname
        if path.exists():
            try:
                from src.model import load_adapter
                adapter, meta = load_adapter(str(path))
                # Use the feature mode the adapter was TRAINED with, not cfg's —
                # a concatenate-trained head fed a difference (D-dim) vector, or
                # vice-versa, would otherwise mismatch the head's input dim.
                self._adapter_feature_mode = meta.get("feature_mode", cfg.feature_mode)
                print(f"Loaded PEFT adapter: {path} (feature_mode={self._adapter_feature_mode})")
                return adapter
            except Exception as exc:
                print(f"Adapter load failed ({exc}); PEFT disabled.")
        self._adapter_feature_mode = cfg.feature_mode
        return None

    # -- patch (localised) scoring --------------------------------------
    def _patch_scores(self, text: str) -> np.ndarray:
        """Localised patch-level Δ-similarity (REPORT Appendix B.10, best DEN
        config). Encodes per-patch embeddings for the loaded corpus once, lazily,
        then caches them on the engine for subsequent queries."""
        from src.benchmark import encode_query
        from src.retrieval import top_patch_change_scores
        if getattr(self, "_patch_t1", None) is None:
            p1, p2 = [], []
            for pk in self.store.pairs:
                im1, im2 = self.dataset.load_pair_images(pk)
                a = self.encoder.encode_image_patches(im1)
                b = self.encoder.encode_image_patches(im2)
                if a is None or b is None:
                    raise RuntimeError(
                        f"{self.encoder.name} exposes no patch tokens; "
                        "pick zero_shot/naive/peft.")
                p1.append(a)
                p2.append(b)
            self._patch_t1 = np.stack(p1)
            self._patch_t2 = np.stack(p2)
        t = encode_query(self.encoder, text)
        return top_patch_change_scores(self._patch_t1, self._patch_t2, t)

    # -- query ----------------------------------------------------------
    def query(
        self,
        text: str,
        approach: str,
        top_k: int,
        *,
        geo_region: str = "All",
        rerank_strategy: Optional[str] = None,
    ) -> List[ChangeEvent]:
        """Retrieve top-K change events matching *text*.

        Parameters
        ----------
        text: Natural-language change query.
        approach: ``"naive"``, ``"zero_shot"``, or ``"peft"``.
        top_k: Number of results to return.
        geo_region: If not ``"All"``, restrict pairs to that region
            (requires ``aoi_metadata.json`` to have been loaded).
        rerank_strategy: ``"diversity"``, ``"coherence"``, or ``None``
            (disabled).  Requires ``aoi_metadata.json``.
        """
        if approach == "patch":
            scores = self._patch_scores(text)
        else:
            if approach == "peft" and self.retriever.adapter is None:
                raise RuntimeError(
                    "PEFT selected but no adapter found. Train one with "
                    "`python -m src.train` or pick zero_shot.")
            scores = self.retriever.score_all(text, approach=approach)

        # Geographic filter — mask excluded pairs with -inf
        if geo_region not in ("All", "", None) and self._geo_filter is not None:
            allowed = {
                p.location_id
                for p in self._geo_filter.filter_by_region(self.store.pairs, geo_region)
            }
            mask = np.array([p.location_id in allowed for p in self.store.pairs])
            scores = np.where(mask, scores, -np.inf)

        # Confidence bounds over finite scores only
        finite = scores[np.isfinite(scores)]
        lo = float(finite.min()) if len(finite) else 0.0
        hi = float(finite.max()) if len(finite) else 1.0

        # Re-ranking or default (stable) argsort
        if rerank_strategy is not None and self._reranker is not None:
            order = self._reranker.rerank(
                scores, self.store.pairs, top_k, strategy=rerank_strategy
            )
        else:
            order = np.argsort(-scores, kind="stable")[:top_k]

        events: List[ChangeEvent] = []
        for rank, i in enumerate(order):
            pair = self.store.pairs[i]
            t1 = t2 = hm = None
            try:
                t1, t2 = self.dataset.load_pair_images(pair)
            except Exception as exc:
                print(f"image load failed for {pair}: {exc}")
            if t1 is not None and t2 is not None:
                try:
                    # Real query-conditioned CHANGE heatmap (per-patch Δ T1->T2);
                    # fall back to query-vs-After match if no patch tokens.
                    _, hm = generate_change_heatmap(np.array(t1), np.array(t2),
                                                    text, self.encoder, alpha=0.5)
                    if hm is None:
                        _, hm = generate_heatmap(np.array(t1), np.array(t2),
                                                 text, self.encoder, alpha=0.5)
                except Exception as exc:
                    print(f"heatmap failed: {exc}")
            conf = (float(scores[i]) - lo) / (hi - lo) if hi > lo else 1.0
            caption, note = self._describe(pair)
            events.append(ChangeEvent(
                rank=rank + 1,
                pair_id=f"{pair.location_id}_{pair.t1_key}_{pair.t2_key}",
                location=pair.location_id, t1_key=pair.t1_key,
                t2_key=pair.t2_key, score=float(scores[i]),
                confidence=round(conf, 3), caption=caption,
                seasonal_note=note, t1_img=t1, t2_img=t2, heatmap=hm,
            ))
        return events

    def _describe(self, pair) -> tuple[str, str]:
        try:
            label = self.dataset.get_pair_label(pair)
        except Exception:
            label = None
        cap_fn = getattr(self.dataset, "text_caption_for_pair", None)
        caption = cap_fn(pair) if cap_fn else (
            label.change_type if label else "unknown change")
        if label is None:
            return caption, "no label (unlabeled dataset)"
        if _SNOW in (label.dominant_t1_class, label.dominant_t2_class):
            return caption, "involves snow/ice - likely SEASONAL, not permanent"
        if label.stable:
            return caption, "labelled stable - weak/no change"
        return caption, "permanent land-cover change"

    # -- gradio ---------------------------------------------------------
    def stats_markdown(self) -> str:
        """One-line corpus stats card (rendered as Gradio markdown)."""
        n_pairs = len(self.store)
        n_locs = len(self.dataset.list_locations())
        per_loc = (n_pairs / n_locs) if n_locs else 0
        adapter = "loaded" if self.retriever.adapter is not None else "none"
        geo_status = "ready" if self._geo_filter is not None else "N/A (no metadata)"
        rerank_status = "ready" if self._reranker is not None else "N/A (no metadata)"
        return (
            f"<div class='stats-card'>"
            f"<b>Dataset:</b> <code>{self.cfg.dataset}</code> &nbsp;|&nbsp; "
            f"<b>Split:</b> <code>{self.cfg.split or 'all'}</code> &nbsp;|&nbsp; "
            f"<b>Pairing:</b> <code>{self.cfg.pairing}</code> &nbsp;|&nbsp; "
            f"<b>Locations:</b> <code>{n_locs}</code> &nbsp;|&nbsp; "
            f"<b>Image pairs:</b> <code>{n_pairs}</code> "
            f"({per_loc:.0f}/location) &nbsp;|&nbsp; "
            f"<b>Encoder:</b> <code>{self.encoder.name}</code> "
            f"({self.encoder.embed_dim}-d) &nbsp;|&nbsp; "
            f"<b>PEFT adapter:</b> <code>{adapter}</code> &nbsp;|&nbsp; "
            f"<b>Geo filter:</b> <code>{geo_status}</code> &nbsp;|&nbsp; "
            f"<b>Re-ranking:</b> <code>{rerank_status}</code>"
            f"</div>"
        )

    def reload(self, dataset: str, encoder: str, approach: str,
               color_mode: str = "rgb", use_lora: bool = False):
        try:
            self.cfg = RunConfig(
                dataset=dataset, encoder=encoder, approach=approach,
                root=self.cfg.root, pairing=self.cfg.pairing,
                split=self.cfg.split, cache_dir=self.cfg.cache_dir,
                color_mode=color_mode, use_lora=use_lora,
                geo_filter=self.cfg.geo_filter, rerank=self.cfg.rerank,
                rerank_strategy=self.cfg.rerank_strategy,
            )
            self._build(self.cfg)
            lora_note = " + LoRA" if use_lora else ""
            status = (f"Loaded {dataset} + {encoder}{lora_note} | "
                      f"color={color_mode} | approach={approach} | "
                      f"{len(self.store)} pairs")
            return status, self.stats_markdown()
        except Exception as exc:
            traceback.print_exc()
            return f"Error: {exc}", "<div class='stats-card stats-err'>Error</div>"

    def build_interface(self):
        import gradio as gr
        engine = self
        from src.encoders import list_encoders

        APPROACH_HELP = (
            "Naive = cos(text, After) — image retrieval baseline.  "
            "Zero-shot = cos(text, After) − cos(text, Before) — no training needed.  "
            "PEFT adapter = cos(text, adapter(Δf)) — trained projection head, best quality."
        )
        ENCODER_HELP = (
            "clip_vitl14 (general purpose, 768-d)  ·  georsclip (RS-pretrained ViT-B/32, 512-d)  "
            "·  remoteclip (RS-pretrained ViT-L/14, 768-d). Weights download on first use."
        )
        DATASET_HELP = (
            "Pair corpus to search over. Switching rebuilds (or loads cached) embeddings — "
            "press Apply Settings afterwards."
        )
        TOPK_HELP = "How many ranked change events to return."
        QUERY_HELP = ("Describe the land-cover change you are looking for. "
                      "Specific transitions (e.g. 'forest cleared to bare soil') work best.")
        COLOR_MODE_HELP = (
            "rgb = standard optical (R-G-B)  ·  nrg = NIR-Red-Green false colour "
            "(best zero-shot with GeoRSCLIP)  ·  ndvi = vegetation index (single channel × 3). "
            "Changing requires Apply Settings."
        )
        LORA_HELP = (
            "Load LoRA-adapted embeddings pre-cached by run_pipeline --lora. "
            "Cache must exist for the selected encoder + color mode. "
            "Changing requires Apply Settings."
        )

        with gr.Blocks(title="Semantic Change Search Engine",
                       analytics_enabled=False) as demo:
            gr.Markdown(
                "# Semantic Change Search Engine\n"
                "Describe a land-cover change in plain language — "
                "the engine finds the satellite image pairs where it happened."
            )

            # ---- Query first — most important ----
            with gr.Row():
                q = gr.Textbox(
                    label="Change query", scale=5,
                    value="new buildings on former farmland",
                    placeholder="e.g. agricultural land converted to wetland",
                    info=QUERY_HELP,
                )
                a_dd = gr.Dropdown(
                    choices=[
                        ("Naive (baseline)", "naive"),
                        ("Zero-shot (no training)", "zero_shot"),
                        ("Patch / localised (best on DEN)", "patch"),
                        ("PEFT adapter (trained)", "peft"),
                    ],
                    value=engine.cfg.approach,
                    label="Approach",
                    info=APPROACH_HELP,
                    scale=2,
                )
                k = gr.Slider(1, 10, value=engine.cfg.top_k, step=1,
                              label="Top-K", info=TOPK_HELP, scale=1)
                go = gr.Button("Search", variant="primary", size="lg",
                               elem_id="search-btn")

            # NOTE: gr.Examples (Dataset component) triggers a frontend render loop in
            # gradio 6.14 when combined with the full component tree — page sticks on
            # "Loading..." and Firefox flags "slowing down". Plain fill-buttons give the
            # same click-to-fill UX without the Dataset component.
            gr.Markdown("Example queries — click to fill")
            with gr.Row():
                for _ex in [
                    "new buildings on former farmland",
                    "forest cleared to bare soil",
                    "agricultural land converted to wetland",
                    "seasonal snow melting away",
                ]:
                    gr.Button(_ex, size="sm").click(lambda v=_ex: v, None, q)

            # ---- Settings (power-user, collapsed) ----
            with gr.Accordion("Settings", open=False):
                stats_md = gr.Markdown(engine.stats_markdown())
                with gr.Row():
                    d_dd = gr.Dropdown(_labeled(_app_datasets(), DATASET_LABELS),
                                       value=engine.cfg.dataset,
                                       label="Dataset", info=DATASET_HELP)
                    e_dd = gr.Dropdown(_labeled(list_encoders(), ENCODER_LABELS),
                                       value=engine.cfg.encoder,
                                       label="Encoder", info=ENCODER_HELP)
                with gr.Row():
                    color_dd = gr.Dropdown(
                        ["rgb", "nrg", "ndvi"], value=engine.cfg.color_mode,
                        label="Color mode", info=COLOR_MODE_HELP,
                    )
                    lora_chk = gr.Checkbox(
                        label="Use LoRA embeddings",
                        value=engine.cfg.use_lora,
                        info=LORA_HELP,
                    )
                with gr.Row():
                    apply = gr.Button("Apply Settings", variant="secondary")
                    status = gr.Textbox(
                        label="Engine status", interactive=False, scale=4,
                        value=f"{engine.cfg.dataset} + {engine.cfg.encoder} | "
                              f"color={engine.cfg.color_mode} | "
                              f"{len(engine.store)} pairs | "
                              f"approach={engine.cfg.approach}")
                apply.click(engine.reload,
                            [d_dd, e_dd, a_dd, color_dd, lora_chk],
                            [status, stats_md])

            # ---- Filters & Re-ranking (per-query, no Apply Settings needed) ----
            _geo_available = engine._geo_filter is not None
            _regions = engine._geo_filter.regions if _geo_available else ["All"]
            _rerank_available = engine._reranker is not None

            with gr.Accordion("Filters & Re-ranking", open=False):
                gr.Markdown(
                    "These options take effect on the **next Search** — "
                    "no Apply Settings needed."
                )
                with gr.Row():
                    geo_chk = gr.Checkbox(
                        label="Geographic filter",
                        value=engine.cfg.geo_filter and _geo_available,
                        interactive=_geo_available,
                        info="Restrict results to one continental region."
                        if _geo_available
                        else "Requires aoi_metadata.json in --root.",
                    )
                    geo_dd = gr.Dropdown(
                        _regions,
                        value="All",
                        label="Region",
                        interactive=_geo_available,
                    )
                with gr.Row():
                    rerank_chk = gr.Checkbox(
                        label="Re-rank results",
                        value=engine.cfg.rerank and _rerank_available,
                        interactive=_rerank_available,
                        info="Post-process ranking for spatial quality."
                        if _rerank_available
                        else "Requires aoi_metadata.json in --root.",
                    )
                    rerank_dd = gr.Dropdown(
                        list(RERANK_STRATEGIES),
                        value=engine.cfg.rerank_strategy,
                        label="Strategy",
                        interactive=_rerank_available,
                        info=(
                            "diversity = prefer unique AOIs per result  ·  "
                            "coherence = cluster near top-1 location"
                        ),
                    )

            # ---- Results ----
            gr.Markdown("## Top match")
            with gr.Group(elem_classes="top-card"):
                with gr.Row():
                    t1i = gr.Image(label="Before", height=300, interactive=False)
                    t2i = gr.Image(label="After", height=300, interactive=False)
                    hmi = gr.Image(label="Change heatmap (Δ T1→T2)", height=300,
                                   interactive=False)
                gr.Markdown(
                    "_Heatmap (jet colormap) overlays the **After** image and shows the "
                    "per-patch **change** in similarity to your query from Before→After — "
                    "**warm red/yellow = query presence grew most**, **cool blue = little/no "
                    "change**. (Falls back to a query-vs-After match map if the encoder "
                    "exposes no patch tokens.)_"
                )
                summary = gr.Markdown("*Press Search to retrieve.*")

            # ---- All matches at a glance (top-K) ----
            gr.Markdown("## All matches")
            gallery = gr.Gallery(
                label="Top-K change heatmaps — click any tile to enlarge",
                columns=5, height=260, object_fit="contain",
                show_label=True, interactive=False,
            )

            with gr.Accordion("Results table", open=True):
                table = gr.Dataframe(
                    headers=["rank", "location", "Before", "After", "score",
                             "match (0–1)", "caption", "land cover change"],
                    interactive=False, wrap=True,
                )

            def _pill(c: float) -> str:
                cls = "low" if c < 0.4 else ("mid" if c < 0.7 else "")
                return (f"<span class='conf-pill {cls}' title='relative match score "
                        f"(min–max normalized over the candidate set), not a calibrated "
                        f"probability'>match {c:.2f}</span>")

            def handle(text, approach, top_k,
                       geo_enabled, geo_region, rerank_enabled, rerank_strategy,
                       progress=gr.Progress()):
                try:
                    progress(0.05, desc="Scoring corpus against your query… "
                             "(first query on a dataset/approach encodes it — a few seconds)")
                    active_geo = geo_region if geo_enabled else "All"
                    active_rerank = rerank_strategy if rerank_enabled else None
                    evs = engine.query(
                        text, approach, int(top_k),
                        geo_region=active_geo,
                        rerank_strategy=active_rerank,
                    )
                except Exception as exc:
                    return None, None, None, f"**Error:** {exc}", [], []
                if not evs:
                    return None, None, None, "*No results.*", [], []
                progress(0.9, desc="Rendering results…")
                top = evs[0]
                rows = [[e.rank, e.location, e.t1_key, e.t2_key,
                         round(e.score, 4), e.confidence, e.caption,
                         e.seasonal_note] for e in evs]
                # Gallery: prefer the localization heatmap, fall back to the After
                # tile when a heatmap could not be generated for that pair.
                gallery_items = [
                    (img, f"#{e.rank} · {e.location} · "
                          f"{e.t1_key}→{e.t2_key} · match {e.confidence:.2f}")
                    for e in evs
                    for img in (e.heatmap or e.t2_img,)
                    if img is not None
                ]
                md = (
                    f"### {top.caption}\n"
                    f"{_pill(top.confidence)} &nbsp; "
                    f"**Location** `{top.location}` &nbsp; "
                    f"**Before → After** `{top.t1_key}` → `{top.t2_key}` &nbsp; "
                    f"**Score** `{top.score:.4f}`\n\n"
                    f"_Reasoning:_ {top.seasonal_note}."
                )
                return top.t1_img, top.t2_img, top.heatmap, md, rows, gallery_items

            go.click(
                handle,
                [q, a_dd, k, geo_chk, geo_dd, rerank_chk, rerank_dd],
                [t1i, t2i, hmi, summary, table, gallery],
            )
            q.submit(
                handle,
                [q, a_dd, k, geo_chk, geo_dd, rerank_chk, rerank_dd],
                [t1i, t2i, hmi, summary, table, gallery],
            )
        return demo


def parse_args():
    p = argparse.ArgumentParser(description="Semantic Change Search Engine")
    from src.encoders import list_encoders
    p.add_argument("--dataset", default="dynamic_earthnet",
                   choices=_app_datasets())
    p.add_argument("--encoder", default="georsclip",
                   choices=list_encoders())
    p.add_argument("--approach", default="patch",
                   choices=list(APPROACHES) + ["patch"])
    p.add_argument("--root", default=str(_PROJECT_ROOT / "data" / "DynamicEarthNet"))
    p.add_argument("--pairing", default="bimonthly")
    p.add_argument("--split", default="test",
                   help="DEN preprocessed split: train|val|test|all. Default test "
                        "(110 held-out pairs); train (605) is the corpus the PEFT "
                        "adapter was fit on, so avoid it for an honest PEFT demo.")
    p.add_argument("--cache-dir", default=str(_PROJECT_ROOT / "data" / "cache"))
    p.add_argument("--port", type=int, default=7860)
    # Spectral / encoder options. Default nrg + georsclip = the REPORT headline config.
    p.add_argument("--color-mode", default="nrg", choices=["rgb", "nrg", "ndvi"],
                   help="Image colour mode passed to dataset loader. "
                        "nrg = NIR-Red-Green (best zero-shot with GeoRSCLIP); "
                        "ndvi = vegetation index. Change in-app via Settings.")
    p.add_argument("--lora", action="store_true", default=False,
                   help="Load LoRA-adapted embeddings (must be pre-cached by run_pipeline --lora). "
                        "Toggle in-app via Settings.")
    p.add_argument("--no-lora", dest="lora", action="store_false")
    # Extension toggles
    p.add_argument("--geo-filter", action="store_true", default=False,
                   help="Enable geographic region filter at startup (toggle in-app anytime).")
    p.add_argument("--no-geo-filter", dest="geo_filter", action="store_false")
    p.add_argument("--rerank", action="store_true", default=False,
                   help="Enable post-retrieval re-ranking at startup (toggle in-app anytime).")
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.add_argument("--rerank-strategy", default="diversity",
                   choices=list(RERANK_STRATEGIES),
                   help="Re-ranking strategy: diversity (default) or coherence.")
    a = p.parse_args()
    return RunConfig(
        dataset=a.dataset, encoder=a.encoder, approach=a.approach,
        root=a.root, pairing=a.pairing,
        split=None if a.split == "all" else a.split,
        cache_dir=a.cache_dir,
        color_mode=a.color_mode,
        use_lora=a.lora,
        geo_filter=a.geo_filter,
        rerank=a.rerank,
        rerank_strategy=a.rerank_strategy,
    ), a.port


def main():
    import gradio as gr
    cfg, port = parse_args()
    print(f"Starting engine: dataset={cfg.dataset} encoder={cfg.encoder} "
          f"approach={cfg.approach}")
    engine = SemanticChangeSearch(cfg)
    demo = engine.build_interface()
    # Local system fonts only — no remote Google-font fetch (blocks render on slow net).
    theme = gr.themes.Ocean(
        font=["system-ui", "-apple-system", "Segoe UI", "Arial", "sans-serif"],
        font_mono=["Consolas", "Monaco", "monospace"],
    )
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True,
                theme=theme, css=SemanticChangeSearch._CSS)


if __name__ == "__main__":
    main()
