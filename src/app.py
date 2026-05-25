"""
Semantic Change Search Engine — Gradio frontend (rewired for real change
retrieval).

A natural-language query is scored against every bi-temporal pair's *change*
representation (not single images), returning a ranked list of change events.
Each result shows the actual T1/T2 tiles of the matched pair, a
query-conditioned heatmap on T2, a confidence, and a seasonal-vs-permanent
note grounded in the dataset's labels.

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
from src.heatmap import generate_heatmap
from src.rerank import RERANK_STRATEGIES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SNOW = "snow_and_ice"


@dataclass
class RunConfig:
    dataset: str = "dynamic_earthnet"
    encoder: str = "clip_vitl14"
    approach: str = "zero_shot"
    root: str = str(_PROJECT_ROOT / "data" / "DynamicEarthNet")
    pairing: str = "bimonthly"
    split: Optional[str] = "train"   # 605 DEN pairs; adapter was fit here
    cache_dir: str = str(_PROJECT_ROOT / "data" / "cache")
    feature_mode: str = "difference"
    top_k: int = 5
    # Spectral / encoder options (require Apply to reload embeddings)
    color_mode: str = "rgb"         # rgb | nrg | ndvi
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
        if adapter is not None:
            self.retriever.set_adapter(adapter, cfg.feature_mode)
        self._adapter = adapter
        self._init_extensions(cfg)
        return self

    def _build(self, cfg: RunConfig) -> None:
        from src.datasets.registry import build_dataset
        from src.embeddings import cache_path as _cache_path
        self.dataset = build_dataset(cfg.dataset, root=cfg.root,
                                     pairing=cfg.pairing, split=cfg.split,
                                     color_mode=cfg.color_mode)
        self.encoder = get_encoder(cfg.encoder)

        # Cache tag: match run_pipeline.py convention so pre-computed caches are reused.
        split_str = cfg.split or "all"
        color_tag = f"_{cfg.color_mode}" if cfg.color_mode != "rgb" else ""
        lora_tag = "_lora" if cfg.use_lora else ""
        # Match embeddings CLI convention: test+rgb (no lora) has no tag suffix
        cache_tag = (
            f"{split_str}{color_tag}{lora_tag}"
            if split_str != "test" or color_tag or lora_tag
            else ""
        )

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
            self.retriever.set_adapter(self._adapter, cfg.feature_mode)
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
        path = Path(cfg.cache_dir).parent / "models" / \
            f"{cfg.dataset}__{cfg.encoder}__adapter.pt"
        if not path.exists():
            path = _PROJECT_ROOT / "models" / \
                f"{cfg.dataset}__{cfg.encoder}__adapter.pt"
        if path.exists():
            try:
                from src.model import load_adapter
                adapter, _ = load_adapter(str(path))
                print(f"Loaded PEFT adapter: {path}")
                return adapter
            except Exception as exc:
                print(f"Adapter load failed ({exc}); PEFT disabled.")
        return None

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

        # Re-ranking or default argsort
        if rerank_strategy is not None and self._reranker is not None:
            order = self._reranker.rerank(
                scores, self.store.pairs, top_k, strategy=rerank_strategy
            )
        else:
            order = np.argsort(-scores)[:top_k]

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
        from src.datasets.registry import list_datasets
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

        with gr.Blocks(title="Semantic Change Search Engine") as demo:
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

            gr.Examples(
                examples=[
                    ["new buildings on former farmland"],
                    ["forest cleared to bare soil"],
                    ["agricultural land converted to wetland"],
                    ["seasonal snow melting away"],
                ],
                inputs=[q],
                label="Example queries — click to fill",
            )

            # ---- Settings (power-user, collapsed) ----
            with gr.Accordion("Settings", open=False):
                stats_md = gr.Markdown(engine.stats_markdown())
                with gr.Row():
                    d_dd = gr.Dropdown(list_datasets(), value=engine.cfg.dataset,
                                       label="Dataset", info=DATASET_HELP)
                    e_dd = gr.Dropdown(list_encoders(), value=engine.cfg.encoder,
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
                    hmi = gr.Image(label="Change heatmap (After)", height=300,
                                   interactive=False)
                summary = gr.Markdown("*Press Search to retrieve.*")

            with gr.Accordion("Results table", open=True):
                table = gr.Dataframe(
                    headers=["rank", "location", "Before", "After", "score",
                             "confidence", "caption", "land cover change"],
                    interactive=False, wrap=True,
                )

            def _pill(c: float) -> str:
                cls = "low" if c < 0.4 else ("mid" if c < 0.7 else "")
                return f"<span class='conf-pill {cls}'>confidence {c:.2f}</span>"

            def handle(text, approach, top_k,
                       geo_enabled, geo_region, rerank_enabled, rerank_strategy):
                try:
                    active_geo = geo_region if geo_enabled else "All"
                    active_rerank = rerank_strategy if rerank_enabled else None
                    evs = engine.query(
                        text, approach, int(top_k),
                        geo_region=active_geo,
                        rerank_strategy=active_rerank,
                    )
                except Exception as exc:
                    return None, None, None, f"**Error:** {exc}", []
                if not evs:
                    return None, None, None, "*No results.*", []
                top = evs[0]
                rows = [[e.rank, e.location, e.t1_key, e.t2_key,
                         round(e.score, 4), e.confidence, e.caption,
                         e.seasonal_note] for e in evs]
                md = (
                    f"### {top.caption}\n"
                    f"{_pill(top.confidence)} &nbsp; "
                    f"**Location** `{top.location}` &nbsp; "
                    f"**Before → After** `{top.t1_key}` → `{top.t2_key}` &nbsp; "
                    f"**Score** `{top.score:.4f}`\n\n"
                    f"_Reasoning:_ {top.seasonal_note}."
                )
                return top.t1_img, top.t2_img, top.heatmap, md, rows

            go.click(
                handle,
                [q, a_dd, k, geo_chk, geo_dd, rerank_chk, rerank_dd],
                [t1i, t2i, hmi, summary, table],
            )
            q.submit(
                handle,
                [q, a_dd, k, geo_chk, geo_dd, rerank_chk, rerank_dd],
                [t1i, t2i, hmi, summary, table],
            )
        return demo


def parse_args():
    p = argparse.ArgumentParser(description="Semantic Change Search Engine")
    from src.datasets.registry import list_datasets
    from src.encoders import list_encoders
    p.add_argument("--dataset", default="dynamic_earthnet",
                   choices=list_datasets())
    p.add_argument("--encoder", default="clip_vitl14",
                   choices=list_encoders())
    p.add_argument("--approach", default="zero_shot",
                   choices=list(APPROACHES))
    p.add_argument("--root", default=str(_PROJECT_ROOT / "data" / "DynamicEarthNet"))
    p.add_argument("--pairing", default="bimonthly")
    p.add_argument("--split", default="train",
                   help="DEN preprocessed split: train|val|test|all "
                        "(train = 605 pairs, the corpus the PEFT adapter was fit on)")
    p.add_argument("--cache-dir", default=str(_PROJECT_ROOT / "data" / "cache"))
    p.add_argument("--port", type=int, default=7860)
    # Spectral / encoder options
    p.add_argument("--color-mode", default="rgb", choices=["rgb", "nrg", "ndvi"],
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
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True,
                theme=gr.themes.Ocean(), css=SemanticChangeSearch._CSS)


if __name__ == "__main__":
    main()
