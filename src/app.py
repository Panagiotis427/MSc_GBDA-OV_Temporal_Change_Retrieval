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
        return self

    def _build(self, cfg: RunConfig) -> None:
        from src.datasets.registry import build_dataset
        self.dataset = build_dataset(cfg.dataset, root=cfg.root,
                                     pairing=cfg.pairing, split=cfg.split)
        self.encoder = get_encoder(cfg.encoder)
        self.store = load_or_compute(self.dataset, self.encoder,
                                     cache_dir=cfg.cache_dir)
        self.retriever = ChangeRetriever(self.store, self.encoder,
                                         feature_mode=cfg.feature_mode)
        self._adapter = self._maybe_load_adapter(cfg)
        if self._adapter is not None:
            self.retriever.set_adapter(self._adapter, cfg.feature_mode)

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
    def query(self, text: str, approach: str, top_k: int) -> List[ChangeEvent]:
        if approach == "peft" and self.retriever.adapter is None:
            raise RuntimeError(
                "PEFT selected but no adapter found. Train one with "
                "`python -m src.train` or pick zero_shot.")
        scores = self.retriever.score_all(text, approach=approach)
        lo, hi = float(scores.min()), float(scores.max())
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
            f"<b>PEFT adapter:</b> <code>{adapter}</code>"
            f"</div>"
        )

    def reload(self, dataset: str, encoder: str, approach: str):
        try:
            self.cfg = RunConfig(dataset=dataset, encoder=encoder,
                                 approach=approach, root=self.cfg.root,
                                 pairing=self.cfg.pairing,
                                 split=self.cfg.split,
                                 cache_dir=self.cfg.cache_dir)
            self._build(self.cfg)
            status = (f"Loaded {dataset} + {encoder} | approach={approach} | "
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

        INTRO = (
            "# Semantic Change Search Engine\n"
            "Type a natural-language description of a land-cover change "
            "(e.g. *'new buildings on former farmland'*). The system ranks "
            "every bi-temporal `(T1, T2)` image pair in the dataset by how "
            "well the change between the two timesteps matches the query, "
            "and returns the matched tiles plus a heatmap on T2."
        )

        APPROACH_HELP = (
            "naive = cos(text, T2)  ·  zero_shot = cos(text, T2) − cos(text, T1) "
            "(Δ-similarity, no training)  ·  peft = cos(text, adapter(Δf)) "
            "(trained ProjectionHead, best quality)"
        )
        ENCODER_HELP = (
            "clip_vitl14 (general, 768-d)  ·  georsclip (RS-pretrained ViT-B/32, 512-d)  "
            "·  remoteclip (RS-pretrained ViT-L/14, 768-d). Weights download on first use."
        )
        DATASET_HELP = (
            "Pair corpus to search over. Pulled live from the dataset registry; "
            "switching rebuilds (or loads cached) embeddings."
        )
        TOPK_HELP = "How many ranked change events to return."
        QUERY_HELP = ("Natural language. Use specific land-cover transitions "
                      "(verbs and class names) for best results.")

        css = """
        .top-card {border: 1px solid var(--border-color-primary, #d0d0d0);
                   border-radius: 12px; padding: 14px 18px;
                   background: var(--background-fill-secondary, #fafafa);}
        .conf-pill {display:inline-block; padding:4px 12px; border-radius:999px;
                    background:#2e7d32; color:white; font-weight:600;}
        .conf-pill.mid {background:#f9a825;}
        .conf-pill.low {background:#c62828;}
        .stats-card {border:1px solid var(--border-color-primary,#d0d0d0);
                     border-left:4px solid #1565c0;
                     border-radius:10px; padding:10px 14px;
                     background:var(--background-fill-secondary,#f5f9ff);
                     font-size:0.95em; line-height:1.7;}
        .stats-card code {background:rgba(21,101,192,0.08); padding:1px 6px;
                          border-radius:4px; font-weight:600;}
        .stats-err {border-left-color:#c62828; background:#fff4f4;}
        """

        with gr.Blocks(title="Semantic Change Search Engine", css=css,
                       theme=gr.themes.Soft()) as demo:
            gr.Markdown(INTRO)

            stats_md = gr.Markdown(engine.stats_markdown())

            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    d_dd = gr.Dropdown(list_datasets(), value=engine.cfg.dataset,
                                       label="Dataset", info=DATASET_HELP)
                    e_dd = gr.Dropdown(list_encoders(), value=engine.cfg.encoder,
                                       label="Encoder", info=ENCODER_HELP)
                    a_dd = gr.Dropdown(list(APPROACHES), value=engine.cfg.approach,
                                       label="Approach", info=APPROACH_HELP)
                with gr.Row():
                    apply = gr.Button("Apply (rebuild)", variant="secondary")
                    status = gr.Textbox(
                        label="Engine status", interactive=False, scale=4,
                        value=f"{engine.cfg.dataset} + {engine.cfg.encoder} | "
                              f"{len(engine.store)} pairs | "
                              f"approach={engine.cfg.approach}")
                apply.click(engine.reload, [d_dd, e_dd, a_dd], [status, stats_md])

            with gr.Row():
                q = gr.Textbox(
                    label="Change query", scale=5,
                    value="new buildings on former farmland",
                    placeholder="e.g. agricultural land converted to wetland",
                    info=QUERY_HELP,
                )
                k = gr.Slider(1, 10, value=engine.cfg.top_k, step=1,
                              label="Top-K", info=TOPK_HELP)
                go = gr.Button("Search", variant="primary", size="lg")

            gr.Markdown("## Top match")
            with gr.Group(elem_classes="top-card"):
                with gr.Row():
                    t1i = gr.Image(label="T1 (earlier)", height=300,
                                   interactive=False)
                    t2i = gr.Image(label="T2 (later)", height=300,
                                   interactive=False)
                    hmi = gr.Image(label="Change heatmap on T2", height=300,
                                   interactive=False)
                summary = gr.Markdown("*Press Search to retrieve.*")

            with gr.Accordion("All ranked change events", open=True):
                table = gr.Dataframe(
                    headers=["rank", "location", "T1", "T2", "score",
                             "confidence", "caption", "note"],
                    interactive=False, wrap=True,
                )

            def _pill(c: float) -> str:
                cls = "low" if c < 0.4 else ("mid" if c < 0.7 else "")
                return f"<span class='conf-pill {cls}'>confidence {c:.2f}</span>"

            def handle(text, approach, top_k):
                try:
                    evs = engine.query(text, approach, int(top_k))
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
                    f"**T1 -> T2** `{top.t1_key}` -> `{top.t2_key}` &nbsp; "
                    f"**Score** `{top.score:.4f}`\n\n"
                    f"_Reasoning:_ {top.seasonal_note}."
                )
                return top.t1_img, top.t2_img, top.heatmap, md, rows

            go.click(handle, [q, a_dd, k], [t1i, t2i, hmi, summary, table])
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
    a = p.parse_args()
    return RunConfig(dataset=a.dataset, encoder=a.encoder, approach=a.approach,
                     root=a.root, pairing=a.pairing,
                     split=None if a.split == "all" else a.split,
                     cache_dir=a.cache_dir), a.port


def main():
    cfg, port = parse_args()
    print(f"Starting engine: dataset={cfg.dataset} encoder={cfg.encoder} "
          f"approach={cfg.approach}")
    engine = SemanticChangeSearch(cfg)
    demo = engine.build_interface()
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)


if __name__ == "__main__":
    main()
