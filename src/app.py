"""
Week 7: Gradio Frontend - Semantic Change Search Interface.
"""
import os
import glob
import traceback

import torch
import numpy as np
import pandas as pd
from PIL import Image
import faiss
import gradio as gr
from transformers import AutoTokenizer, AutoModel, AutoProcessor

os.environ["TRANSFORMERS_CACHE"] = "/tmp/clip-cache"

try:
    from src.heatmap import apply_heatmap_only, resize_heatmap
    from src.error_analysis import generate_false_positive_report
    from src.data_loader import load_parquet_as_embeddings
except ImportError as e:
    raise RuntimeError(f"Missing local modules: {e}")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_CACHE_PATH = os.path.join(_DATA_DIR, "clip_embeddings.npz")
_MODEL_NAME = "openai/clip-vit-large-patch14"


class SemanticChangeSearch:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load CLIP
        self.clip_model = AutoModel.from_pretrained(_MODEL_NAME, cache_dir="/tmp/clip-cache").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, cache_dir="/tmp/clip-cache")
        self.processor = AutoProcessor.from_pretrained(_MODEL_NAME, cache_dir="/tmp/clip-cache")

        # Find parquet shards
        parquet_files = sorted(glob.glob(os.path.join(_DATA_DIR, "*.parquet")))
        if not parquet_files:
            raise RuntimeError(
                f"No .parquet files found in '{_DATA_DIR}/'. "
                "Download at least one shard from the QFabric dataset and place it there."
            )

        print(f"Found {len(parquet_files)} shard(s): {[os.path.basename(p) for p in parquet_files]}")
        self.embedding_lookup, self.metadata_df, self.image_lookup = load_parquet_as_embeddings(
            parquet_paths=parquet_files,
            clip_model=self.clip_model,
            processor=self.processor,
            device=self.device,
            cache_path=_CACHE_PATH,
        )

        # Build Faiss index (L2 over L2-normalised embeddings = cosine search)
        dim = list(self.embedding_lookup.values())[0].shape[1]
        self.index = faiss.IndexFlatL2(dim)
        for emb_array in self.embedding_lookup.values():
            norm = np.linalg.norm(emb_array, axis=1, keepdims=True)
            norm[norm == 0] = 1
            self.index.add((emb_array / norm).astype(np.float32))
        print(f"Faiss index built with {self.index.ntotal} vectors")

    def _encode_text(self, text_query: str) -> np.ndarray:
        """Encode a text query to a 768-dim numpy vector."""
        input_ids = self.tokenizer(
            text_query, return_tensors="pt", max_length=77,
            padding=True, truncation=True
        ).input_ids.to(self.device)
        with torch.no_grad():
            hidden = self.clip_model.text_model(input_ids).last_hidden_state  # [1, 77, 768]
        return hidden.mean(dim=1).squeeze(0).cpu().numpy()  # [768]

    def _make_heatmap(self, img: Image.Image, text_query: str) -> Image.Image:
        """Generate a 224×224 jet-coloured attention heatmap for one image."""
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
        input_ids = self.tokenizer(
            text_query, return_tensors="pt", max_length=77, truncation=True
        ).input_ids.to(self.device)
        with torch.no_grad():
            patches = self.clip_model.vision_model(pixel_values).last_hidden_state  # [1, 577, 1024]
            text_vec = self.clip_model.text_model(input_ids).last_hidden_state.mean(dim=1)  # [1, 768]
            patch_tokens = patches[0, 1:]  # [576, 1024]
            patch_proj = self.clip_model.visual_projection(patch_tokens)  # [576, 768]
            text_norm = torch.nn.functional.normalize(text_vec, dim=-1)  # [1, 768]
            patch_norm = torch.nn.functional.normalize(patch_proj, dim=-1)  # [576, 768]
            scores = (patch_norm @ text_norm.T).squeeze(-1).detach().cpu().numpy()  # [576]

        grid_size = int(len(scores) ** 0.5)
        grid = scores.reshape(grid_size, grid_size)
        grid_norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        return apply_heatmap_only(resize_heatmap(grid_norm, 224, 224))

    def query_changes(self, text_query: str):
        text_vec = self._encode_text(text_query)
        print(f"Querying for '{text_query}'...")

        scores, indices = self.index.search(text_vec.reshape(1, -1).astype(np.float32), 3)
        valid_indices = np.clip(indices[0], 0, len(self.metadata_df) - 1)

        # Best match → display images
        best_row = self.metadata_df.iloc[valid_indices[0]]
        best_loc = best_row['location']
        imgs = self.image_lookup.get(best_loc, [])
        display_t1 = imgs[0] if len(imgs) > 0 else None
        display_t2 = imgs[1] if len(imgs) > 1 else imgs[0] if imgs else None

        # Heatmap from T1 of best match
        heatmap = None
        if display_t1 is not None:
            try:
                heatmap = self._make_heatmap(display_t1, text_query)
            except Exception as e:
                print(f"Heatmap error: {e}")
                traceback.print_exc()

        # Error analysis for each match
        confidences, reasons = [], []
        for i, idx in enumerate(valid_indices):
            loc_id = self.metadata_df.iloc[idx]['location']
            emb_array = self.embedding_lookup[loc_id]
            try:
                report = generate_false_positive_report(
                    location_id=f"{loc_id} (score={float(scores[0][i]):.3f})",
                    embeddings=emb_array,
                    timestamps=self.metadata_df[self.metadata_df['location'] == loc_id]['timestamp'],
                )
                confidences.append(report.confidence_score)
                reasons.append(f"{loc_id}: " + "; ".join(report.misclassification_reasons[:3]))
            except Exception as e:
                confidences.append(0.5)
                reasons.append(f"{loc_id}: Error ({str(e)[:80]})")

        confidence = float(np.mean(confidences)) if confidences else 0.5
        return display_t1, display_t2, heatmap, confidence, reasons

    def create_gradio_interface(self):
        engine = self  # reuse the already-loaded instance
        with gr.Blocks(title="Semantic Change Search Engine") as demo:
            gr.Markdown("# Semantic Change Search Engine")
            with gr.Row():
                query_input = gr.Textbox(
                    label="Enter a text query",
                    placeholder='e.g., "snow melting"',
                    value="snow melting"
                )
                submit_btn = gr.Button("Search Changes", variant="primary")
            with gr.Row():
                t1_image = gr.Image(label="T1 Image (earlier timepoint)", height=384, interactive=False)
                t2_image = gr.Image(label="T2 Image (later timepoint)", height=384, interactive=False)
            with gr.Row():
                heatmap_display = gr.Image(label="CLIP Attention Heatmap", height=224, interactive=False)
            with gr.Row():
                confidence_num = gr.Number(label="Confidence Score (0–1)", precision=3)
                reasons_text = gr.Textbox(
                    label="Change Classification Reasoning",
                    interactive=False, lines=4
                )

            def handle_query(query):
                if not query or not query.strip():
                    return None, None, None, None, "Please enter a query."
                try:
                    t1, t2, heatmap, confidence, reasons = engine.query_changes(query)
                    return t1, t2, heatmap, confidence, "\n".join(reasons)
                except Exception as e:
                    traceback.print_exc()
                    return None, None, None, None, f"Error: {str(e)[:300]}"

            submit_btn.click(
                fn=handle_query,
                inputs=[query_input],
                outputs=[t1_image, t2_image, heatmap_display, confidence_num, reasons_text],
            )
        return demo


def main():
    print("Starting Semantic Change Search Engine...")
    engine = SemanticChangeSearch()
    demo = engine.create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0", server_port=7860,
        share=False, debug=True, show_error=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
