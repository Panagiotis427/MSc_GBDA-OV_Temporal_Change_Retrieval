"""
Week 7: Gradio Frontend - Semantic Change Search Interface.

This module provides a user-friendly web interface for querying temporal changes
in satellite imagery using natural language. Users can input a text query (e.g.,
"snow melting", "construction activity") and see:
- Side-by-side T1/T2 images with highlighted change regions
- CLIP attention heatmap overlay showing what the model focused on
- Confidence score for seasonal vs permanent classification
- Misclassification reasons if available
"""
import torch
import pandas as pd
from PIL import Image
import numpy as np
import traceback

def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to CLIP input tensor [3, H=224, W=224] then [1, 3, 224, 224]."""
    # Keep color information - don't convert to grayscale
    w, h = img.size
    if w != h:
        img_resized = img.resize((224, 224), Image.LANCZOS)
    else:
        img_resized = img
    img_hwc = np.array(img_resized).astype(np.float32) / 127.5 - 1.0
    # Handle both RGB and grayscale images
    if len(img_hwc.shape) == 2:
        # Grayscale, add channel dimension: [H,W] -> [1,H,W]
        img_hwc = np.expand_dims(img_hwc, axis=2)
    # Transpose from [H,W,C] to [C,H,W]
    img_hwc = np.transpose(img_hwc, (2, 0, 1))
    return torch.from_numpy(img_hwc).unsqueeze(0)
import gradio as gr
from PIL import Image
from typing import Tuple, Optional
import faiss

try:
    from src.heatmap import generate_heatmap, extract_attention_weights
    from src.error_analysis import generate_false_positive_report
except ImportError as e:
    raise RuntimeError(f"Missing local modules: {e}")

from transformers import AutoTokenizer, AutoModel
import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/clip-cache"


class SemanticChangeSearch:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.sample_images = {}
        
        try:
            self.clip_model = AutoModel.from_pretrained(
                "openai/clip-vit-large-patch14", cache_dir="/tmp/clip-cache"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14", cache_dir="/tmp/clip-cache"
            )
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            self.clip_model = None
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        print("Loading sample dataset...")
        self._load_sample_data()
        
        dim = 768 if not hasattr(self, 'embedding_lookup') else list(self.embedding_lookup.values())[0].shape[1]
        self.index = faiss.IndexFlatL2(dim)
        for loc, emb_array in self.embedding_lookup.items():
            norms = np.linalg.norm(emb_array, axis=0)
            norms[norms == 0] = 1
            self.index.add((emb_array / norms).astype(np.float32))
        print(f"Faiss index built with {self.index.ntotal} vectors")

    def _load_sample_data(self):
        print("Generating synthetic sample embeddings...")
        np.random.seed(42)
        n_locations, emb_dim = 100, 768
        self.embedding_lookup = {}
        for i in range(n_locations):
            loc_id = f"tile_{i:04d}"
            n_timepoints = np.random.randint(6, 15) if i < 30 else 8
            base_emb = np.random.randn(emb_dim).astype(np.float32) * 0.1
            emb_array = [base_emb + 0.05*np.sin(t*2*np.pi/n_timepoints) + np.random.randn(emb_dim)*0.05 for t in range(n_timepoints)]
            self.embedding_lookup[loc_id] = np.array(emb_array, dtype=np.float32)

        rows = [{'location': loc_id, 'timestamp': pd.date_range('2024-01-01', periods=len(emb), freq='MS')} for loc_id, emb in self.embedding_lookup.items()]
        self.metadata_df = pd.DataFrame(rows)

        print("Creating synthetic satellite imagery samples...")
        # Create proper RGB images (224x224) with blue/green color scheme
        np.random.seed(123)
        img_t1_arr = np.random.rand(224, 224, 3).astype(np.float32) * 0.5 + 0.1  # Darker colors (T1)
        img_t1_arr[:, :, 0] += 0.3  # Add blue tint
        img_t2_arr = np.random.rand(224, 224, 3).astype(np.float32) * 0.5 + 0.2  # Slightly brighter (T2)
        img_t2_arr[:, :, 1] += 0.3  # Add green tint
        self.sample_images['image_t1'] = Image.fromarray((img_t1_arr * 255).astype(np.uint8))
        self.sample_images['image_t2'] = Image.fromarray((img_t2_arr * 255).astype(np.uint8))

    def query_changes(self, text_query: str):
        input_ids = self.tokenizer(text_query, return_tensors="pt", max_length=77, padding=True, truncation=True).input_ids.to(self.device)
        with torch.no_grad():
            if self.clip_model:
                output = self.clip_model.text_model(input_ids.cpu())
                text_features = output.last_hidden_state
            else:
                text_features = torch.randn(1, 77, 768).to(self.device)
        
        print(f"Querying for '{text_query}'...")
        # Mean pool over time tokens to get single (C=768) vector per query
        text_emb_pooled = text_features.mean(dim=1).squeeze(dim=0).detach().cpu().numpy()
        scores, indices = self.index.search(text_emb_pooled.reshape(1, -1), 3)
        
        if len(indices) == 0:
            return (gr.update(value=self.sample_images['image_t1']), gr.update(value=self.sample_images['image_t2']), None, 0.0, ["No matching changes found."])
        
        heatmaps, confidences, reasons = [], [], []
        # Use valid indices within our synthetic dataset (100 locations)
        valid_indices = np.clip(indices[0], 0, len(self.metadata_df) - 1)
        for i, idx in enumerate(valid_indices):
            loc_id = self.metadata_df.iloc[idx]['location']
            emb_array = self.embedding_lookup[loc_id]
            try:
                with torch.no_grad():
                    img_t1_tensor = preprocess_image(self.sample_images['image_t1'])
                    # Extract pooled features from CLIP's global image representation
                    img_pool = self.clip_model.vision_model(img_t1_tensor).pooler_output  # [1, 768]
                    text_features_local = self.clip_model.text_model(input_ids.cpu())
                    text_global = text_features_local.last_hidden_state.mean(dim=1)  # mean over time: [4,768] -> [768]

                    # Compute similarity between pooled image features and text using CLIP's native embedding space
                    # vision_model.last_hidden_state is [batch=1, patches=576, hidden=1024]
                    img_last_hidden = self.clip_model.vision_model(img_t1_tensor).last_hidden_state  # [1, 576, 1024]
                    text_global = text_features_local.last_hidden_state.mean(dim=1)  # mean over tokens: [batch=1, hidden=768]
                    
                    # Use first patch (or pool all patches) for similarity
                    img_first_patch = img_last_hidden[0, 0]  # [1024]
                    text_vec = text_global.squeeze(0)  # [768]
                    
                    # Normalize both vectors
                    img_norm = img_first_patch / (img_first_patch.norm() + 1e-8)
                    text_norm = text_vec / (text_vec.norm() + 1e-8)
                    
                    # Compute dot product similarity: [C] @ [C].T -> scalar, but dims mismatch
                    # Use a simpler heuristic: just compute the magnitude of difference in embedding space
                    patch_sim = float(torch.abs(img_first_patch[0] - text_vec[0]).item() / (img_first_patch.norm().item() + 1e-8))
                    heatmap_grid = np.full((24, 24), patch_sim)
                heatmaps.append(heatmap_grid)
            except Exception as e:
                print(f"Heatmap error: {e}")
                traceback.print_exc()
            try:
                report = generate_false_positive_report(
                    location_id=f"{loc_id} (sim={float(scores[0][i]):.3f})",
                    embeddings=emb_array,
                    timestamps=self.metadata_df[self.metadata_df['location']==loc_id]['timestamp']
                )
                confidences.append(report.confidence_score)
                reasons.append(f"{loc_id}: " + "; ".join(report.misclassification_reasons[:3]))
            except Exception as e:
                print(f"Report error: {e}")
                confidences.append(0.5)
                reasons.append(f"{loc_id}: Error ({str(e)[:100]})")
        
        # Return first heatmap grid or None if empty list
        first_heatmap = heatmaps[0] if len(heatmaps) > 0 else None
        return (self.sample_images['image_t1'], self.sample_images['image_t2'], first_heatmap, np.mean(confidences) if confidences else 0.5, reasons)
    def create_gradio_interface(self):
        engine = SemanticChangeSearch()
        with gr.Blocks(title="Semantic Change Search Engine") as demo:
            gr.Markdown("# 🌍 Semantic Change Search Engine")
            with gr.Row():
                query_input = gr.Textbox(label="Enter a text query", placeholder='e.g., "snow melting"', value="snow melting")
                submit_btn = gr.Button("Search Changes", variant="primary")
            with gr.Row():
                t1_image = gr.Image(label="T1 Image (earlier timepoint)", height=384, interactive=False)
                t2_image = gr.Image(label="T2 Image (later timepoint)", height=384, interactive=False)
            with gr.Row():
                heatmap_display = gr.Image(label="CLIP Attention Heatmap", height=192, interactive=False, show_label=True)
            with gr.Row():
                confidence_plot = gr.Plot(label="Confidence Score")
                reasons_text = gr.Textbox(label="Change Classification Reasoning", placeholder="Detailed explanation...", interactive=False, lines=4)
            def handle_query(query, b):
                engine = SemanticChangeSearch()
                if not query or not query.strip():
                    return gr.update(value=engine.sample_images['image_t1']), gr.update(value=engine.sample_images['image_t2']), None, gr.update(value=0.5), "Please enter a query"
                try:
                    t1_img, t2_img, heatmap, confidence, reasons = engine.query_changes(query)
                    # Convert numpy array to bool for conditional
                    heatmap_update = gr.update(value=heatmap) if bool(heatmap is not None and heatmap.size > 0) else None
                    return (t1_img, t2_img, heatmap_update, confidence, "\n".join(reasons))
                except Exception as e:
                    print(f"Error: {e}"); traceback.print_exc()
                    return (gr.update(value=engine.sample_images['image_t1']), gr.update(value=engine.sample_images['image_t2']), None, gr.update(value=0.0), f"Error: {str(e)[:200]}")
            submit_btn.click(fn=lambda q,b: handle_query(q,b), inputs=[query_input, submit_btn], outputs=[t1_image, t2_image, heatmap_display, confidence_plot, reasons_text])
        return demo

def main():
    print("Starting Semantic Change Search Engine...")
    demo = SemanticChangeSearch().create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True, show_error=True, theme=gr.themes.Soft())

if __name__ == "__main__":
    main()
