# Project Implementation Plan: Semantic Change Search Engine
**Target Executor:** Qwen 3.5 9B Coding Agent
**Project Goal:** Open-Vocabulary Temporal Change Retrieval using frozen Vision-Language Models.

## General Agent Directives (System Instructions)
* **Modularity:** Write small, independent Python modules (`data.py`, `model.py`, `train.py`, `app.py`).
* **Compute Constraint:** Assume limited GPU memory. Prioritize frozen backbones and lightweight projection heads.
* **Documentation:** Add docstrings to all functions describing input shapes and tensor dimensions.

---

## Week 1: Infrastructure & Data Ingestion
**Focus:** Scaffold the repository and set up the foundation for handling multitemporal data.

* **Task 1.1: Project Setup.** Create a basic repository structure (`src/`, `data/`, `notebooks/`, `tests/`) and a `requirements.txt` containing `torch`, `transformers`, `gradio`, `faiss-cpu`, `mlflow`, `pandas`, `numpy`, and `opencv-python`.
* **Task 1.2: Embedding Loader.** Write a data utility script (`src/data_loader.py`) to load pre-computed QFabric or Dynamic EarthNet features. The script must parse metadata to extract geographic coordinates and timestamps.
* **Task 1.3: Temporal Pairing Logic.** Implement a function that groups image embeddings by location and sorts them temporally to yield `(T1, T2)` pairs for the same spatial tile.

**Acceptance Criteria:**
* `src/data_loader.py` can successfully return a PyTorch dataset of paired `(embedding_T1, embedding_T2)` vectors.

---

## Week 2: Change Feature Engineering & Text Processing
**Focus:** Calculate the change representation and prepare text embeddings.

* **Task 2.1: Feature Difference.** In `src/features.py`, implement the logic to compute the "Change Feature". Calculate the difference vector: `delta_f = f_T2 - f_T1` (where `f` represents the embedding). Also, provide an option for concatenation `[f_T1, f_T2]`.
* **Task 2.2: Text Embedding Pipeline.** Write a function using the HuggingFace `transformers` library to load a frozen text encoder (e.g., CLIP text model) and convert natural language change queries (e.g., "new industrial buildings") into embeddings.
* **Task 2.3: Unit Tests.** Write simple `pytest` functions to ensure `delta_f` tensor shapes match expected dimensions (e.g., `[batch_size, embedding_dim]`).

**Acceptance Criteria:**
* Agent outputs a functioning text encoder script and a robust temporal difference calculator with passing unit tests.

---

## Week 3: Model Architecture (The Projection Head)
**Focus:** Define the trainable parameters of the system.

* **Task 3.1: Adapter Network.** Create `src/model.py`. Define a PyTorch `nn.Module` for the lightweight Projection Head (a multi-layer perceptron with dropout and LayerNorm) that takes the Change Feature (`delta_f`) and maps it to a common multimodal space.
* **Task 3.2: InfoNCE Loss.** Implement the InfoNCE (Contrastive) Loss function. The loss should maximize cosine similarity between the projected change feature and the text embedding of the true description, while minimizing similarity with negative descriptions in the batch.

**Acceptance Criteria:**
* A complete PyTorch model class that can execute a forward pass without shape mismatch errors.

---

## Week 4: Training Loop & Evaluation Setup
**Focus:** Model training and retrieval metric calculation.

* **Task 4.1: Training Script.** Write `src/train.py`. Implement the standard PyTorch training loop (optimizer, loss calculation, backpropagation) for the Projection Head.
* **Task 4.2: MLflow Integration.** Add basic MLflow logging to track loss, learning rate, and epochs.
* **Task 4.3: Evaluation Metrics.** Implement a function to calculate `Recall@K` and `Mean Average Precision (mAP)`. The agent must write a script that ranks all temporal pairs in the validation set against a given set of text queries.

**Acceptance Criteria:**
* The training script runs end-to-end on CPU/GPU and logs metrics to a local MLflow directory. Validation script outputs `Recall@10`.

---

## Week 5: Spatial Heatmap Generation (Core CV Task)
**Focus:** Explainability and visual localization of the change.

* **Task 5.1: Attention/Activation Extraction.** *This is complex for an agent; provide strict guidance.* Instruct the agent to write a script (`src/heatmap.py`) that extracts patch-level embeddings from the vision backbone (if using full images) or uses a similarity gradient method to identify which spatial regions contribute most to the cosine similarity with the text prompt.
* **Task 5.2: Image Overlay.** Write an OpenCV/Matplotlib utility to resize the resulting 2D attention map, apply a color map (e.g., `cv2.COLORMAP_JET`), and overlay it transparently onto the `T2` image.

**Acceptance Criteria:**
* A function `generate_heatmap(img_T1, img_T2, text_query, model)` that returns a NumPy array representing the blended heatmap image.

---

## Week 6: Semantic Drift & Error Analysis
**Focus:** Qualitative evaluation and robust testing.

* **Task 6.1: Edge-Case Dataset.** Instruct the agent to write a script to filter the dataset for known seasonal changes (e.g., "snow melting", "leaves falling") vs. permanent changes (e.g., "construction", "deforestation").
* **Task 6.2: Analysis Script.** Run the model against these edge cases and generate a CSV report of False Positives (where the model confused seasonal drift for structural change).
* **Task 6.3: Prompt Engineering.** Experiment with adding negative prompts (e.g., "ignore snow", "ignore seasonal vegetation") to the text encoder to see if zero-shot performance improves.

**Acceptance Criteria:**
* A documented Jupyter Notebook (`notebooks/error_analysis.ipynb`) summarizing the semantic drift findings.

---

## Week 7: Gradio Frontend Development
**Focus:** Building the user interface.

* **Task 7.1: UI Layout.** Create `app.py`. Instruct the agent to build a Gradio Blocks interface featuring:
    * A text input box for the query.
    * A search button.
    * A Gallery or custom HTML output to display results.
* **Task 7.2: Backend Integration.** Connect the search button to the Faiss index/model inference pipeline.
* **Task 7.3: Result Formatting.** For each top-K result, the UI must display: `T1 Image` (left), `T2 Image` (middle), `Heatmap` (right), and a `Confidence Score` (text).

**Acceptance Criteria:**
* `python app.py` launches a local web server with a fully functional UI that queries the trained model and displays side-by-side images with heatmaps.

---

## Week 8: Dockerization & Delivery
**Focus:** Packaging the application.

* **Task 8.1: Dockerfile.** Write a `Dockerfile` that packages the Gradio app, the PyTorch models, and the Faiss index.
* **Task 8.2: Entrypoint.** Ensure the Docker container exposes the correct Gradio port (e.g., 7860) and runs the app on startup.
* **Task 8.3: README.** Instruct the agent to generate a comprehensive `README.md` containing architecture diagrams (text-based or mermaid), setup instructions, and examples of queries that work well.

**Acceptance Criteria:**
* The system can be started entirely via `docker build` and `docker run` commands.
