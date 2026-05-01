# EXAMPLE CASE #11: OPEN-VOCABULARY TEMPORAL CHANGE RETRIEVAL

## General Description

Traditional change detection is often limited to a few predefined categories (e.g., "urban growth" or "forest loss"). This project challenges students to move toward **open-vocabulary change detection** by leveraging Vision-Language Models (VLMs). Instead of training a model for a specific class, students will build a system that identifies complex, semantically defined transitions based on natural language queries (e.g., "new industrial buildings appearing on former agricultural land" or "coastal erosion following a storm event"). By aligning text and image-change embeddings, the system will search across large multitemporal datasets to find the specific tiles and time-steps where the described change occurred.

## Data Guidelines

To ensure feasibility on limited compute, students should utilize datasets that offer either pre-computed features or manageable temporal stacks:

*   **QFabric:** Highly recommended for its pre-computed foundation model embeddings, which allow students to bypass the heavy feature extraction phase.
*   **Dynamic EarthNet:** Provides daily, multi-spectral Planet Fusion data, ideal for testing the model's ability to pinpoint exact time-steps of change.
*   **fMoW (Functional Map of the World):** The temporal subset of fMoW can be used to identify functional land-use changes over several years across diverse global locations.

## Model Suggestions

The architecture should prioritize **frozen backbones** to minimize GPU requirements.

*   Students can utilize a dual-stream encoder (e.g., a frozen CLIP, GeoRSCLIP, or RemoteClip) to extract embeddings for two different time-steps (T1 and T2).
*   The "Change Feature" can be represented as the difference vector (Δf = f_T2 - f_T1) or a concatenated representation passed through a lightweight, trainable **Linear Adapter** or **Projection Head**.
*   The system will then calculate the cosine similarity between this change embedding and the text embedding generated from the user's natural language query.

## Training & Evaluation

Because the project aims for low-compute feasibility, the "training" phase should focus on **Zero-Shot** inference or **Parameter-Efficient Fine-Tuning (PEFT)**.

*   If labels are available (e.g., in Dynamic EarthNet), students can evaluate the system using **Retrieval Metrics** such as **Recall@K** and **Mean Average Precision (mAP)**.
*   Evaluation should verify if the model correctly identifies the specific temporal window where the change was most prominent compared to "stable" time-steps.
*   Students should perform an error analysis on "semantic drift" where the model might confuse seasonal vegetation changes with permanent land-cover transitions.

## Product Delivery

The final deliverable should be a **Semantic Change Search Engine** (e.g., built with Gradio).

*   The interface will feature a text input box where the user describes a transition.
*   Upon execution, the system will scan a database of image tiles and return a **ranked list of "change events."**
*   Each result should display the T1 and T2 image pair side-by-side, a heatmap highlighting the specific spatial region of the change, and a confidence score reflecting how well the visual transition aligns with the textual description.
