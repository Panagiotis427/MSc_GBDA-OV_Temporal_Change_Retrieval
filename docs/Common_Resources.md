**Open Vocabulary Temporal Change Retrieval**

Complete Resource Reference

*Datasets · Models · Download Links*

# **1. Datasets** {#datasets}

## **1.1 QFabric** {#qfabric}

A multi-task change detection dataset with 450,000 change polygons annotated across 504 locations in 100 cities, covering 6 change types and 9 change status classes. Presented at CVPR EarthVision 2021 by Verma et al. (Granular AI).

| **Resource** | **Link** |
|---|---|
| Paper (CVPR EarthVision 2021) | [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.pdf) |
| IEEE Xplore | [ieeexplore.ieee.org](https://ieeexplore.ieee.org/document/9523090/) |
| arXiv / HAL | [hal.science/hal-03294534](https://hal.science/hal-03294534) |
| Project Page (Granular AI) | [granular.ai](https://www.granular.ai/resources/blog/qfabric:-multi-task-change-detection-dataset) |
| Download — **`labaerien/qfabric`** (HF, **gated**; 5-date rasters + COCO polygon masks — the form we use) | [huggingface.co/datasets/labaerien/qfabric](https://huggingface.co/datasets/labaerien/qfabric) |
| Download — Granular AI Engine (free account) | [engine.granular.ai](https://engine.granular.ai/organizations/granular/projects/631e0974b59aa3b615b0d29a/overview) |
| Reduced form already in-repo (TEOChatlas, 2-date crops, `qfabric_teo`) | [huggingface.co/datasets/jirvin16/TEOChatlas](https://huggingface.co/datasets/jirvin16/TEOChatlas) |
| ⚠ AVOID — `EVER-Z/QFabric_mt_images_1024` | 298 GB, image-only (no polygon masks); out of scope by size |

*⚠ The original project page (sagarverma.github.io/qfabric) is a 404. The full pentatemporal +
polygon-mask plan (the committed future work) is specified in
[`QFABRIC_FUTURE_WORK.md`](QFABRIC_FUTURE_WORK.md).*

## **1.2 Dynamic EarthNet** {#dynamic-earthnet}

Daily multi-spectral satellite observations of 75 areas of interest worldwide, using Planet Labs imagery. Pairs daily observations with pixel-wise monthly semantic segmentation labels across 7 LULC classes. Published at CVPR 2022 by Toker et al.

| **Resource**                                            | **Link / Command**                                                                                                                                 |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Paper (CVPR 2022)                                       | [[arxiv.org/abs/2203.12560]{.underline}](https://arxiv.org/abs/2203.12560)                                                                         |
| GitHub (Official Implementation)                        | [[github.com/aysim/dynnet]{.underline}](https://github.com/aysim/dynnet)                                                                           |
| Download --- Official TUM Mediatum (\~525 GB raw)       | [[mediatum.ub.tum.de/1650201]{.underline}](https://mediatum.ub.tum.de/1650201)                                                                     |
| Download --- HuggingFace HEVC-compressed (much smaller) | [[huggingface.co/datasets/tacofoundation/DynamicEarthNet-video]{.underline}](https://huggingface.co/datasets/tacofoundation/DynamicEarthNet-video) |
| Download --- Preprocessed via gdown (\~7 GB)            | gdown 1cMP57SPQWYKMy8X60iK217C28RFBkd2z                                                                                                            |
| Train/Val/Test Splits                                   | [[github.com/aysim/dynnet]{.underline}](https://github.com/aysim/dynnet)                                                                           |
| Dataset Hugging Face                                    | [https://huggingface.co/datasets/torchgeo/dynamic_earthnet]{.underline}                                                                            |

*💡 For storage-constrained setups, the HuggingFace HEVC-compressed version or the \~7 GB gdown preprocessed version are the most practical starting points.*

## **1.3 fMoW --- Functional Map of the World** {#fmow-functional-map-of-the-world}

Over 1 million images from 200+ countries with temporal sequences, targeting functional land-use classification across 63 categories. Originally an IARPA challenge dataset; published at CVPR 2018 by Christie et al.

| **Resource**                                    | **Link / Command**                                                                                                                       |
|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Paper (CVPR 2018)                               | [[arxiv.org/abs/1711.07846]{.underline}](https://arxiv.org/abs/1711.07846)                                                               |
| GitHub --- Dataset info & manifest              | [[github.com/fMoW/dataset]{.underline}](https://github.com/fMoW/dataset)                                                                 |
| GitHub --- Baseline code                        | [[github.com/fMoW/baseline]{.underline}](https://github.com/fMoW/baseline)                                                               |
| SpaceNet Page                                   | [[spacenet.ai/iarpa-functional-map-of-the-world-fmow]{.underline}](https://spacenet.ai/iarpa-functional-map-of-the-world-fmow/)          |
| Download --- fMoW-rgb (\~200 GB, via AWS CLI)   | aws s3 sync s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb .                                                                        |
| Download --- fMoW-full (\~3.5 TB, via AWS CLI)  | aws s3 sync s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-full .                                                                       |
| fMoW-Sentinel (Sentinel-2 version, HuggingFace) | [[huggingface.co/datasets/jonathan-roberts1/fMoW-Sentinel]{.underline}](https://huggingface.co/datasets/jonathan-roberts1/fMoW-Sentinel) |
| fMoW-Sentinel (Stanford Digital Repository)     | [[purl.stanford.edu/vg497cb6002]{.underline}](https://purl.stanford.edu/vg497cb6002)                                                     |

*💡 fMoW-rgb (\~200 GB JPEG) is recommended over fMoW-full (\~3.5 TB TIFF) for most research purposes. Both are hosted free on AWS S3. The fMoW-Sentinel variant on HuggingFace pairs locations with Sentinel-2 imagery.*

# **2. Models** {#models}

## **2.1 CLIP (OpenAI)** {#clip-openai}

The foundational contrastive vision-language model by Radford et al. (2021). Pre-trained on 400M image-text pairs. Serves as the backbone for both RemoteCLIP and GeoRSCLIP.

| **Resource**                         | **Link**                                                                                                           |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Paper (Radford et al., 2021)         | [[arxiv.org/abs/2103.00020]{.underline}](https://arxiv.org/abs/2103.00020)                                         |
| GitHub (OpenAI Official)             | [[github.com/openai/CLIP]{.underline}](https://github.com/openai/CLIP)                                             |
| HuggingFace Model Hub                | [[huggingface.co/openai/clip-vit-large-patch14]{.underline}](https://huggingface.co/openai/clip-vit-large-patch14) |
| OpenCLIP (community, multi-backbone) | [[github.com/mlfoundations/open_clip]{.underline}](https://github.com/mlfoundations/open_clip)                     |

*💡 The OpenCLIP library provides a unified interface for loading CLIP, RemoteCLIP, and GeoRSCLIP checkpoints with a consistent API --- ideal for the dual-stream encoder setup described in the project.*

## **2.2 RemoteCLIP** {#remoteclip}

The first vision-language foundation model specifically designed for remote sensing. Fine-tuned on remote sensing image-text data; outperforms vanilla CLIP by up to 6.39% on zero-shot classification and 9.14% mean recall on retrieval benchmarks. Published in IEEE TGRS.

| **Resource**           | **Link**                                                                                           |
|------------------------|----------------------------------------------------------------------------------------------------|
| Paper (IEEE TGRS)      | [[arxiv.org/abs/2306.11029]{.underline}](https://arxiv.org/abs/2306.11029)                         |
| GitHub (Official)      | [[github.com/ChenDelong1999/RemoteCLIP]{.underline}](https://github.com/ChenDelong1999/RemoteCLIP) |
| HuggingFace Paper Page | [[huggingface.co/papers/2306.11029]{.underline}](https://huggingface.co/papers/2306.11029)         |

## **2.3 GeoRSCLIP** {#georsclip}

CLIP fine-tuned on RS5M, a large-scale remote sensing vision-language dataset assembled by Om AI Lab. Improves over prior state-of-the-art by 4--5% on Semantic Localization (SeLo) tasks. Supports LoRA, Pfeiffer, Prefix Tuning, and UniPELT PEFT methods.

| **Resource**                    | **Link**                                                                               |
|---------------------------------|----------------------------------------------------------------------------------------|
| Paper (RS5M + GeoRSCLIP)        | [[arxiv.org/abs/2306.11300]{.underline}](https://arxiv.org/abs/2306.11300)             |
| GitHub (Official --- RS5M repo) | [[github.com/om-ai-lab/RS5M]{.underline}](https://github.com/om-ai-lab/RS5M)           |
| HuggingFace Model               | [[huggingface.co/Zilun/GeoRSCLIP]{.underline}](https://huggingface.co/Zilun/GeoRSCLIP) |

# **3. Implementation Notes**

This file is a **download-link / resource reference only**. The architecture (frozen dual-stream
encoder, Δf change feature, cosine retrieval, Recall@K/mAP evaluation, Gradio deliverable) is
documented once in [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`../README.md`](../README.md) — not
duplicated here.
