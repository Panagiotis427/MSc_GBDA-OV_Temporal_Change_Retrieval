# Cloud manifest — remotes, archives, download-pointers

*Half-static, hand-maintained (tiny). Last update: 2026-06-10.*

## Git remotes

- `origin` = GitHub `Panagiotis427/MSc_GBDA-OV_Temporal_Change_Retrieval` (source of truth)
- `space` = HuggingFace Space (deployed Gradio engine; push `main` → redeploy)

## HF

- Space: the live Semantic Change Search Engine (course deliverable product)
- Cache models used locally: GeoRSCLIP (`Zilun/GeoRSCLIP`); RemoteCLIP + CLIP ViT-L/14 under repo
  `.model_cache/` (see machine manifests)
- Hub private repos: none yet (cold-archive option if artifacts outgrow git)

## Download-pointers (canonical sources — never redistributed)

- **Dynamic EarthNet**: official source (earthnet.tech) — local copy in `data/` (laptop manifest)
- **QFabric**: via `jirvin16/TEOChatlas` parsed subsets (`qfabric_teo`, `qfabric_status`)
- **LEVIR-CC / LEVIR-MCI**: `lcybuaa/LEVIR-MCI` (Zenodo/HF) is a strict superset of LEVIR-CC
  (identical pairs + captions, plus change masks) — stored once under `data/_levir_mci/`; both the
  `levir_cc` and `levir_mci` loaders read it (no duplicate LEVIR-CC copy)
- **SECOND-CC**: Zenodo `10.5281/zenodo.16937571` (`SECOND-CC-AUG.zip`, CC-BY-4.0, public); loader + benchmark in-repo
- fMoW (optional future): same-source via TEOChatlas
