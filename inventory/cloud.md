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
- **LEVIR-CC**: official release; loader + benchmark in-repo
- fMoW (optional future): same-source via TEOChatlas

## Related

- Thesis sibling repo consumes THIS repo at tag `v1.0` (library)
