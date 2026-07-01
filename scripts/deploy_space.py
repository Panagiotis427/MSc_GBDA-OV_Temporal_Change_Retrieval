"""Deploy the app to its HuggingFace Space (slim, Xet-safe).

Plain ``git push`` to the Space is rejected by HF because the repo history
contains binaries (models/*.pt, demos/*.mp4, figures). This uploads only what the
Space needs, via the ``huggingface_hub`` API, which stores binaries through
Xet/LFS automatically. It ships:

  * ``app.py`` (Space entry point), ``src/`` (the package), ``requirements.txt``
  * ``tests/fixtures/den_tiny`` (the bundled demo corpus the Space runs on)
  * ``report/figures/`` (so the README "Technical report" images resolve)

and deliberately EXCLUDES the ~44 MB ``models/`` + ``demos/`` (Space-irrelevant),
``data/``, ``local/``, tests, ``main.tex``, and ``pyproject.toml`` (its cu128 torch
pins would break HF's build — the Space installs plain ``requirements.txt``).

``delete_patterns=["*"]`` gives the Space a clean slate (stale files not re-uploaded
are removed in the same commit).

Auth: uses your cached ``huggingface-cli login`` token (or the ``HF_TOKEN`` env var).

Run from the repo root:
    python scripts/deploy_space.py
"""
from __future__ import annotations

from huggingface_hub import HfApi

REPO_ID = "panagiotis427/Open_Vocabulary_Temporal_Change_Retrieval"

ALLOW = [
    "app.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "src/*",             # fnmatch '*' spans '/', so this is recursive
    "tests/fixtures/*",
    "report/figures/*",
]


def main() -> None:
    api = HfApi()
    who = api.whoami().get("name", "?")
    print(f"Deploying {REPO_ID} as {who} (slim: code + fixture + figures)…")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=".",
        allow_patterns=ALLOW,
        delete_patterns=["*"],  # clean slate — drop stale files not re-uploaded
        commit_message="Deploy current UI (code + fixture + figures; gradio 6.18)",
    )
    print(f"Done. Watch the build: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
