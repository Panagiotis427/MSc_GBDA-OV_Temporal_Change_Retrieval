"""Deploy the app to its HuggingFace Space (slim, Xet-safe).

Plain ``git push`` to the Space is rejected by HF because the repo history
contains binaries (models/*.pt, demos/*.mp4, figures). This uploads only what the
Space needs, via the ``huggingface_hub`` API, which stores binaries through
Xet/LFS automatically. It ships:

  * ``app.py`` (Space entry point), ``src/`` (the package), ``requirements.txt``
  * ``tests/fixtures/den_tiny`` (the bundled demo corpus the Space runs on)
  * ``scripts/space_readme.md`` uploaded as the Space's ``README.md`` — its YAML
    front-matter is the Space config, kept separate so the GitHub README stays
    front-matter-free

and deliberately EXCLUDES the ~44 MB ``models/`` + ``demos/`` (Space-irrelevant),
the repo ``README.md``, ``report/``, ``data/``, ``local/``, tests, and
``pyproject.toml`` (its cu128 torch pins would break HF's build — the Space
installs plain ``requirements.txt``).

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
    "src/*",             # fnmatch '*' spans '/', so this is recursive
    "tests/fixtures/*",
]


def main() -> None:
    api = HfApi()
    who = api.whoami().get("name", "?")
    print(f"Deploying {REPO_ID} as {who} (slim: code + fixture)…")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=".",
        allow_patterns=ALLOW,
        delete_patterns=["*"],  # clean slate — drop stale files not re-uploaded
        commit_message="Deploy current UI (code + fixture; gradio 6.18)",
    )
    # The Space's landing page = its own front-matter config + a concise app blurb,
    # uploaded as README.md so the GitHub repo README stays front-matter-free.
    api.upload_file(
        path_or_fileobj="scripts/space_readme.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Space landing page (front-matter config + app description)",
    )
    print(f"Done. Watch the build: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
