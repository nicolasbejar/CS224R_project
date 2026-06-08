"""Standalone Modal launcher for the real-LLM-judge experiment.

This mirrors the image/volume setup of the project's ``modal_train.py`` but is
fully self-contained inside the extension package, so it does not touch any
default-project file. It runs ``offline_real_judge.py`` with the real vLLM judge
on a single GPU and commits the result JSON + figures back to the Modal volume.

Run from the repository root:

    modal run maxrl_trainer/extension/experiments/real_judge_modal.py \
        --judge-model Qwen/Qwen2.5-7B-Instruct

Then download the artifacts from the volume (they are written under
``/vol/extension_figures``):

    modal volume get default-proj-training extension_figures ./real_judge_out
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal

# Resolve the local project root only when running locally (image build time).
# On the remote container this file lives at /root/ with fewer parents, so guard
# the lookup and fall back to the remote project root.
_THIS = Path(__file__).resolve()
LOCAL_PROJECT_ROOT = _THIS.parents[3] if len(_THIS.parents) > 3 else _THIS.parent
REMOTE_ROOT = "/root/default_proj"
REMOTE_VOL = "/vol"
REMOTE_REQS = REMOTE_ROOT + "/modal_requirements.txt"

APP_NAME = os.environ.get("MODAL_APP_NAME", "default-proj-training")
GPU_CONFIG = os.environ.get("MODAL_GPU", "H100!")
TIMEOUT_SECONDS = int(os.environ.get("MODAL_TIMEOUT_SECONDS", "3600"))
STARTUP_TIMEOUT_SECONDS = int(os.environ.get("MODAL_STARTUP_TIMEOUT_SECONDS", "1800"))
CPU_COUNT = int(os.environ.get("MODAL_CPU_COUNT", "8"))
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "default-proj-training")
PIP_EXTRA_INDEX_URL = os.environ.get(
    "MODAL_PIP_EXTRA_INDEX_URL", "https://download.pytorch.org/whl/cu128"
)

VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _secrets() -> list[modal.Secret]:
    vals = {k: os.environ[k] for k in ("HF_TOKEN",) if os.environ.get(k)}
    return [modal.Secret.from_dict(vals)] if vals else []


image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(str(LOCAL_PROJECT_ROOT), remote_path=REMOTE_ROOT, copy=True)
    .run_commands(
        "cd " + shlex.quote(REMOTE_ROOT) + " && python -m pip install --upgrade "
        "pip==25.3 setuptools==80.10.2 wheel==0.46.3",
        "cd " + shlex.quote(REMOTE_ROOT) + " && python -m pip install "
        f"--extra-index-url {shlex.quote(PIP_EXTRA_INDEX_URL)} "
        f"-r {shlex.quote(REMOTE_REQS)}",
        "cd " + shlex.quote(REMOTE_ROOT) + " && python -m pip install --no-deps -e .",
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    timeout=TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={REMOTE_VOL: VOLUME},
    secrets=_secrets(),
)
def run_real_judge(judge_model: str, extra_args: list[str]) -> str:
    out_dir = Path(REMOTE_VOL) / "extension_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    hf_home = Path(REMOTE_VOL) / "cache" / "huggingface"
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    command = [
        "python",
        "maxrl_trainer/extension/experiments/offline_real_judge.py",
        "--judge-backend", "vllm",
        "--judge-model", judge_model,
        "--out-dir", str(out_dir),
        *extra_args,
    ]
    print(f"Executing on Modal: {shlex.join(command)}")
    try:
        subprocess.run(command, cwd=REMOTE_ROOT, env=env, check=True)
    finally:
        VOLUME.commit()
    return f"Real-judge run complete. Artifacts in volume '{VOLUME_NAME}' under {out_dir}."


@app.local_entrypoint()
def main(judge_model: str = "Qwen/Qwen2.5-7B-Instruct", quick: bool = False) -> None:
    extra_args = ["--quick"] if quick else []
    print(run_real_judge.remote(judge_model, extra_args))
