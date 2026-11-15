"""Download Qwen/Qwen3.5-0.8B (the revision code/apply_edit.py names) and build weights/base, the text-only model the
experiment edited: the language-model tensors of the release (names unchanged, model.language_model.*) with the
config files of ../base_model/Qwen3.5-0.8B-text.   python download.py"""
import os
import shutil
import sys

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True   # keep ../code free of __pycache__
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import st_numpy as S  # noqa: E402

REPO, REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
TEXT = os.path.join(HERE, "..", "base_model", "Qwen3.5-0.8B-text")
HF, BASE = os.path.join(HERE, "weights", "hf"), os.path.join(HERE, "weights", "base")

if __name__ == "__main__":
    snapshot_download(REPO, revision=REV, local_dir=HF, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
    os.makedirs(BASE, exist_ok=True)
    for f in os.listdir(TEXT):
        shutil.copy(os.path.join(TEXT, f), BASE)
    shutil.copy(os.path.join(HF, "tokenizer.json"), BASE)
    src = os.path.join(HF, "model.safetensors-00001-of-00001.safetensors")
    names = [n for n in S.read_header(src)[0] if n.startswith("model.language_model.")]
    S.save(os.path.join(BASE, "model.safetensors"), S.load(src, names), {"format": "pt"})
    print(f"{len(names)} language-model tensors -> {BASE}")
