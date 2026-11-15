"""Rebuild the published p50c edit on weights/base with ../code/apply_edit.py (it checks the sha256 of every base and
rebuilt tensor against the evaluated checkpoint) -> weights/their_edit.   python replay_edit.py"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True   # keep ../code free of __pycache__
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import apply_edit  # noqa: E402

BASE, OUT = os.path.join(HERE, "weights", "base"), os.path.join(HERE, "weights", "their_edit")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(BASE):
        if f != "model.safetensors":
            shutil.copy(os.path.join(BASE, f), OUT)
    apply_edit.main(os.path.join(BASE, "model.safetensors"), os.path.join(HERE, "..", "runs", "p50c", "edit"),
                    os.path.join(OUT, "model.safetensors"))
