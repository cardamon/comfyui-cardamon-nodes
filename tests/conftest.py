"""Make ComfyUI's modules (comfy_api, comfy, ...) importable for tests.

Set COMFYUI_PATH to the root of a ComfyUI checkout, e.g.
    COMFYUI_PATH=~/ComfyUI pytest
"""

import os
import sys
from pathlib import Path

import pytest

_comfyui_path = os.environ.get("COMFYUI_PATH")
if not _comfyui_path or not (Path(_comfyui_path).expanduser() / "comfy_api").is_dir():
    pytest.exit("Set COMFYUI_PATH to the root of a ComfyUI checkout to run the tests.", returncode=4)

sys.path.insert(0, str(Path(_comfyui_path).expanduser().resolve()))
